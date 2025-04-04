#include "bindings.h"
#include "helpers.cuh"
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include "utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * Quat-Scale to Covariance and Precision
 ****************************************************************************/

__global__ void
quat_scale_to_covar_preci_fwd_kernel(const uint32_t N,
                                     const float *__restrict__ quats,  // [N, 4]
                                     const float *__restrict__ scales, // [N, 3]
                                     const bool triu,
                                     // outputs
                                     float *__restrict__ covars, // [N, 3, 3] or [N, 6]
                                     float *__restrict__ precis  // [N, 3, 3] or [N, 6]
) {
    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers to the current gaussian
    quats += idx * 4;
    scales += idx * 3;

    // compute the matrices
    glm::mat3 covar, preci;
    quat_scale_to_covar_preci(glm::make_vec4(quats), glm::make_vec3(scales),
                              covars ? &covar : nullptr, precis ? &preci : nullptr);

    // write to outputs: glm is column-major but we want row-major
    if (covars != nullptr) {
        if (triu) {
            covars += idx * 6;
            covars[0] = covar[0][0];
            covars[1] = covar[0][1];
            covars[2] = covar[0][2];
            covars[3] = covar[1][1];
            covars[4] = covar[1][2];
            covars[5] = covar[2][2];
        } else {
            covars += idx * 9;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    covars[i * 3 + j] = covar[j][i];
                }
            }
        }
    }
    if (precis != nullptr) {
        if (triu) {
            precis += idx * 6;
            precis[0] = preci[0][0];
            precis[1] = preci[0][1];
            precis[2] = preci[0][2];
            precis[3] = preci[1][1];
            precis[4] = preci[1][2];
            precis[5] = preci[2][2];
        } else {
            precis += idx * 9;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    precis[i * 3 + j] = preci[j][i];
                }
            }
        }
    }
}

__global__ void quat_scale_to_covar_preci_bwd_kernel(
    const uint32_t N,
    // fwd inputs
    const float *__restrict__ quats,  // [N, 4]
    const float *__restrict__ scales, // [N, 3]
    // grad outputs
    const float *__restrict__ v_covars, // [N, 3, 3] or [N, 6]
    const float *__restrict__ v_precis, // [N, 3, 3] or [N, 6]
    const bool triu,
    // grad inputs
    float *__restrict__ v_scales, // [N, 3]
    float *__restrict__ v_quats   // [N, 4]
) {
    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers to the current gaussian
    v_scales += idx * 3;
    v_quats += idx * 4;

    glm::vec4 quat = glm::make_vec4(quats + idx * 4);
    glm::vec3 scale = glm::make_vec3(scales + idx * 3);
    glm::mat3 rotmat = quat_to_rotmat(quat);

    glm::vec4 v_quat(0.f);
    glm::vec3 v_scale(0.f);
    if (v_covars != nullptr) {
        // glm is column-major, input is row-major
        glm::mat3 v_covar;
        if (triu) {
            v_covars += idx * 6;
            v_covar = glm::mat3(v_covars[0], v_covars[1] * .5f, v_covars[2] * .5f,
                                v_covars[1] * .5f, v_covars[3], v_covars[4] * .5f,
                                v_covars[2] * .5f, v_covars[4] * .5f, v_covars[5]);
        } else {
            v_covars += idx * 9;
            v_covar = glm::transpose(glm::make_mat3(v_covars));
        }
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
    }
    if (v_precis != nullptr) {
        // glm is column-major, input is row-major
        glm::mat3 v_preci;
        if (triu) {
            v_precis += idx * 6;
            v_preci = glm::mat3(v_precis[0], v_precis[1] * .5f, v_precis[2] * .5f,
                                v_precis[1] * .5f, v_precis[3], v_precis[4] * .5f,
                                v_precis[2] * .5f, v_precis[4] * .5f, v_precis[5]);
        } else {
            v_precis += idx * 9;
            v_preci = glm::transpose(glm::make_mat3(v_precis));
        }
        quat_scale_to_preci_vjp(quat, scale, rotmat, v_preci, v_quat, v_scale);
    }

    // write out results
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < 3; ++k) {
        v_scales[k] = v_scale[k];
    }
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < 4; ++k) {
        v_quats[k] = v_quat[k];
    }
}

std::tuple<torch::Tensor, torch::Tensor>
quat_scale_to_covar_preci_fwd_tensor(const torch::Tensor &quats,  // [N, 4]
                                     const torch::Tensor &scales, // [N, 3]
                                     const bool compute_covar, const bool compute_preci,
                                     const bool triu) {
    DEVICE_GUARD(quats);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);

    uint32_t N = quats.size(0);

    torch::Tensor covars, precis;
    if (compute_covar) {
        if (triu) {
            covars = torch::empty({N, 6}, quats.options());
        } else {
            covars = torch::empty({N, 3, 3}, quats.options());
        }
    }
    if (compute_preci) {
        if (triu) {
            precis = torch::empty({N, 6}, quats.options());
        } else {
            precis = torch::empty({N, 3, 3}, quats.options());
        }
    }

    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        quat_scale_to_covar_preci_fwd_kernel<<<(N + N_THREADS - 1) / N_THREADS,
                                               N_THREADS, 0, stream>>>(
            N, quats.data_ptr<float>(), scales.data_ptr<float>(), triu,
            compute_covar ? covars.data_ptr<float>() : nullptr,
            compute_preci ? precis.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(covars, precis);
}

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_bwd_tensor(
    const torch::Tensor &quats,                  // [N, 4]
    const torch::Tensor &scales,                 // [N, 3]
    const at::optional<torch::Tensor> &v_covars, // [N, 3, 3] or [N, 6]
    const at::optional<torch::Tensor> &v_precis, // [N, 3, 3] or [N, 6]
    const bool triu) {
    DEVICE_GUARD(quats);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    if (v_covars.has_value()) {
        CHECK_INPUT(v_covars.value());
    }
    if (v_precis.has_value()) {
        CHECK_INPUT(v_precis.value());
    }

    uint32_t N = quats.size(0);

    torch::Tensor v_scales = torch::empty_like(scales);
    torch::Tensor v_quats = torch::empty_like(quats);

    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        quat_scale_to_covar_preci_bwd_kernel<<<(N + N_THREADS - 1) / N_THREADS,
                                               N_THREADS, 0, stream>>>(
            N, quats.data_ptr<float>(), scales.data_ptr<float>(),
            v_covars.has_value() ? v_covars.value().data_ptr<float>() : nullptr,
            v_precis.has_value() ? v_precis.value().data_ptr<float>() : nullptr, triu,
            v_scales.data_ptr<float>(), v_quats.data_ptr<float>());
    }

    return std::make_tuple(v_quats, v_scales);
}

/****************************************************************************
 * Perspective Projection
 ****************************************************************************/

__global__ void persp_proj_fwd_kernel(const uint32_t C, const uint32_t N,
                                      const float *__restrict__ means,  // [C, N, 3]
                                      const float *__restrict__ covars, // [C, N, 3, 3]
                                      const float *__restrict__ Ks,     // [C, 3, 3]
                                      const uint32_t width, const uint32_t height,
                                      float *__restrict__ means2d, // [C, N, 2]
                                      float *__restrict__ covars2d // [C, N, 2, 2]
) { // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    // const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    Ks += cid * 9;
    means2d += idx * 2;
    covars2d += idx * 4;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    glm::mat2 covar2d;
    glm::vec2 mean2d;
    persp_proj(glm::make_vec3(means), glm::make_mat3(covars), fx, fy, cx, cy, width,
               height, covar2d, mean2d);

    // write to outputs: glm is column-major but we want row-major
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) { // rows
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 2; j++) { // cols
            covars2d[i * 2 + j] = covar2d[j][i];
        }
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
        means2d[i] = mean2d[i];
    }
}

__global__ void
persp_proj_bwd_kernel(const uint32_t C, const uint32_t N,
                      const float *__restrict__ means,  // [C, N, 3]
                      const float *__restrict__ covars, // [C, N, 3, 3]
                      const float *__restrict__ Ks,     // [C, 3, 3]
                      const uint32_t width, const uint32_t height,
                      const float *__restrict__ v_means2d,  // [C, N, 2]
                      const float *__restrict__ v_covars2d, // [C, N, 2, 2]
                      float *__restrict__ v_means,          // [C, N, 3]
                      float *__restrict__ v_covars          // [C, N, 3, 3]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    // const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    v_means += idx * 3;
    v_covars += idx * 9;
    Ks += cid * 9;
    v_means2d += idx * 2;
    v_covars2d += idx * 4;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    glm::mat3 v_covar(0.f);
    glm::vec3 v_mean(0.f);
    persp_proj_vjp(glm::make_vec3(means), glm::make_mat3(covars), fx, fy, cx, cy, width,
                   height, glm::transpose(glm::make_mat2(v_covars2d)),
                   glm::make_vec2(v_means2d), v_mean, v_covar);

    // write to outputs: glm is column-major but we want row-major
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) { // rows
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 3; j++) { // cols
            v_covars[i * 3 + j] = v_covar[j][i];
        }
    }

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_means[i] = v_mean[i];
    }
}

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_fwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const uint32_t width, const uint32_t height) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor covars2d = torch::empty({C, N, 2, 2}, covars.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        persp_proj_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            Ks.data_ptr<float>(), width, height, means2d.data_ptr<float>(),
            covars2d.data_ptr<float>());
    }
    return std::make_tuple(means2d, covars2d);
}

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_bwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const uint32_t width, const uint32_t height,
                      const torch::Tensor &v_means2d, // [C, N, 2]
                      const torch::Tensor &v_covars2d // [C, N, 2, 2]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_covars2d);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor v_means = torch::empty({C, N, 3}, means.options());
    torch::Tensor v_covars = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        persp_proj_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            Ks.data_ptr<float>(), width, height, v_means2d.data_ptr<float>(),
            v_covars2d.data_ptr<float>(), v_means.data_ptr<float>(),
            v_covars.data_ptr<float>());
    }
    return std::make_tuple(v_means, v_covars);
}

/****************************************************************************
 * World to Camera Transformation
 ****************************************************************************/

__global__ void world_to_cam_fwd_kernel(const uint32_t C, const uint32_t N,
                                        const float *__restrict__ means,    // [N, 3]
                                        const float *__restrict__ covars,   // [N, 3, 3]
                                        const float *__restrict__ viewmats, // [C, 4, 4]
                                        float *__restrict__ means_c,        // [C, N, 3]
                                        float *__restrict__ covars_c // [C, N, 3, 3]
) { // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    covars += gid * 9;
    viewmats += cid * 16;

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    if (means_c != nullptr) {
        glm::vec3 mean_c;
        pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
        means_c += idx * 3;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 3; i++) { // rows
            means_c[i] = mean_c[i];
        }
    }

    // write to outputs: glm is column-major but we want row-major
    if (covars_c != nullptr) {
        glm::mat3 covar_c;
        covar_world_to_cam(R, glm::make_mat3(covars), covar_c);
        covars_c += idx * 9;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 3; i++) { // rows
            PRAGMA_UNROLL
            for (uint32_t j = 0; j < 3; j++) { // cols
                covars_c[i * 3 + j] = covar_c[j][i];
            }
        }
    }
}

__global__ void
world_to_cam_bwd_kernel(const uint32_t C, const uint32_t N,
                        const float *__restrict__ means,      // [N, 3]
                        const float *__restrict__ covars,     // [N, 3, 3]
                        const float *__restrict__ viewmats,   // [C, 4, 4]
                        const float *__restrict__ v_means_c,  // [C, N, 3]
                        const float *__restrict__ v_covars_c, // [C, N, 3, 3]
                        float *__restrict__ v_means,          // [N, 3]
                        float *__restrict__ v_covars,         // [N, 3, 3]
                        float *__restrict__ v_viewmats        // [C, 4, 4]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    covars += gid * 9;
    viewmats += cid * 16;

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    glm::vec3 v_mean(0.f);
    glm::mat3 v_covar(0.f);
    glm::mat3 v_R(0.f);
    glm::vec3 v_t(0.f);

    if (v_means_c != nullptr) {
        glm::vec3 v_mean_c = glm::make_vec3(v_means_c + idx * 3);
        pos_world_to_cam_vjp(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    }
    if (v_covars_c != nullptr) {
        glm::mat3 v_covar_c = glm::transpose(glm::make_mat3(v_covars_c + idx * 9));
        covar_world_to_cam_vjp(R, glm::make_mat3(covars), v_covar_c, v_R, v_covar);
    }

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 9;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_covars + i * 3 + j, v_covar[j][i]);
                }
            }
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor>
world_to_cam_fwd_tensor(const torch::Tensor &means,   // [N, 3]
                        const torch::Tensor &covars,  // [N, 3, 3]
                        const torch::Tensor &viewmats // [C, 4, 4]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(viewmats);

    uint32_t N = means.size(0);
    uint32_t C = viewmats.size(0);

    torch::Tensor means_c = torch::empty({C, N, 3}, means.options());
    torch::Tensor covars_c = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        world_to_cam_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            viewmats.data_ptr<float>(), means_c.data_ptr<float>(),
            covars_c.data_ptr<float>());
    }
    return std::make_tuple(means_c, covars_c);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_cam_bwd_tensor(const torch::Tensor &means,                    // [N, 3]
                        const torch::Tensor &covars,                   // [N, 3, 3]
                        const torch::Tensor &viewmats,                 // [C, 4, 4]
                        const at::optional<torch::Tensor> &v_means_c,  // [C, N, 3]
                        const at::optional<torch::Tensor> &v_covars_c, // [C, N, 3, 3]
                        const bool means_requires_grad, const bool covars_requires_grad,
                        const bool viewmats_requires_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(viewmats);
    if (v_means_c.has_value()) {
        CHECK_INPUT(v_means_c.value());
    }
    if (v_covars_c.has_value()) {
        CHECK_INPUT(v_covars_c.value());
    }
    uint32_t N = means.size(0);
    uint32_t C = viewmats.size(0);

    torch::Tensor v_means, v_covars, v_viewmats;
    if (means_requires_grad) {
        v_means = torch::zeros({N, 3}, means.options());
    }
    if (covars_requires_grad) {
        v_covars = torch::zeros({N, 3, 3}, means.options());
    }
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros({C, 4, 4}, means.options());
    }

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        world_to_cam_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            v_means_c.has_value() ? v_means_c.value().data_ptr<float>() : nullptr,
            v_covars_c.has_value() ? v_covars_c.value().data_ptr<float>() : nullptr,
            means_requires_grad ? v_means.data_ptr<float>() : nullptr,
            covars_requires_grad ? v_covars.data_ptr<float>() : nullptr,
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_means, v_covars, v_viewmats);
}

/****************************************************************************
 * Camera projection of Gaussians
 ****************************************************************************/

__global__ void
fully_fused_projection_fwd_kernel(const uint32_t C, const uint32_t N,
                                  const float *__restrict__ means,                // [N, 3]
                                  const float *__restrict__ covars,               // [N, 6] optional
                                  const float *__restrict__ quats,                // [N, 4] optional
                                  const float *__restrict__ scales,               // [N, 3] optional
                                  const float *__restrict__ velocities,           // [N, 3] optional
                                  const float *__restrict__ viewmats,             // [C, 4, 4]
                                  const float *__restrict__ Ks,                   // [C, 3, 3]
                                  const int32_t image_width,
                                  const int32_t image_height,
                                  const float *__restrict__ lin_vel,              // [C, 3]
                                  const float *__restrict__ ang_vel,              // [C, 3]
                                  const float *__restrict__ rolling_shutter_time, // [C]
                                  const float eps2d,
                                  const float near_plane,
                                  const float far_plane,
                                  const float radius_clip,
                                  // outputs
                                  int32_t *__restrict__ radii,                    // [C, N, 2]
                                  float *__restrict__ means2d,                    // [C, N, 2]
                                  float *__restrict__ depths,                     // [C, N]
                                  float *__restrict__ conics,                     // [C, N, 3]
                                  float *__restrict__ compensations,              // [C, N] optional
                                  float *__restrict__ pix_vels                    // [C, N, 2] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    const float rs_time = rolling_shutter_time[cid];

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);


    // transform Gaussian center to camera space
    glm::vec3 mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    glm::mat3 covar;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                          covars[1], covars[3], covars[4], // 2nd column
                          covars[2], covars[4], covars[5]  // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
        quat_scale_to_covar_preci(glm::make_vec4(quats), glm::make_vec3(scales), &covar,
                                  nullptr);
    }
    glm::mat3 covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // perspective projection
    glm::mat2 covar2d;
    glm::vec2 mean2d;
    persp_proj(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], image_width, image_height,
               covar2d, mean2d);

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        return;
    }


    // take 3 sigma as the radius (non differentiable)
    // float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    // float v1 = b + sqrt(max(0.01f, b * b - det));
    // float radius = ceil(3.f * sqrt(v1));
    // float v2 = b - sqrt(max(0.1f, b * b - det));
    // float radius = ceil(3.f * sqrt(max(v1, v2)));

    float extent_x = 3.f * sqrt(covar2d[0][0]);
    float extent_y = 3.f * sqrt(covar2d[1][1]);

    if (extent_x <= radius_clip && extent_y <= radius_clip) {
        radii[idx * 2] = 0;
        return;
    }

    // increase radius to compensate for rolling shutter
    glm::vec2 pix_vel = { 0.f, 0.f };
    if (rs_time > 0) {
        // zero initialize velocity, rotate to camera space
        glm::vec3 vel_c(0.f);
        if (velocities != nullptr){
            glm::vec3 vel_w = glm::make_vec3(velocities + gid * 3);
            vel_world_to_cam(R, vel_w, vel_c);
        }
        compute_pix_velocity(mean_c, glm::make_vec3(lin_vel), glm::make_vec3(ang_vel), vel_c, Ks[0], Ks[4], Ks[2], Ks[5], image_width, image_height, pix_vel);
        extent_x += fabs(pix_vel.x) * 0.5f * rs_time;
        extent_y += fabs(pix_vel.y) * 0.5f * rs_time;
        //radius += hypotf(pix_vel.x, pix_vel.y) * 0.5f * rs_time;
    }

    // mask out gaussians outside the image region
    
    if (mean2d.x + extent_x <= 0 || mean2d.x - extent_x >= image_width ||
        mean2d.y + extent_y <= 0 || mean2d.y - extent_y >= image_height) {
        radii[idx * 2] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv;
    inverse(covar2d, covar2d_inv);

    // write to outputs
    radii[idx * 2] = (int32_t)extent_x;
    radii[idx * 2 + 1] = (int32_t)extent_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
    pix_vels[idx * 2] = pix_vel.x;
    pix_vels[idx * 2 + 1] = pix_vel.y;
}

__global__ void fully_fused_projection_bwd_kernel(
    // fwd inputs
    const uint32_t C, const uint32_t N,
    const float *__restrict__ means,                // [N, 3]
    const float *__restrict__ covars,               // [N, 6] optional
    const float *__restrict__ quats,                // [N, 4] optional
    const float *__restrict__ scales,               // [N, 3] optional
    const float *__restrict__ velocities,           // [N, 3] optional
    const float *__restrict__ viewmats,             // [C, 4, 4]
    const float *__restrict__ Ks,                   // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float *__restrict__ lin_vel,              // [C, 3]
    const float *__restrict__ ang_vel,              // [C, 3]
    const float *__restrict__ rolling_shutter_time, // [C]
    const float eps2d,
    // fwd outputs
    const int32_t *__restrict__ radii,              // [C, N, 2]
    const float *__restrict__ conics,               // [C, N, 3]
    const float *__restrict__ compensations,        // [C, N] optional
    // grad outputs
    const float *__restrict__ v_means2d,            // [C, N, 2]
    const float *__restrict__ v_depths,             // [C, N]
    const float *__restrict__ v_conics,             // [C, N, 3]
    const float *__restrict__ v_compensations,      // [C, N] optional
    const float *__restrict__ v_pix_vels,           // [C, N, 2] 
    // grad inputs
    float *__restrict__ v_means,                    // [N, 3]
    float *__restrict__ v_covars,                   // [N, 6] optional
    float *__restrict__ v_quats,                    // [N, 4] optional
    float *__restrict__ v_scales,                   // [N, 3] optional
    float *__restrict__ v_viewmats                  // [C, 4, 4] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx * 2] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    rolling_shutter_time += cid;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;
    v_pix_vels += idx * 2;

    // vjp: compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv = glm::mat2(conics[0], conics[1], conics[1], conics[2]);
    glm::mat2 v_covar2d_inv =
        glm::mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    glm::mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(eps2d, covar2d_inv, compensation, v_compensation, v_covar2d);
    }

    // transform Gaussian to camera space
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    glm::mat3 covar;
    glm::vec4 quat;
    glm::vec3 scale;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                          covars[1], covars[3], covars[4], // 2nd column
                          covars[2], covars[4], covars[5]  // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quat = glm::make_vec4(quats + gid * 4);
        scale = glm::make_vec3(scales + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    glm::vec3 mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    glm::mat3 covar_c;
    covar_world_to_cam(R, covar, covar_c);

    glm::vec3 v_p_view_pix_vel(0.f);
    glm::mat3 v_R(0.f);
    if (rolling_shutter_time[0] > 0 ) {
        glm::vec3 vel_c(0.f);
        glm::vec3 vel_w(0.f);
        glm::vec3 v_vel_c(0.f);  
        if (velocities != nullptr) {
            vel_w = glm::make_vec3(velocities + gid * 3);
            vel_world_to_cam(R, vel_w, vel_c);
        }
        compute_and_sum_pix_velocity_vjp(
            mean_c,
            glm::make_vec3(lin_vel),
            glm::make_vec3(ang_vel),
            vel_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            glm::make_vec2(v_pix_vels),
            v_p_view_pix_vel,
            v_vel_c);
        glm::vec3 v_vel_w(0.f);
        if (velocities != nullptr) {
            vel_world_to_cam_vjp(R, vel_w, v_vel_c, v_R, v_vel_w);
        }
    }

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    glm::mat3 v_covar_c(0.f);
    glm::vec3 v_mean_c(0.f);
    persp_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height,
                   v_covar2d, glm::make_vec2(v_means2d), v_mean_c, v_covar_c);

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // add contribution from pix velocities
    v_mean_c.x += v_p_view_pix_vel.x;
    v_mean_c.y += v_p_view_pix_vel.y;
    v_mean_c.z += v_p_view_pix_vel.z;

    // vjp: transform Gaussian covariance to camera space
    glm::vec3 v_mean(0.f);
    glm::mat3 v_covar(0.f);
    glm::vec3 v_t(0.f);
    pos_world_to_cam_vjp(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        }
    } else {
        // Directly output gradients w.r.t. the quaternion and scale
        glm::mat3 rotmat = quat_to_rotmat(quat);
        glm::vec4 v_quat(0.f);
        glm::vec3 v_scale(0.f);
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
        warpSum(v_quat, warp_group_g);
        warpSum(v_scale, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_quats += gid * 4;
            v_scales += gid * 3;
            gpuAtomicAdd(v_quats, v_quat[0]);
            gpuAtomicAdd(v_quats + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + 3, v_quat[3]);
            gpuAtomicAdd(v_scales, v_scale[0]);
            gpuAtomicAdd(v_scales + 1, v_scale[1]);
            gpuAtomicAdd(v_scales + 2, v_scale[2]);
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3] optional
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const torch::Tensor &linear_velocity,      // [C, 3]
    const torch::Tensor &angular_velocity,     // [C, 3]
    const torch::Tensor &rolling_shutter_time, // [C]
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    if (velocities.has_value()) {
        CHECK_INPUT(velocities.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(linear_velocity);
    CHECK_INPUT(angular_velocity);
    CHECK_INPUT(rolling_shutter_time);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii = torch::empty({C, N, 2}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({C, N}, means.options());
    }
    torch::Tensor pix_vels = torch::empty({C, N, 2}, means.options());
    if (C && N) {
        fully_fused_projection_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS,
                                            N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            linear_velocity.data_ptr<float>(),
            angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d,
            near_plane,
            far_plane,
            radius_clip, 
            radii.data_ptr<int32_t>(),
            means2d.data_ptr<float>(),
            depths.data_ptr<float>(),
            conics.data_ptr<float>(),
            calc_compensations ? compensations.data_ptr<float>() : nullptr,
            pix_vels.data_ptr<float>()
            );
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations, pix_vels);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                         // [N, 3]
    const at::optional<torch::Tensor> &covars,          // [N, 6] optional
    const at::optional<torch::Tensor> &quats,           // [N, 4] optional
    const at::optional<torch::Tensor> &scales,          // [N, 3] optional
    const at::optional<torch::Tensor> &velocities,      // [N, 3] optional
    const torch::Tensor &viewmats,                      // [C, 4, 4]
    const torch::Tensor &Ks,                            // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const torch::Tensor &linear_velocity,               // [C, 3]
    const torch::Tensor &angular_velocity,              // [C, 3]
    const torch::Tensor &rolling_shutter_time,          // [C]
    const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,                         // [C, N, 2]
    const torch::Tensor &conics,                        // [C, N, 3]
    const at::optional<torch::Tensor> &compensations,   // [C, N] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [C, N, 2]
    const torch::Tensor &v_depths,                      // [C, N]
    const torch::Tensor &v_conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const torch::Tensor &v_pix_vels,                    // [C, N, 2]
    const bool viewmats_requires_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    if (velocities.has_value()) {
        CHECK_INPUT(velocities.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(linear_velocity);
    CHECK_INPUT(angular_velocity);
    CHECK_INPUT(rolling_shutter_time);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }
    CHECK_INPUT(v_pix_vels);


    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars, v_quats, v_scales; // optional
    if (covars.has_value()) {
        v_covars = torch::zeros_like(covars.value());
    } else {
        v_quats = torch::zeros_like(quats.value());
        v_scales = torch::zeros_like(scales.value());
    }
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        fully_fused_projection_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS,
                                            N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
            covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(), 
            image_width,
            image_height,
            linear_velocity.data_ptr<float>(),
            angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d,
            radii.data_ptr<int32_t>(),
            conics.data_ptr<float>(),
            compensations.has_value() ? compensations.value().data_ptr<float>()
                                      : nullptr,
            v_means2d.data_ptr<float>(),
            v_depths.data_ptr<float>(),
            v_conics.data_ptr<float>(),
            v_compensations.has_value() ? v_compensations.value().data_ptr<float>()
                                        : nullptr,
            v_pix_vels.data_ptr<float>(),
            v_means.data_ptr<float>(),
            covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
            covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
            covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

__global__ void fully_fused_projection_packed_fwd_kernel(
    const uint32_t C, const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] Optional
    const float *__restrict__ quats,    // [N, 4] Optional
    const float *__restrict__ scales,   // [N, 3] Optional
    const float *__restrict__ velocities, // [N, 3] Optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width, const int32_t image_height, 
    const float *__restrict__ linear_velocity, // [C, 3]
    const float *__restrict__ angular_velocity, // [C, 3]
    const float *__restrict__ rolling_shutter_time, // [C]
    const float eps2d,
    const float near_plane, const float far_plane, const float radius_clip,
    const int32_t *__restrict__ block_accum, // [C * blocks_per_row] packing helper
    int32_t *__restrict__ block_cnts,        // [C * blocks_per_row] packing helper
    // outputs
    int32_t *__restrict__ indptr,       // [C + 1]
    int64_t *__restrict__ camera_ids,   // [nnz]
    int64_t *__restrict__ gaussian_ids, // [nnz]
    int32_t *__restrict__ radii,        // [nnz, 2]
    float *__restrict__ means2d,        // [nnz, 2]
    float *__restrict__ depths,         // [nnz]
    float *__restrict__ conics,         // [nnz, 3]
    float *__restrict__ compensations,   // [nnz] optional
    float *__restrict__ pix_vels        // [nnz, 2] optional
) {
    int32_t blocks_per_row = gridDim.x;

    int32_t row_idx = blockIdx.y; // cid
    int32_t block_col_idx = blockIdx.x;
    int32_t block_idx = row_idx * blocks_per_row + block_col_idx;

    int32_t col_idx = block_col_idx * blockDim.x + threadIdx.x; // gid

    bool valid = (row_idx < C) && (col_idx < N);

    // check if points are with camera near and far plane
    glm::vec3 mean_c;
    glm::mat3 R;
    if (valid) {
        // shift pointers to the current camera and gaussian
        means += col_idx * 3;
        viewmats += row_idx * 16;

        // glm is column-major but input is row-major
        R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                      viewmats[1], viewmats[5], viewmats[9], // 2nd column
                      viewmats[2], viewmats[6], viewmats[10] // 3rd column
        );
        glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

        // transform Gaussian center to camera space
        pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
        if (mean_c.z < near_plane || mean_c.z > far_plane) {
            valid = false;
        }
    }

    glm::vec3 vel_c(0.f);
    if (valid && velocities != nullptr) {
        glm::vec3 vel_w = glm::make_vec3(velocities + col_idx * 3);
        vel_world_to_cam(R, vel_w, vel_c);
    }

    // check if the perspective projection is valid.
    glm::mat2 covar2d;
    glm::vec2 mean2d;
    glm::mat2 covar2d_inv;
    float compensation;
    float det;
    if (valid) {
        // transform Gaussian covariance to camera space
        glm::mat3 covar;
        if (covars != nullptr) {
            // if a precomputed covariance is provided
            covars += col_idx * 6;
            covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                              covars[1], covars[3], covars[4], // 2nd column
                              covars[2], covars[4], covars[5]  // 3rd column
            );
        } else {
            // if not then compute it from quaternions and scales
            quats += col_idx * 4;
            scales += col_idx * 3;
            quat_scale_to_covar_preci(glm::make_vec4(quats), glm::make_vec3(scales),
                                      &covar, nullptr);
        }
        glm::mat3 covar_c;
        covar_world_to_cam(R, covar, covar_c);

        // perspective projection
        Ks += row_idx * 9;
        persp_proj(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], image_width,
                   image_height, covar2d, mean2d);

        det = add_blur(eps2d, covar2d, compensation);
        if (det <= 0.f) {
            valid = false;
        } else {
            // compute the inverse of the 2d covariance
            inverse(covar2d, covar2d_inv);
        }
    }

    // check if the points are in the image region
    // float radius;
    glm::vec2 pix_vel;
    float extent_x;
    float extent_y;
    if (valid) {
        // take 3 sigma as the radius (non differentiable)
        // float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        // float v1 = b + sqrt(max(0.1f, b * b - det));
        // float v2 = b - sqrt(max(0.1f, b * b - det));
        // radius = ceil(3.f * sqrt(max(v1, v2)));
        extent_x = 3.f * sqrt(covar2d[0][0]);
        extent_y = 3.f * sqrt(covar2d[1][1]);

        if (extent_x <= radius_clip && extent_y <= radius_clip) {
            valid = false;
        }

        // increase radius to compensate for rolling shutter
        rolling_shutter_time += row_idx;
        if (valid && rolling_shutter_time[0] > 0) {
            linear_velocity += row_idx * 3;
            angular_velocity += row_idx * 3;

            compute_pix_velocity(mean_c, glm::make_vec3(linear_velocity),
                                        glm::make_vec3(angular_velocity), vel_c, Ks[0], Ks[4], Ks[2], Ks[5], image_width, image_height, pix_vel);
            extent_x += fabs(pix_vel.x) * 0.5f * rolling_shutter_time[0];
            extent_y += fabs(pix_vel.y) * 0.5f * rolling_shutter_time[0];
        }

        // mask out gaussians outside the image region
        if (mean2d.x + extent_x <= 0 || mean2d.x - extent_x >= image_width ||
            mean2d.y + extent_y <= 0 || mean2d.y - extent_y >= image_height) {
            valid = false;
        }
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        // First pass: compute the block-wide sum
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, N_THREADS> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        } else {
            aggregate = 0;
        }
        if (threadIdx.x == 0) {
            block_cnts[block_idx] = aggregate;
        }
    } else {
        // Second pass: write out the indices of the non zero elements
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, N_THREADS> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if (valid) {
            if (block_idx > 0) {
                int32_t offset = block_accum[block_idx - 1];
                thread_data += offset;
            }
            // write to outputs
            camera_ids[thread_data] = row_idx;   // cid
            gaussian_ids[thread_data] = col_idx; // gid
            radii[thread_data * 2] = (int32_t)extent_x;
            radii[thread_data * 2 + 1] = (int32_t)extent_y;
            means2d[thread_data * 2] = mean2d.x;
            means2d[thread_data * 2 + 1] = mean2d.y;
            depths[thread_data] = mean_c.z;
            conics[thread_data * 3] = covar2d_inv[0][0];
            conics[thread_data * 3 + 1] = covar2d_inv[0][1];
            conics[thread_data * 3 + 2] = covar2d_inv[1][1];
            pix_vels[thread_data * 2] = pix_vel.x;
            pix_vels[thread_data * 2 + 1] = pix_vel.y;
            if (compensations != nullptr) {
                compensations[thread_data] = compensation;
            }
        }
        // lane 0 of the first block in each row writes the indptr
        if (threadIdx.x == 0 && block_col_idx == 0) {
            if (row_idx == 0) {
                indptr[0] = 0;
                indptr[C] = block_accum[C * blocks_per_row - 1];
            } else {
                indptr[row_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_packed_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const at::optional<torch::Tensor> &quats,  // [N, 3]
    const at::optional<torch::Tensor> &scales, // [N, 3]
    const at::optional<torch::Tensor> &velocities, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width, const uint32_t image_height, 
    const torch::Tensor &linear_velocity,       // [C, 3]
    const torch::Tensor &angular_velocity,      // [C, 3]
    const torch::Tensor &rolling_shutter_time,  // [C]
    const float eps2d,
    const float near_plane, const float far_plane, const float radius_clip,
    const bool calc_compensations) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    if (velocities.has_value()) {
        CHECK_INPUT(velocities.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(linear_velocity);
    CHECK_INPUT(angular_velocity);
    CHECK_INPUT(rolling_shutter_time);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    auto opt = means.options().dtype(torch::kInt32);

    uint32_t nrows = C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + N_THREADS - 1) / N_THREADS;

    dim3 threads = {N_THREADS, 1, 1};
    // limit on the number of blocks: [2**31 - 1, 65535, 65535]
    dim3 blocks = {blocks_per_row, nrows, 1};

    // first pass
    int32_t nnz;
    torch::Tensor block_accum;
    if (C && N) {
        torch::Tensor block_cnts = torch::empty({nrows * blocks_per_row}, opt);
        fully_fused_projection_packed_fwd_kernel<<<blocks, threads, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(), Ks.data_ptr<float>(), image_width, image_height,
            linear_velocity.data_ptr<float>(), angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d, near_plane, far_plane, radius_clip, nullptr,
            block_cnts.data_ptr<int32_t>(), nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr);
        block_accum = torch::cumsum(block_cnts, 0, torch::kInt32);
        nnz = block_accum[-1].item<int32_t>();
    } else {
        nnz = 0;
    }

    // second pass
    torch::Tensor indptr = torch::empty({C + 1}, opt);
    torch::Tensor camera_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor gaussian_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor radii = torch::zeros({nnz, 2}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({nnz, 2}, means.options());
    torch::Tensor depths = torch::empty({nnz}, means.options());
    torch::Tensor conics = torch::empty({nnz, 3}, means.options());
    torch::Tensor pix_vels = torch::empty({nnz, 2}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({nnz}, means.options());
    }

    if (nnz) {
        fully_fused_projection_packed_fwd_kernel<<<blocks, threads, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(), Ks.data_ptr<float>(), image_width, image_height,
            linear_velocity.data_ptr<float>(), angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d, near_plane, far_plane, radius_clip, block_accum.data_ptr<int32_t>(),
            nullptr, indptr.data_ptr<int32_t>(), camera_ids.data_ptr<int64_t>(),
            gaussian_ids.data_ptr<int64_t>(), radii.data_ptr<int32_t>(),
            means2d.data_ptr<float>(), depths.data_ptr<float>(),
            conics.data_ptr<float>(),
            calc_compensations ? compensations.data_ptr<float>() : nullptr,
            pix_vels.data_ptr<float>());
    } else {
        indptr.fill_(0);
    }

    return std::make_tuple(indptr, camera_ids, gaussian_ids, radii, means2d, depths,
                           conics, compensations, pix_vels);
}

__global__ void fully_fused_projection_packed_bwd_kernel(
    // fwd inputs
    const uint32_t C, const uint32_t N, const uint32_t nnz,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] Optional
    const float *__restrict__ quats,    // [N, 4] Optional
    const float *__restrict__ scales,   // [N, 3] Optional
    const float *__restrict__ velocities, // [N, 3] Optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width, const int32_t image_height, 
    const float *__restrict__ linear_velocity, // [C, 3]
    const float *__restrict__ angular_velocity, // [C, 3]
    const float *__restrict__ rolling_shutter_time, // [C]
    const float eps2d,
    // fwd outputs
    const int64_t *__restrict__ camera_ids,   // [nnz]
    const int64_t *__restrict__ gaussian_ids, // [nnz]
    const float *__restrict__ conics,         // [nnz, 3]
    const float *__restrict__ compensations,  // [nnz] optional
    const float *__restrict__ pix_vels,       // [nnz, 2]
    // grad outputs
    const float *__restrict__ v_means2d,       // [nnz, 2]
    const float *__restrict__ v_depths,        // [nnz]
    const float *__restrict__ v_conics,        // [nnz, 3]
    const float *__restrict__ v_compensations, // [nnz] optional
    const float *__restrict__ v_pix_vels,      // [nnz, 2]
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    // grad inputs
    float *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    float *__restrict__ v_covars,  // [N, 6] or [nnz, 6] Optional
    float *__restrict__ v_quats,   // [N, 4] or [nnz, 4] Optional
    float *__restrict__ v_scales,  // [N, 3] or [nnz, 3] Optional
    float *__restrict__ v_viewmats // [C, 4, 4] Optional
) {
    // parallelize over nnz.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= nnz) {
        return;
    }
    const int64_t cid = camera_ids[idx];   // camera id
    const int64_t gid = gaussian_ids[idx]; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;
    linear_velocity += cid * 3;
    angular_velocity += cid * 3;
    rolling_shutter_time += cid;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;
    v_pix_vels += idx * 2;

    // vjp: compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv = glm::mat2(conics[0], conics[1], conics[1], conics[2]);
    glm::mat2 v_covar2d_inv =
        glm::mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    glm::mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(eps2d, covar2d_inv, compensation, v_compensation, v_covar2d);
    }

    // transform Gaussian to camera space
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);
    glm::mat3 covar;
    glm::vec4 quat;
    glm::vec3 scale;
    if (covars != nullptr) {
        // if a precomputed covariance is provided
        covars += gid * 6;
        covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                          covars[1], covars[3], covars[4], // 2nd column
                          covars[2], covars[4], covars[5]  // 3rd column
        );
    } else {
        // if not then compute it from quaternions and scales
        quat = glm::make_vec4(quats + gid * 4);
        scale = glm::make_vec3(scales + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    glm::vec3 mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    glm::mat3 covar_c;
    covar_world_to_cam(R, covar, covar_c);

    glm::vec3 vel_c(0.f);
    glm::vec3 vel_w(0.f);
    if (velocities != nullptr) {
        vel_w = glm::make_vec3(velocities + gid * 3);
        vel_world_to_cam(R, vel_w, vel_c);
    }
    glm::mat3 v_R(0.f);
    glm::vec3 v_vel_c(0.f);
    glm::vec3 v_vel_w(0.f);
    // vjp: velocity term
    glm::vec3 v_p_view_pix_vel(0.f);
    if (rolling_shutter_time[0] > 0 ) {
        compute_and_sum_pix_velocity_vjp(
            mean_c,
            glm::make_vec3(linear_velocity),
            glm::make_vec3(angular_velocity),
            vel_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            glm::make_vec2(v_pix_vels),
            v_p_view_pix_vel,
            v_vel_c);
    }
    if (velocities != nullptr) {
        vel_world_to_cam_vjp(R, vel_w, v_vel_c, v_R, v_vel_w);
    }

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    glm::mat3 v_covar_c(0.f);
    glm::vec3 v_mean_c(0.f);
    persp_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height,
                   v_covar2d, glm::make_vec2(v_means2d), v_mean_c, v_covar_c);

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // add contribution from pix velocities
    v_mean_c.x += v_p_view_pix_vel.x;
    v_mean_c.y += v_p_view_pix_vel.y;
    v_mean_c.z += v_p_view_pix_vel.z;

    // vjp: transform Gaussian covariance to camera space
    glm::vec3 v_mean(0.f);
    glm::mat3 v_covar(0.f);
    glm::vec3 v_t(0.f);
    pos_world_to_cam_vjp(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    if (sparse_grad) {
        // write out results with sparse layout
        if (v_means != nullptr) {
            v_means += idx * 3;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                v_means[i] = v_mean[i];
            }
        }
        if (v_covars != nullptr) {
            v_covars += idx * 6;
            v_covars[0] = v_covar[0][0];
            v_covars[1] = v_covar[0][1] + v_covar[1][0];
            v_covars[2] = v_covar[0][2] + v_covar[2][0];
            v_covars[3] = v_covar[1][1];
            v_covars[4] = v_covar[1][2] + v_covar[2][1];
            v_covars[5] = v_covar[2][2];
        } else {
            glm::mat3 rotmat = quat_to_rotmat(quat);
            glm::vec4 v_quat(0.f);
            glm::vec3 v_scale(0.f);
            quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
            v_quats += idx * 4;
            v_scales += idx * 3;
            v_quats[0] = v_quat[0];
            v_quats[1] = v_quat[1];
            v_quats[2] = v_quat[2];
            v_quats[3] = v_quat[3];
            v_scales[0] = v_scale[0];
            v_scales[1] = v_scale[1];
            v_scales[2] = v_scale[2];
        }
    } else {
        // write out results with dense layout
        // #if __CUDA_ARCH__ >= 700
        // write out results with warp-level reduction
        auto warp_group_g = cg::labeled_partition(warp, gid);
        if (v_means != nullptr) {
            warpSum(v_mean, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_means += gid * 3;
                PRAGMA_UNROLL
                for (uint32_t i = 0; i < 3; i++) {
                    gpuAtomicAdd(v_means + i, v_mean[i]);
                }
            }
        }
        if (v_covars != nullptr) {
            // Directly output gradients w.r.t. the covariance
            warpSum(v_covar, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_covars += gid * 6;
                gpuAtomicAdd(v_covars, v_covar[0][0]);
                gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
                gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
                gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
                gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
                gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
            }
        } else {
            // Directly output gradients w.r.t. the quaternion and scale
            glm::mat3 rotmat = quat_to_rotmat(quat);
            glm::vec4 v_quat(0.f);
            glm::vec3 v_scale(0.f);
            quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
            warpSum(v_quat, warp_group_g);
            warpSum(v_scale, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_quats += gid * 4;
                v_scales += gid * 3;
                gpuAtomicAdd(v_quats, v_quat[0]);
                gpuAtomicAdd(v_quats + 1, v_quat[1]);
                gpuAtomicAdd(v_quats + 2, v_quat[2]);
                gpuAtomicAdd(v_quats + 3, v_quat[3]);
                gpuAtomicAdd(v_scales, v_scale[0]);
                gpuAtomicAdd(v_scales + 1, v_scale[1]);
                gpuAtomicAdd(v_scales + 2, v_scale[2]);
            }
        }
    }
    // v_viewmats is always in dense layout
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_packed_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const at::optional<torch::Tensor> &quats,  // [N, 4]
    const at::optional<torch::Tensor> &scales, // [N, 3]
    const at::optional<torch::Tensor> &velocities, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width, const uint32_t image_height, 
    const torch::Tensor &linear_velocity,       // [C, 3]
    const torch::Tensor &angular_velocity,      // [C, 3]
    const torch::Tensor &rolling_shutter_time,  // [C]
    const float eps2d,
    // fwd outputs
    const torch::Tensor &camera_ids,                  // [nnz]
    const torch::Tensor &gaussian_ids,                // [nnz]
    const torch::Tensor &conics,                      // [nnz, 3]
    const at::optional<torch::Tensor> &compensations, // [nnz] optional
    const torch::Tensor &pix_vels,                    // [nnz, 2]
    // grad outputs
    const torch::Tensor &v_means2d,                     // [nnz, 2]
    const torch::Tensor &v_depths,                      // [nnz]
    const torch::Tensor &v_conics,                      // [nnz, 3]
    const at::optional<torch::Tensor> &v_compensations, // [nnz] optional
    const torch::Tensor &v_pix_vels,                    // [nnz, 2]
    const bool viewmats_requires_grad, const bool sparse_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    if (velocities.has_value()) {
        CHECK_INPUT(velocities.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(camera_ids);
    CHECK_INPUT(gaussian_ids);
    CHECK_INPUT(conics);
    CHECK_INPUT(linear_velocity);
    CHECK_INPUT(angular_velocity);
    CHECK_INPUT(rolling_shutter_time);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    CHECK_INPUT(pix_vels);
    CHECK_INPUT(v_pix_vels);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    uint32_t nnz = camera_ids.size(0);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means, v_covars, v_quats, v_scales, v_viewmats;
    if (sparse_grad) {
        v_means = torch::zeros({nnz, 3}, means.options());
        if (covars.has_value()) {
            v_covars = torch::zeros({nnz, 6}, covars.value().options());
        } else {
            v_quats = torch::zeros({nnz, 4}, quats.value().options());
            v_scales = torch::zeros({nnz, 3}, scales.value().options());
        }
        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros({C, 4, 4}, viewmats.options());
        }
    } else {
        v_means = torch::zeros_like(means);
        if (covars.has_value()) {
            v_covars = torch::zeros_like(covars.value());
        } else {
            v_quats = torch::zeros_like(quats.value());
            v_scales = torch::zeros_like(scales.value());
        }
        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros_like(viewmats);
        }
    }
    if (nnz) {
        fully_fused_projection_packed_bwd_kernel<<<(nnz + N_THREADS - 1) / N_THREADS,
                                                   N_THREADS, 0, stream>>>(
            C, N, nnz, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
            covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(), Ks.data_ptr<float>(), image_width, image_height,
            linear_velocity.data_ptr<float>(), angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d, camera_ids.data_ptr<int64_t>(), gaussian_ids.data_ptr<int64_t>(),
            conics.data_ptr<float>(),
            compensations.has_value() ? compensations.value().data_ptr<float>()
                                      : nullptr,
            pix_vels.data_ptr<float>(),
            v_means2d.data_ptr<float>(), v_depths.data_ptr<float>(),
            v_conics.data_ptr<float>(),
            v_compensations.has_value() ? v_compensations.value().data_ptr<float>()
                                        : nullptr,
            v_pix_vels.data_ptr<float>(),
            sparse_grad, v_means.data_ptr<float>(),
            covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
            covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
            covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}

/****************************************************************************
 * Lidar projection of Gaussians
 ****************************************************************************/
__global__ void lidar_proj_fwd_kernel(const uint32_t C, const uint32_t N,
                                        const float *__restrict__ means,    // [C, N, 3]
                                        const float *__restrict__ covars,   // [C, N, 3, 3]
                                        const float eps2d,
                                        float *__restrict__ means2d,        // [C, N, 2]
                                        float *__restrict__ covars2d,        // [C, N, 2, 2]
                                        float *__restrict__ depth_compensations // [C, N, 2]
) { // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    // shift pointers to the current lidar and gaussian
    means += idx * 3;
    covars += idx * 9;
    means2d += idx * 2;
    covars2d += idx * 4;
    depth_compensations += idx * 2;

    glm::vec2 mean2d;
    glm::mat2 covar2d;
    glm::vec2 depth_compensation;
    glm::mat3 jacobian;
    lidar_proj(glm::make_vec3(means), glm::make_mat3(covars), eps2d, mean2d, covar2d, depth_compensation, jacobian);

    // write to outputs: glm is column-major but we want row-major
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) { // rows
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 2; j++) { // cols
            covars2d[i * 2 + j] = covar2d[j][i];
        }
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
        means2d[i] = mean2d[i];
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
        depth_compensations[i] = depth_compensation[i];
    }
}

__global__ void
lidar_proj_bwd_kernel(const uint32_t C, const uint32_t N,
                      const float *__restrict__ means,  // [C, N, 3]
                      const float *__restrict__ covars, // [C, N, 3, 3]
                      const float eps2d,
                      const float *__restrict__ v_means2d,  // [C, N, 2]
                      const float *__restrict__ v_covars2d, // [C, N, 2, 2]
                      const float *__restrict__ v_depth_compensations, // [C, N, 2]
                      float *__restrict__ v_means,          // [C, N, 3]
                      float *__restrict__ v_covars          // [C, N, 3, 3]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    // shift pointers to the current lidar and gaussian
    means += idx * 3;
    covars += idx * 9;
    v_means += idx * 3;
    v_covars += idx * 9;
    v_depth_compensations += idx * 2;
    v_means2d += idx * 2;
    v_covars2d += idx * 4;

    glm::mat3 v_covar(0.f);
    glm::vec3 v_mean(0.f);
    lidar_proj_vjp(glm::make_vec3(means), glm::make_mat3(covars), eps2d,
                   glm::make_vec2(v_means2d), glm::transpose(glm::make_mat2(v_covars2d)), glm::make_vec2(v_depth_compensations),
                   v_mean, v_covar);

    // write to outputs: glm is column-major but we want row-major
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) { // rows
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 3; j++) { // cols
            v_covars[i * 3 + j] = v_covar[j][i];
        }
    }

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_means[i] = v_mean[i];
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
lidar_proj_fwd_tensor(const torch::Tensor &means,   // [C, N, 3]
                        const torch::Tensor &covars,  // [C, N, 3, 3]
                        const float eps2d
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor covars2d = torch::empty({C, N, 2, 2}, covars.options());
    torch::Tensor depth_compensations = torch::empty({C, N, 2}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        lidar_proj_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            eps2d,
            means2d.data_ptr<float>(), covars2d.data_ptr<float>(),
            depth_compensations.data_ptr<float>());
    }
    return std::make_tuple(means2d, covars2d, depth_compensations);
}

std::tuple<torch::Tensor, torch::Tensor>
lidar_proj_bwd_tensor(const torch::Tensor &means,                      // [C, N, 3]
                      const torch::Tensor &covars,                   // [C, N, 3, 3]
                      const float eps2d,
                      const torch::Tensor &v_means2d, // [C, N, 2]
                      const torch::Tensor &v_covars2d, // [C, N, 2, 2]
                      const torch::Tensor &v_depth_compensations // [C, N, 2]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_covars2d);
    CHECK_INPUT(v_depth_compensations);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor v_means = torch::empty({C, N, 3}, means.options());
    torch::Tensor v_covars = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        lidar_proj_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(), eps2d,
            v_means2d.data_ptr<float>(), v_covars2d.data_ptr<float>(), v_depth_compensations.data_ptr<float>(),
            v_means.data_ptr<float>(), v_covars.data_ptr<float>());
    }
    return std::make_tuple(v_means, v_covars);
}

__global__ void compute_lidar_velocity_fwd_kernel(const uint32_t C, const uint32_t N,
                                        const float *__restrict__ p_view,    // [C, N, 3]
                                        const float *__restrict__ lin_vel,   // [C, 3]
                                        const float *__restrict__ ang_vel,   // [C, 3]
                                        const float *__restrict__ v_view,    // [C, N, 3]
                                        float *__restrict__ total_vel_pix   // [C, N, 3]
) { // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C *N) {
        return;
    }
    const uint32_t cid = idx / N; // lidar id
    // shift pointers to the current lidar and gaussian
    p_view += idx * 3;
    v_view += idx * 3;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    total_vel_pix += idx * 3;

    glm::mat3 J(0.f);
    glm::vec3 total_vel_pix_local;
    compute_lidar_velocity(glm::make_vec3(p_view), glm::make_vec3(lin_vel), glm::make_vec3(ang_vel), glm::make_vec3(v_view), J, total_vel_pix_local);

    // write to outputs
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        total_vel_pix[i] = total_vel_pix_local[i];
    }
}

__global__ void compute_lidar_velocity_bwd_kernel(const uint32_t C, const uint32_t N,
                      const float *__restrict__ p_view,    // [C, N, 3]
                      const float *__restrict__ lin_vel,   // [C, 3]
                      const float *__restrict__ ang_vel,   // [C, 3]
                      const float *__restrict__ vel_view,  // [C, N, 3]
                      const float *__restrict__ v_spherical_velocity, // [C, N, 3]
                      float *__restrict__ v_p_view_accumulator,   // [C, N, 3]
                      float *__restrict__ v_vel_view_accumulator // [C, N, 3]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // lidar id
    // shift pointers to the current lidar and gaussian
    p_view += idx * 3;
    vel_view += idx * 3;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    v_spherical_velocity += idx * 3;
    v_p_view_accumulator += idx * 3;
    v_vel_view_accumulator += idx * 3;

    glm::vec3 v_p_view_spherical_vel = { 0.f, 0.f, 0.f };
    glm::vec3 v_vel_view = { 0.f, 0.f, 0.f };
    compute_and_sum_lidar_velocity_vjp(glm::make_vec3(p_view), glm::make_vec3(lin_vel), glm::make_vec3(ang_vel), glm::make_vec3(vel_view), glm::make_vec3(v_spherical_velocity), v_p_view_spherical_vel, v_vel_view);

    // write to outputs
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_p_view_accumulator[i] = v_p_view_spherical_vel[i];
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_vel_view_accumulator[i] = v_vel_view[i];
    }
}

torch::Tensor compute_lidar_velocity_fwd_tensor(const torch::Tensor &p_view,   // [C, N, 3]
                        const torch::Tensor &lin_vel,   // [C, 3]
                        const torch::Tensor &ang_vel,   // [C, 3]
                        const torch::Tensor &v_view // [C, N, 3]
) {
    DEVICE_GUARD(p_view);
    CHECK_INPUT(p_view);
    CHECK_INPUT(lin_vel);
    CHECK_INPUT(ang_vel);
    CHECK_INPUT(v_view);

    uint32_t C = p_view.size(0);
    uint32_t N = p_view.size(1);

    torch::Tensor spherical_vel = torch::empty({C, N, 3}, p_view.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        compute_lidar_velocity_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, p_view.data_ptr<float>(), lin_vel.data_ptr<float>(),
            ang_vel.data_ptr<float>(), v_view.data_ptr<float>(),
            spherical_vel.data_ptr<float>());
    }
    return spherical_vel;
}

std::tuple<torch::Tensor, torch::Tensor>
compute_lidar_velocity_bwd_tensor(const torch::Tensor &p_view,   // [C, N, 3]
                        const torch::Tensor &lin_vel,   // [C, 3]
                        const torch::Tensor &ang_vel,   // [C, 3]
                        const torch::Tensor &v_view,   // [C, N, 3]
                        const torch::Tensor &v_spherical_velocity // [C, N, 3]
) {
    DEVICE_GUARD(p_view);
    CHECK_INPUT(p_view);
    CHECK_INPUT(lin_vel);
    CHECK_INPUT(ang_vel);
    CHECK_INPUT(v_view);
    CHECK_INPUT(v_spherical_velocity);

    uint32_t C = p_view.size(0);
    uint32_t N = p_view.size(1);

    torch::Tensor v_p_view = torch::empty({C, N, 3}, p_view.options());
    torch::Tensor v_vel_view = torch::empty({C, N, 3}, p_view.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        compute_lidar_velocity_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, p_view.data_ptr<float>(), lin_vel.data_ptr<float>(),
            ang_vel.data_ptr<float>(), v_view.data_ptr<float>(), 
            v_spherical_velocity.data_ptr<float>(), 
            v_p_view.data_ptr<float>(), v_vel_view.data_ptr<float>());
    }
    return std::make_tuple(v_p_view, v_vel_view);
}

__global__ void compute_pix_velocity_fwd_kernel(const uint32_t C, const uint32_t N,
                                        const float *__restrict__ p_view,    // [C, N, 3]
                                        const float *__restrict__ lin_vel,   // [C, 3]
                                        const float *__restrict__ ang_vel,   // [C, 3]
                                        const float *__restrict__ v_view,    // [C, N, 3]
                                        const float *__restrict__ Ks,        // [C, 3, 3]
                                        const uint32_t width, const uint32_t height,
                                        float *__restrict__ total_vel_pix   // [C, N, 2]
) { // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C *N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    // shift pointers to the current camera and gaussian
    p_view += idx * 3;
    v_view += idx * 3;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    Ks += cid * 9;
    total_vel_pix += idx * 2;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];

    glm::vec2 total_vel_pix_local = { 0.f, 0.f };
    compute_pix_velocity(glm::make_vec3(p_view), glm::make_vec3(lin_vel), glm::make_vec3(ang_vel), glm::make_vec3(v_view), fx, fy, cx, cy, width, height, total_vel_pix_local);

    // write to outputs
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
        total_vel_pix[i] = total_vel_pix_local[i];
    }
}

__global__ void compute_pix_velocity_bwd_kernel(const uint32_t C, const uint32_t N,
                      const float *__restrict__ p_view,    // [C, N, 3]
                      const float *__restrict__ lin_vel,   // [C, 3]
                      const float *__restrict__ ang_vel,   // [C, 3]
                      const float *__restrict__ v_view,   // [C, N, 3]
                      const float *__restrict__ Ks,       // [C, 3, 3]
                      uint32_t image_width, uint32_t image_height,
                      const float *__restrict__ v_pix_velocity, // [C, N, 2]
                      float *__restrict__ v_p_view_accumulator,   // [C, N, 3]
                      float *__restrict__ v_vel_view_accumulator // [C, N, 3]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    // shift pointers to the current camera and gaussian
    p_view += idx * 3;
    v_view += idx * 3;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    Ks += cid * 9;
    v_pix_velocity += idx * 2;
    v_p_view_accumulator += idx * 3;
    v_vel_view_accumulator += idx * 3;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];

    glm::vec3 v_p_view_pix_vel = { 0.f, 0.f, 0.f };
    glm::vec3 v_vel_view(0.f);
    compute_and_sum_pix_velocity_vjp(glm::make_vec3(p_view), glm::make_vec3(lin_vel), glm::make_vec3(ang_vel), glm::make_vec3(v_view), fx, fy, cx, cy, image_width, image_height, glm::make_vec2(v_pix_velocity), v_p_view_pix_vel, v_vel_view);

    // write to outputs
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_p_view_accumulator[i] = v_p_view_pix_vel[i];
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_vel_view_accumulator[i] = v_vel_view[i];
    }
}

torch::Tensor compute_pix_velocity_fwd_tensor(const torch::Tensor &p_view,   // [C, N, 3]
                        const torch::Tensor &lin_vel,   // [C, 3]
                        const torch::Tensor &ang_vel,   // [C, 3]
                        const torch::Tensor &v_view,   // [C, N, 3]
                        const torch::Tensor &Ks,   // [C, 3, 3]
                        const uint32_t image_width, const uint32_t image_height
) {
    DEVICE_GUARD(p_view);
    CHECK_INPUT(p_view);
    CHECK_INPUT(lin_vel);
    CHECK_INPUT(ang_vel);
    CHECK_INPUT(v_view);
    CHECK_INPUT(Ks);

    uint32_t C = p_view.size(0);
    uint32_t N = p_view.size(1);

    torch::Tensor pix_vel = torch::empty({C, N, 2}, p_view.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        compute_pix_velocity_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, p_view.data_ptr<float>(), lin_vel.data_ptr<float>(),
            ang_vel.data_ptr<float>(),
            v_view.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width, image_height,
            pix_vel.data_ptr<float>());
    }
    return pix_vel;
}

std::tuple<torch::Tensor, torch::Tensor>
compute_pix_velocity_bwd_tensor(const torch::Tensor &p_view,   // [C, N, 3]
                        const torch::Tensor &lin_vel,   // [C, 3]
                        const torch::Tensor &ang_vel,   // [C, 3]
                        const torch::Tensor &v_view,   // [C, N, 3]
                        const torch::Tensor &Ks,   // [C, 3, 3]
                        uint32_t image_width, uint32_t image_height,
                        const torch::Tensor &v_pix_velocity // [C, N, 2]
) {
    DEVICE_GUARD(p_view);
    CHECK_INPUT(p_view);
    CHECK_INPUT(lin_vel);
    CHECK_INPUT(ang_vel);
    CHECK_INPUT(v_view);
    CHECK_INPUT(Ks);
    CHECK_INPUT(v_pix_velocity);

    uint32_t C = p_view.size(0);
    uint32_t N = p_view.size(1);

    torch::Tensor v_p_view = torch::empty({C, N, 3}, p_view.options());
    torch::Tensor v_vel_view = torch::empty({C, N, 3}, p_view.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        compute_pix_velocity_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, p_view.data_ptr<float>(), lin_vel.data_ptr<float>(),
            ang_vel.data_ptr<float>(),
            v_view.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width, image_height,
            v_pix_velocity.data_ptr<float>(), 
            v_p_view.data_ptr<float>(),
            v_vel_view.data_ptr<float>());
    }
    return std::make_tuple(v_p_view, v_vel_view);
}

__global__ void
fully_fused_lidar_projection_fwd_kernel(const uint32_t C, const uint32_t N,
                                  const float *__restrict__ means,                // [N, 3]
                                  const float *__restrict__ covars,               // [N, 6] optional
                                  const float *__restrict__ quats,                // [N, 4] optional
                                  const float *__restrict__ scales,               // [N, 3] optional
                                  const float *__restrict__ velocities,           // [N, 3]
                                  const float *__restrict__ viewmats,             // [C, 4, 4]
                                  const float min_elevation,
                                  const float max_elevation,
                                  const float min_azimuth,
                                  const float max_azimuth,
                                  const float *__restrict__ lin_vel,              // [C, 3]
                                  const float *__restrict__ ang_vel,              // [C, 3]
                                  const float *__restrict__ rolling_shutter_time, // [C]
                                  const float eps2d,
                                  const float near_plane,
                                  const float far_plane,
                                  const float radius_clip,
                                  // outputs
                                  float *__restrict__ radii,                      // [C, N]
                                  float *__restrict__ means2d,                    // [C, N, 2]
                                  float *__restrict__ depths,                     // [C, N]
                                  float *__restrict__ conics,                     // [C, N, 3]
                                  float *__restrict__ compensations,              // [C, N] optional
                                  float *__restrict__ pix_vels,                    // [C, N, 3] 
                                  float *__restrict__ depth_compensations        // [C, N, 2] 
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // lidar id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current lidar and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    const float rs_time = rolling_shutter_time[cid];

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to lidar space
    glm::vec3 mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    float distance = norm3df(mean_c.x , mean_c.y, mean_c.z);
    if (distance < near_plane || distance > far_plane) {
        radii[idx * 2] = 0.f;
        return;
    }

    // get covariance, either directly from input or compute from quaternions and scales
    glm::mat3 covar;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                          covars[1], covars[3], covars[4], // 2nd column
                          covars[2], covars[4], covars[5]  // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
        quat_scale_to_covar_preci(glm::make_vec4(quats), glm::make_vec3(scales), &covar,
                                  nullptr);
    }
    glm::mat3 covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // project to spherical lidar
    glm::mat2 covar2d;
    glm::vec2 mean2d;
    glm::vec2 depth_compensation;
    glm::mat3 jacobian;
    lidar_proj(mean_c, covar_c, eps2d, mean2d, covar2d, depth_compensation, jacobian);

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0.f;
        return;
    }


    // take 3 sigma as the radius (non differentiable)
    // float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    // float v1 = b + sqrt(max(1e-6, b * b - det));
    // float radius = 3.f * sqrt(v1);
    // float v2 = b - sqrt(max(0.1f, b * b - det));
    // float radius = ceil(3.f * sqrt(max(v1, v2)));
    float extent_azimuth = 3.f * sqrt(max(0.f, covar2d[0][0]));
    float extent_elevation = 3.f * sqrt(max(0.f, covar2d[1][1]));

    if (extent_azimuth <= radius_clip && extent_elevation <= radius_clip) {
        radii[idx * 2] = 0.f;
        return;
    }

    // increase radius to compensate for rolling shutter
    glm::vec3 pix_vel = { 0.f, 0.f, 0.f };
    if (rs_time > 0) {
        // move velocities to lidar space
        glm::vec3 vel_c(0.f);
        if (velocities != nullptr) {
            glm::vec3 vel_w = glm::make_vec3(velocities + gid * 3);
            vel_world_to_cam(R, vel_w, vel_c);
        }
        compute_lidar_velocity(mean_c, glm::make_vec3(lin_vel), glm::make_vec3(ang_vel), vel_c, jacobian, pix_vel);
        extent_azimuth += fabs(pix_vel.x) * 0.5f * rs_time;
        extent_elevation += fabs(pix_vel.y) * 0.5f * rs_time;
    }

    if (mean2d.y + extent_elevation <= min_elevation 
        || mean2d.y - extent_elevation >= max_elevation
        || mean2d.x + extent_azimuth <= min_azimuth
        || mean2d.x - extent_azimuth >= max_azimuth) {
        radii[idx * 2] = 0.f;
        return;
    }

    // compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv;
    inverse(covar2d, covar2d_inv);

    // write to outputs
    radii[idx * 2] = extent_azimuth;
    radii[idx * 2 + 1] = extent_elevation;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = distance;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
    pix_vels[idx * 3] = pix_vel.x;
    pix_vels[idx * 3 + 1] = pix_vel.y;
    pix_vels[idx * 3 + 2] = pix_vel.z;
    depth_compensations[idx * 2] = depth_compensation.x;
    depth_compensations[idx * 2 + 1] = depth_compensation.y;
}

__global__ void fully_fused_lidar_projection_bwd_kernel(
    // fwd inputs
    const uint32_t C, const uint32_t N,
    const float *__restrict__ means,                // [N, 3]
    const float *__restrict__ covars,               // [N, 6] optional
    const float *__restrict__ quats,                // [N, 4] optional
    const float *__restrict__ scales,               // [N, 3] optional
    const float *__restrict__ velocities,           // [N, 3] optional
    const float *__restrict__ viewmats,             // [C, 4, 4]
    const float min_elevation,
    const float max_elevation,
    const float min_azimuth,
    const float max_azimuth,
    const float *__restrict__ lin_vel,              // [C, 3]
    const float *__restrict__ ang_vel,              // [C, 3]
    const float *__restrict__ rolling_shutter_time, // [C]
    const float eps2d,
    // fwd outputs
    const float *__restrict__ radii,                // [C, N, 2]
    const float *__restrict__ conics,               // [C, N, 3]
    const float *__restrict__ compensations,        // [C, N] optional
    // grad outputs
    const float *__restrict__ v_means2d,            // [C, N, 2]
    const float *__restrict__ v_depths,             // [C, N]
    const float *__restrict__ v_conics,             // [C, N, 3]
    const float *__restrict__ v_compensations,      // [C, N] optional
    const float *__restrict__ v_pix_vels,           // [C, N, 3]
    const float *__restrict__ v_depth_compensations, // [C, N, 2]
    // grad inputs
    float *__restrict__ v_means,                    // [N, 3]
    float *__restrict__ v_covars,                   // [N, 6] optional
    float *__restrict__ v_quats,                    // [N, 4] optional
    float *__restrict__ v_scales,                   // [N, 3] optional
    float *__restrict__ v_viewmats                  // [C, 4, 4] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx * 2] <= 0.f) {
        return;
    }
    const uint32_t cid = idx / N; // lidar id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current lidar and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    lin_vel += cid * 3;
    ang_vel += cid * 3;
    rolling_shutter_time += cid;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;
    v_pix_vels += idx * 3;
    v_depth_compensations += idx * 2;

    // vjp: compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv = glm::mat2(conics[0], conics[1], conics[1], conics[2]);
    glm::mat2 v_covar2d_inv =
        glm::mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    glm::mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(eps2d, covar2d_inv, compensation, v_compensation, v_covar2d);
    }

    // transform Gaussian to lidar space
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    glm::mat3 covar;
    glm::vec4 quat;
    glm::vec3 scale;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                          covars[1], covars[3], covars[4], // 2nd column
                          covars[2], covars[4], covars[5]  // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quat = glm::make_vec4(quats + gid * 4);
        scale = glm::make_vec3(scales + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    glm::vec3 mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    glm::mat3 covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // vjp: vel world to cam
    
    glm::vec3 v_p_view_pix_vel(0.f);
    glm::mat3 v_R(0.f);
    if (rolling_shutter_time[0] > 0 ) {
        glm::vec3 vel_c(0.f);
        glm::vec3 vel_w(0.f);
        glm::vec3 v_vel_c(0.f);
        if (velocities != nullptr) {
            vel_w = glm::make_vec3(velocities + gid * 3);
            vel_world_to_cam(R, vel_w, vel_c);
        }
        compute_and_sum_lidar_velocity_vjp(
            mean_c,
            glm::make_vec3(lin_vel),
            glm::make_vec3(ang_vel),
            vel_c,
            glm::make_vec3(v_pix_vels),
            v_p_view_pix_vel,
            v_vel_c);
        glm::vec3 v_vel_w(0.f);
        if (velocities != nullptr) {
            vel_world_to_cam_vjp(R, vel_w, v_vel_c, v_R, v_vel_w);
        }
    }

    // vjp: lidar projection
    glm::mat3 v_covar_c(0.f);
    glm::vec3 v_mean_c(0.f);
    lidar_proj_vjp(mean_c, covar_c, eps2d, glm::make_vec2(v_means2d), v_covar2d, glm::make_vec2(v_depth_compensations), v_mean_c, v_covar_c);

    // add contribution from v_depths
    const float disparity = rnorm3df(mean_c.x, mean_c.y, mean_c.z);
    v_mean_c.x += mean_c.x * disparity * v_depths[0];
    v_mean_c.y += mean_c.y * disparity * v_depths[0];
    v_mean_c.z += mean_c.z * disparity * v_depths[0];

    // add contribution from pix velocities
    v_mean_c.x += v_p_view_pix_vel.x;
    v_mean_c.y += v_p_view_pix_vel.y;
    v_mean_c.z += v_p_view_pix_vel.z;

    // vjp: transform Gaussian covariance to lidar space
    glm::vec3 v_mean(0.f);
    glm::mat3 v_covar(0.f);
    glm::vec3 v_t(0.f);
    pos_world_to_cam_vjp(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        }
    } else {
        // Directly output gradients w.r.t. the quaternion and scale
        glm::mat3 rotmat = quat_to_rotmat(quat);
        glm::vec4 v_quat(0.f);
        glm::vec3 v_scale(0.f);
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
        warpSum(v_quat, warp_group_g);
        warpSum(v_scale, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_quats += gid * 4;
            v_scales += gid * 3;
            gpuAtomicAdd(v_quats, v_quat[0]);
            gpuAtomicAdd(v_quats + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + 3, v_quat[3]);
            gpuAtomicAdd(v_scales, v_scale[0]);
            gpuAtomicAdd(v_scales + 1, v_scale[1]);
            gpuAtomicAdd(v_scales + 2, v_scale[2]);
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_lidar_projection_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3] optional
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const float min_elevation,
    const float max_elevation,
    const float min_azimuth,
    const float max_azimuth,
    const torch::Tensor &linear_velocity,      // [C, 3]
    const torch::Tensor &angular_velocity,     // [C, 3]
    const torch::Tensor &rolling_shutter_time, // [C]
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    if (velocities.has_value()) {
        CHECK_INPUT(velocities.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(linear_velocity);
    CHECK_INPUT(angular_velocity);
    CHECK_INPUT(rolling_shutter_time);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of lidars
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii = torch::empty({C, N, 2}, means.options());
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({C, N}, means.options());
    }
    torch::Tensor pix_vels = torch::empty({C, N, 3}, means.options());
    torch::Tensor depth_compensations = torch::empty({C, N, 2}, means.options());
    if (C && N) {
        fully_fused_lidar_projection_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS,
                                            N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
            scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(),
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity.data_ptr<float>(),
            angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            radii.data_ptr<float>(),
            means2d.data_ptr<float>(),
            depths.data_ptr<float>(),
            conics.data_ptr<float>(),
            calc_compensations ? compensations.data_ptr<float>() : nullptr,
            pix_vels.data_ptr<float>(),
            depth_compensations.data_ptr<float>()
            );
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations, pix_vels, depth_compensations);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_lidar_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                         // [N, 3]
    const at::optional<torch::Tensor> &covars,          // [N, 6] optional
    const at::optional<torch::Tensor> &quats,           // [N, 4] optional
    const at::optional<torch::Tensor> &scales,          // [N, 3] optional
    const at::optional<torch::Tensor> &velocities,      // [N, 3] optional
    const torch::Tensor &viewmats,                      // [C, 4, 4]
    const float min_elevation,
    const float max_elevation,
    const float min_azimuth,
    const float max_azimuth,
    const torch::Tensor &linear_velocity,               // [C, 3]
    const torch::Tensor &angular_velocity,              // [C, 3]
    const torch::Tensor &rolling_shutter_time,          // [C]
    const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,                         // [C, N, 2]
    const torch::Tensor &conics,                        // [C, N, 3]
    const at::optional<torch::Tensor> &compensations,   // [C, N] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [C, N, 2]
    const torch::Tensor &v_depths,                      // [C, N]
    const torch::Tensor &v_conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const torch::Tensor &v_pix_vels,                    // [C, N, 3]
    const torch::Tensor &v_depth_compensations,         // [C, N, 2]
    const bool viewmats_requires_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    if (velocities.has_value()) {
        CHECK_INPUT(velocities.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(linear_velocity);
    CHECK_INPUT(angular_velocity);
    CHECK_INPUT(rolling_shutter_time);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }
    CHECK_INPUT(v_pix_vels);
    CHECK_INPUT(v_depth_compensations);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of lidars
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars, v_quats, v_scales; // optional
    if (covars.has_value()) {
        v_covars = torch::zeros_like(covars.value());
    } else {
        v_quats = torch::zeros_like(quats.value());
        v_scales = torch::zeros_like(scales.value());
    }
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        fully_fused_lidar_projection_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS,
                                            N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(),
            covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
            covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
            covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
            velocities.has_value() ? velocities.value().data_ptr<float>() : nullptr,
            viewmats.data_ptr<float>(),
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity.data_ptr<float>(),
            angular_velocity.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            eps2d,
            radii.data_ptr<float>(),
            conics.data_ptr<float>(),
            compensations.has_value() ? compensations.value().data_ptr<float>()
                                      : nullptr,
            v_means2d.data_ptr<float>(),
            v_depths.data_ptr<float>(),
            v_conics.data_ptr<float>(),
            v_compensations.has_value() ? v_compensations.value().data_ptr<float>()
                                        : nullptr,
            v_pix_vels.data_ptr<float>(),
            v_depth_compensations.data_ptr<float>(),
            v_means.data_ptr<float>(),
            covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
            covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
            covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}
