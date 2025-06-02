#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define N_THREADS 256

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                 \
    CHECK_CUDA(x);                                                                     \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten)                                                             \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define CUB_WRAPPER(func, ...)                                                         \
    do {                                                                               \
        size_t temp_storage_bytes = 0;                                                 \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                                \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();           \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);            \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);                     \
    } while (false)

std::tuple<torch::Tensor, torch::Tensor>
quat_scale_to_covar_preci_fwd_tensor(const torch::Tensor &quats,  // [N, 4]
                                     const torch::Tensor &scales, // [N, 3]
                                     const bool compute_covar, const bool compute_preci,
                                     const bool triu);

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_bwd_tensor(
    const torch::Tensor &quats,                  // [N, 4]
    const torch::Tensor &scales,                 // [N, 3]
    const at::optional<torch::Tensor> &v_covars, // [N, 3, 3]
    const at::optional<torch::Tensor> &v_precis, // [N, 3, 3]
    const bool triu);

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_fwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const uint32_t width, const uint32_t height);

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_bwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const uint32_t width, const uint32_t height,
                      const torch::Tensor &v_means2d, // [C, N, 2]
                      const torch::Tensor &v_covars2d // [C, N, 2, 2]
);

std::tuple<torch::Tensor, torch::Tensor>
world_to_cam_fwd_tensor(const torch::Tensor &means,   // [N, 3]
                        const torch::Tensor &covars,  // [N, 3, 3]
                        const torch::Tensor &viewmats // [C, 4, 4]
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_cam_bwd_tensor(const torch::Tensor &means,                    // [N, 3]
                        const torch::Tensor &covars,                   // [N, 3, 3]
                        const torch::Tensor &viewmats,                 // [C, 4, 4]
                        const at::optional<torch::Tensor> &v_means_c,  // [C, N, 3]
                        const at::optional<torch::Tensor> &v_covars_c, // [C, N, 3, 3]
                        const bool means_requires_grad, const bool covars_requires_grad,
                        const bool viewmats_requires_grad);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
lidar_proj_fwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars,  // [C, N, 3, 3]
                      const float eps2d
);

std::tuple<torch::Tensor, torch::Tensor>
lidar_proj_bwd_tensor(const torch::Tensor &means,     // [C, N, 3]
                      const torch::Tensor &covars,    // [C, N, 3, 3]
                      const float eps2d,
                      const torch::Tensor &v_means2d, // [C, N, 2]
                      const torch::Tensor &v_covars2d, // [C, N, 2, 2]
                      const torch::Tensor &v_depth_compensations // [C, N, 2]
);

torch::Tensor compute_pix_velocity_fwd_tensor( 
    const torch::Tensor &p_view,  // [C, N, 3]
    const torch::Tensor &lin_vel, // [C, N, 3]
    const torch::Tensor &ang_vel,  // [C, N, 3]
    const torch::Tensor &v_view,  // [C, N, 3]
    const torch::Tensor &Ks,  // [C, 3, 3]
    const uint32_t width, 
    const uint32_t height
);

std::tuple<torch::Tensor, torch::Tensor>
compute_pix_velocity_bwd_tensor( 
    const torch::Tensor &p_view,  // [C, N, 3]
    const torch::Tensor &lin_vel, // [C, N, 3]
    const torch::Tensor &ang_vel,  // [C, N, 3]
    const torch::Tensor &v_view,  // [C, N, 3]
    const torch::Tensor &Ks,  // [C, 3, 3]
    const uint32_t width, 
    const uint32_t height,
    const torch::Tensor &v_pix_velocity  // [C, N, 3]
);

torch::Tensor compute_lidar_velocity_fwd_tensor( 
    const torch::Tensor &p_view,  // [C, N, 3]
    const torch::Tensor &lin_vel, // [C, 3]
    const torch::Tensor &ang_vel,  // [C, 3]
    const torch::Tensor &v_view  // [C, N, 3]
);

std::tuple<torch::Tensor, torch::Tensor>
compute_lidar_velocity_bwd_tensor( 
    const torch::Tensor &p_view,  // [C, N, 3]
    const torch::Tensor &lin_vel, // [C, N, 3]
    const torch::Tensor &ang_vel,  // [C, N, 3]
    const torch::Tensor &v_view,  // [C, N, 3]
    const torch::Tensor &v_spherical_velocity  // [C, N, 3]
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3] 
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
    const bool calc_compensations);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                         // [N, 3]
    const at::optional<torch::Tensor> &covars,          // [N, 6] optional
    const at::optional<torch::Tensor> &quats,           // [N, 4] optional
    const at::optional<torch::Tensor> &scales,          // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3] 
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
    const bool viewmats_requires_grad);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_lidar_projection_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3] 
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
    const bool calc_compensations);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_lidar_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                         // [N, 3]
    const at::optional<torch::Tensor> &covars,          // [N, 6] optional
    const at::optional<torch::Tensor> &quats,           // [N, 4] optional
    const at::optional<torch::Tensor> &scales,          // [N, 3] optional
    const at::optional<torch::Tensor> &velocities, // [N, 3] 
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
    const torch::Tensor &v_pix_vels,                    // [C, N, 2]
    const torch::Tensor &v_depth_compensations,         // [C, N, 2]
    const bool viewmats_requires_grad);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_tensor(const torch::Tensor &means2d, // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &radii,   // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &depths,  // [C, N] or [nnz]
                   const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                   const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
                   const uint32_t C, const uint32_t tile_size,
                   const uint32_t tile_width, const uint32_t tile_height,
                   const bool sort, const bool double_buffer);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_lidar_tiles_tensor(const torch::Tensor &means2d, // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &radii,   // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &depths,  // [C, N] or [nnz]
                   const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                   const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
                   const uint32_t C,
                   const torch::Tensor &elev_boundaries,
                   const float tile_azim_resolution,
                   const float min_azim,
                   const bool sort, const bool double_buffer);

torch::Tensor isect_offset_encode_tensor(const torch::Tensor &isect_ids, // [n_isects]
                                         const uint32_t C, const uint32_t tile_width,
                                         const uint32_t tile_height);

std::tuple<torch::Tensor, torch::Tensor>
map_points_to_lidar_tiles_tensor(
    const torch::Tensor &points2d, // [C, N, 2] or [nnz, 2]
    const at::optional<torch::Tensor> &camera_ids,   // [nnz]
    const uint32_t C,
    const torch::Tensor &elev_boundaries, // [M] 
    const float tile_azim_resolution, // [1]
    const float min_azim, // [1]
    const bool sort, const bool double_buffer
);

torch::Tensor points_mapping_offset_encode_tensor(const torch::Tensor &point_ids, // [n_points]
                                         const uint32_t C, const uint32_t tile_width,
                                         const uint32_t tile_height);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_to_pixels_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, D]
    const torch::Tensor &opacities,                 // [N]
    const torch::Tensor &pix_vels,                  // [C, N, 2]
    const torch::Tensor &rolling_shutter_time,      // [C]
    const at::optional<torch::Tensor> &backgrounds, // [C, D]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);

torch::Tensor populate_image_from_points_tensor(
    // Points to fill the raster image with
    const torch::Tensor &points,   // [C, N, 5] or [nnz, 5]
    // image size
    const uint32_t image_width, const uint32_t image_height, 
    const uint32_t tile_width, const uint32_t tile_height,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_to_points_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, D]
    const torch::Tensor &opacities,                 // [N]
    const torch::Tensor &pix_vels,                  // [C, N, 3]
    const torch::Tensor &depth_compensations,     // [C, N, 2]
    const at::optional<torch::Tensor> &backgrounds, // [C, D]
    // Points to rasterize
    const torch::Tensor &raster_pts, // [C, image_height, image_width, 4]
    // image size
    const uint32_t image_width, const uint32_t image_height, 
    const uint32_t tile_width, const uint32_t tile_height,
    // compute alphas until point
    const bool compute_alpha_sum_until_points,
    const float compute_alpha_sum_until_points_threshold,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    // depth channel index
    const uint32_t depth_channel_idx
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, 3]
    const torch::Tensor &opacities,                 // [N]
    const torch::Tensor &pix_vels,                  // [C, N, 2]
    const torch::Tensor &rolling_shutter_time,      // [C]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    // image size
    const uint32_t image_width, const uint32_t image_height, const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    // options
    bool absgrad);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_points_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &colors,                    // [C, N, 3]
    const torch::Tensor &opacities,                 // [N]
    const torch::Tensor &pix_vels,                  // [C, N, 3]
    const torch::Tensor &depth_compensations,       // [C, N, 2]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    // Points to rasterize
    const torch::Tensor &raster_pts, // [C, image_height, image_width, 4]
    // image size
    const uint32_t image_width, const uint32_t image_height, 
    const uint32_t tile_width, const uint32_t tile_height,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
    const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &v_alpha_sum_until_points, // [C, image_height, image_width, 1])
    // options
    bool absgrad,
    const bool compute_alpha_sum_until_points,
    const float compute_alpha_sum_until_points_threshold,
    const uint32_t depth_channel_idx
);

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_tensor(
    const uint32_t range_start, const uint32_t range_end, // iteration steps
    const torch::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const torch::Tensor &means2d,              // [C, N, 2]
    const torch::Tensor &conics,               // [C, N, 3]
    const torch::Tensor &opacities,            // [N]
    const torch::Tensor &pix_vels,             // [C, N, 2]
    const torch::Tensor &rolling_shutter_time, // [C]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_lidar_tensor(
    const uint32_t range_start, const uint32_t range_end, // iteration steps
    const torch::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &opacities, // [N]
    const torch::Tensor &pix_vels,             // [C, N, 3]
    // Points to rasterize
    const torch::Tensor &raster_pts,  // [C, image_height, image_width, 4]
    // image size
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_width, const uint32_t tile_height,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
);


torch::Tensor compute_sh_fwd_tensor(const uint32_t degrees_to_use,
                                    torch::Tensor &dirs,              // [..., 3]
                                    torch::Tensor &coeffs,            // [..., K, 3]
                                    at::optional<torch::Tensor> masks // [...]
);
std::tuple<torch::Tensor, torch::Tensor>
compute_sh_bwd_tensor(const uint32_t K, const uint32_t degrees_to_use,
                      torch::Tensor &dirs,               // [..., 3]
                      torch::Tensor &coeffs,             // [..., K, 3]
                      at::optional<torch::Tensor> masks, // [...]
                      torch::Tensor &v_colors,           // [..., 3]
                      bool compute_v_dirs);

/****************************************************************************************
 * Packed Version
 ****************************************************************************************/
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
    const bool calc_compensations);

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
    const bool viewmats_requires_grad, const bool sparse_grad);

std::tuple<torch::Tensor, torch::Tensor>
compute_relocation_tensor(
    torch::Tensor& opacities,
    torch::Tensor& scales,
    torch::Tensor& ratios,
    torch::Tensor& binoms,
    const int n_max
);
