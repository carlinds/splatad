#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"
#include <cmath>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/

__global__ void isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C, const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const float2 *__restrict__ means2d,              // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N, 2] or [nnz, 2]
    const float *__restrict__ depths,                // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N))
        return;
    if (radii[idx * 2] <= 0) {
        if (first_pass)
            tiles_per_gauss[idx] = 0;
        return;
    }

    float tile_extent_x = radii[idx * 2] / static_cast<float>(tile_size);
    float tile_extent_y = radii[idx * 2 + 1] / static_cast<float>(tile_size);
    float tile_x = means2d[idx].x / tile_size;
    float tile_y = means2d[idx].y / tile_size;

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_extent_x)), tile_width);
    tile_min.y = min(max(0, (uint32_t)floor(tile_y - tile_extent_y)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_extent_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_extent_y)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] =
            static_cast<int32_t>((tile_max.y - tile_min.y) * (tile_max.x - tile_min.x));
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_tensor(const torch::Tensor &means2d, // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &radii,   // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &depths,  // [C, N] or [nnz]
                   const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                   const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
                   const uint32_t C, const uint32_t tile_size,
                   const uint32_t tile_width, const uint32_t tile_height,
                   const bool sort, const bool double_buffer) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    if (camera_ids.has_value()) {
        CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz, total_elems;
    int64_t *camera_ids_ptr;
    int64_t *gaussian_ids_ptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        assert(camera_ids.has_value() && gaussian_ids.has_value());
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so check if
    // we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    torch::Tensor tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));

    int64_t n_isects;
    torch::Tensor cum_tiles_per_gauss;
    if (total_elems) {
        isect_tiles<<<(total_elems + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                      stream>>>(
            packed, C, N, nnz, camera_ids_ptr, gaussian_ids_ptr,
            (float2 *)means2d.data_ptr<float>(), radii.data_ptr<int32_t>(),
            depths.data_ptr<float>(), nullptr, tile_size, tile_width, tile_height,
            tile_n_bits, tiles_per_gauss.data_ptr<int32_t>(), nullptr, nullptr);
        cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        isect_tiles<<<(total_elems + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                      stream>>>(
            packed, C, N, nnz, camera_ids_ptr, gaussian_ids_ptr,
            (float2 *)means2d.data_ptr<float>(), radii.data_ptr<int32_t>(),
            depths.data_ptr<float>(), cum_tiles_per_gauss.data_ptr<int64_t>(),
            tile_size, tile_width, tile_height, tile_n_bits, nullptr,
            isect_ids.data_ptr<int64_t>(), flatten_ids.data_ptr<int32_t>());
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(isect_ids.data_ptr<int64_t>(),
                                              isect_ids_sorted.data_ptr<int64_t>());
            cub::DoubleBuffer<int32_t> d_values(flatten_ids.data_ptr<int32_t>(),
                                                flatten_ids_sorted.data_ptr<int32_t>());
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, d_keys, d_values, n_isects, 0,
                        32 + tile_n_bits + cam_n_bits, stream);
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n", d_values.selector);
        } else {
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, isect_ids.data_ptr<int64_t>(),
                        isect_ids_sorted.data_ptr<int64_t>(),
                        flatten_ids.data_ptr<int32_t>(),
                        flatten_ids_sorted.data_ptr<int32_t>(), n_isects, 0,
                        32 + tile_n_bits + cam_n_bits, stream);
        }
        return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

__global__ void isect_offset_encode(const uint32_t n_isects,
                                    const int64_t *__restrict__ isect_ids,
                                    const uint32_t C, const uint32_t n_tiles,
                                    const uint32_t tile_n_bits,
                                    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid, tile_id)
        // pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t cid_prev = isect_id_prev >> tile_n_bits;
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

torch::Tensor isect_offset_encode_tensor(const torch::Tensor &isect_ids, // [n_isects]
                                         const uint32_t C, const uint32_t tile_width,
                                         const uint32_t tile_height) {
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    uint32_t n_isects = isect_ids.size(0);
    torch::Tensor offsets = torch::empty({C, tile_height, tile_width},
                                         isect_ids.options().dtype(torch::kInt32));
    if (n_isects) {
        uint32_t n_tiles = tile_width * tile_height;
        uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        isect_offset_encode<<<(n_isects + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                              stream>>>(n_isects, isect_ids.data_ptr<int64_t>(), C,
                                        n_tiles, tile_n_bits,
                                        offsets.data_ptr<int32_t>());
    } else {
        offsets.fill_(0);
    }
    return offsets;
}

__global__ void isect_lidar_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C, const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const float2 *__restrict__ means2d,              // [C, N, 2] or [nnz, 2]
    const float *__restrict__ radii,               // [C, N, 2] or [nnz, 2]
    const float *__restrict__ depths,                // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const float *__restrict__ elev_boundaries,      // [M]
    const float tile_azim_resolution, // [1]
    const float min_azim, // [1]
    const uint32_t n_tiles_azim, const uint32_t n_tiles_elev,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N))
        return;
    if (radii[idx * 2] <= 0) {
        if (first_pass)
            tiles_per_gauss[idx] = 0;
        return;
    }

    float tile_elev = means2d[idx].y;
    float tile_elev_low = tile_elev - radii[idx * 2 + 1];
    float tile_elev_high = tile_elev + radii[idx * 2 + 1];

    float azim = means2d[idx].x - min_azim;
    float azim_low = azim - radii[idx * 2];
    float azim_high = azim + radii[idx * 2];
    float azim_max = static_cast<float>(n_tiles_azim) * tile_azim_resolution;
    float tile_azim_low;
    if (azim_low >= 0) {
        tile_azim_low = azim_low / tile_azim_resolution;
    } else {
        tile_azim_low = (fmodf((azim_low + 360.f), 360.f) - azim_max) / tile_azim_resolution;
    }
    float tile_azim_high;
    if (azim_high <= 360.f) {
        tile_azim_high = azim_high / tile_azim_resolution;
    } else {
        tile_azim_high = static_cast<float>(n_tiles_azim) + (fmodf((azim_high + 360.f), 360.f)) / tile_azim_resolution;
    }

    // tile_min is inclusive, tile_max is exclusive
    int2 tile_min, tile_max;
    tile_min.x = static_cast<int32_t>(floor(tile_azim_low));
    tile_max.x = static_cast<int32_t>(ceil(tile_azim_high));

    int32_t elev_min=0, elev_max=n_tiles_elev, i=0;
    while (i <= n_tiles_elev && elev_boundaries[i] < tile_elev_low) {
        ++i;
    }
    elev_min = max(i-1,0);
    while (i <= n_tiles_elev && elev_boundaries[i] < tile_elev_high) {
        ++i;
    }
    elev_max = min(i, n_tiles_elev);
    tile_min.y = elev_min;
    tile_max.y = elev_max;

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] =
            static_cast<int32_t>((tile_max.y - tile_min.y) * (tile_max.x - tile_min.x));
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }

    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            // wrap j to [0, n_tiles_azim)
            int32_t wrapped_j = (j + n_tiles_azim) % n_tiles_azim;

            int64_t tile_id = i * n_tiles_azim + wrapped_j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_lidar_tiles_tensor(const torch::Tensor &means2d, // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &radii,   // [C, N, 2] or [nnz, 2]
                   const torch::Tensor &depths,  // [C, N] or [nnz]
                   const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                   const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
                   const uint32_t C,
                   const torch::Tensor &elev_boundaries, // [M] 
                   const float tile_azim_resolution, // [1]
                   const float min_azim, // [1]
                   const bool sort, const bool double_buffer) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    if (camera_ids.has_value()) {
        CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        CHECK_INPUT(gaussian_ids.value());
    }
    CHECK_INPUT(elev_boundaries);
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz, total_elems;
    int64_t *camera_ids_ptr;
    int64_t *gaussian_ids_ptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        assert(camera_ids.has_value() && gaussian_ids.has_value());
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles_azim = (uint32_t)ceil(360.f / tile_azim_resolution);
    uint32_t n_tiles_elev = (uint32_t)(elev_boundaries.size(0) - 1);
    uint32_t n_tiles = n_tiles_azim * n_tiles_elev;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so check if
    // we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    torch::Tensor tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));

    int64_t n_isects;
    torch::Tensor cum_tiles_per_gauss;
    if (total_elems) {
        isect_lidar_tiles<<<(total_elems + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                      stream>>>(
            packed, C, N, nnz, camera_ids_ptr, gaussian_ids_ptr,
            (float2 *)means2d.data_ptr<float>(), radii.data_ptr<float>(),
            depths.data_ptr<float>(), nullptr, elev_boundaries.data_ptr<float>(),
            tile_azim_resolution, min_azim, n_tiles_azim, n_tiles_elev,
            tile_n_bits, tiles_per_gauss.data_ptr<int32_t>(), nullptr, nullptr);
        cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        isect_lidar_tiles<<<(total_elems + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                      stream>>>(
            packed, C, N, nnz, camera_ids_ptr, gaussian_ids_ptr,
            (float2 *)means2d.data_ptr<float>(), radii.data_ptr<float>(),
            depths.data_ptr<float>(), cum_tiles_per_gauss.data_ptr<int64_t>(),
            elev_boundaries.data_ptr<float>(), 
            tile_azim_resolution, min_azim, n_tiles_azim, n_tiles_elev,
            tile_n_bits, nullptr,
            isect_ids.data_ptr<int64_t>(), flatten_ids.data_ptr<int32_t>());
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(isect_ids.data_ptr<int64_t>(),
                                              isect_ids_sorted.data_ptr<int64_t>());
            cub::DoubleBuffer<int32_t> d_values(flatten_ids.data_ptr<int32_t>(),
                                                flatten_ids_sorted.data_ptr<int32_t>());
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, d_keys, d_values, n_isects, 0,
                        32 + tile_n_bits + cam_n_bits, stream);
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n", d_values.selector);
        } else {
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, isect_ids.data_ptr<int64_t>(),
                        isect_ids_sorted.data_ptr<int64_t>(),
                        flatten_ids.data_ptr<int32_t>(),
                        flatten_ids_sorted.data_ptr<int32_t>(), n_isects, 0,
                        32 + tile_n_bits + cam_n_bits, stream);
        }
        return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

__global__ void points_mapping_offset_encode(
    const uint32_t n_points,
    const int64_t *__restrict__ point_ids,
    const uint32_t C, const uint32_t n_tiles,
    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_points)
        return;

    const int64_t point_id_curr = point_ids[idx];
    const int64_t cid_curr = point_id_curr >> 32;
    const int64_t tid_curr = point_id_curr & ((1ULL << 32) - 1);
    const int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i) 
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_points - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_points);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid, tile_id)
        // pair changes.
        const int64_t point_id_prev = point_ids[idx - 1];
        if (point_id_prev == point_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        const int64_t cid_prev = point_id_prev >> 32;
        const int64_t tid_prev = point_id_prev & ((1ULL << 32) - 1);
        const int64_t id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

torch::Tensor points_mapping_offset_encode_tensor(const torch::Tensor &point_ids, // [n_points]
                                         const uint32_t C, const uint32_t tile_width,
                                         const uint32_t tile_height) {
    DEVICE_GUARD(point_ids);
    CHECK_INPUT(point_ids);

    uint32_t n_points = point_ids.size(0);
    torch::Tensor offsets = torch::empty({C, tile_height, tile_width},
                                         point_ids.options().dtype(torch::kInt32));

    if (n_points) {
        uint32_t n_tiles = tile_width * tile_height;
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        points_mapping_offset_encode<<<(n_points + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                              stream>>>(n_points, point_ids.data_ptr<int64_t>(), C,
                                        n_tiles, offsets.data_ptr<int32_t>());
    } else {
        offsets.fill_(0);
    }
    return offsets;
}
__global__ void map_points_to_lidar_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C, const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    // data
    const float2 *__restrict__ points2d,              // [C, N, 2] or [nnz, 2]
    const float *__restrict__ elev_boundaries,      // [M]
    const float tile_azim_resolution, // [1]
    const float min_azim, // [1]
    const uint32_t n_tiles_azim, const uint32_t n_tiles_elev,
    int64_t *__restrict__ tile_ids, // [C * N]
    int32_t *__restrict__ flatten_ids // [C * N]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    // bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N))
        return;

    float tile_elev = points2d[idx].y;
    float azim = points2d[idx].x - min_azim;
    
    float tile_azim;
    if ((azim >= 0.f) && (azim < 360.f)) {
        tile_azim = azim / tile_azim_resolution;
    } else {
        tile_azim = fmodf((azim + 360.f), 360.f) / tile_azim_resolution;
    }

    int2 tile_coord;
    tile_coord.x = static_cast<int32_t>(tile_azim);

    int32_t i=0;
    while (i < n_tiles_elev && elev_boundaries[i] < tile_elev) {
        ++i;
    }
    tile_coord.y = max(i-1,0);

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
    }

    // create encoding from camera id (first 32 bits) and tile id (last 32 bits) [camera_id | tile_id]
    const int64_t cid_enc = cid << 32;

    // wrap j to [0, n_tiles_azim)
    const int64_t tile_id = tile_coord.y * n_tiles_azim + tile_coord.x;
    
    tile_ids[idx] = cid_enc | tile_id;
    flatten_ids[idx] = idx;
}

std::tuple<torch::Tensor, torch::Tensor>
map_points_to_lidar_tiles_tensor(const torch::Tensor &points2d, // [C, N, 2] or [nnz, 2]
                   const at::optional<torch::Tensor> &camera_ids,   // [nnz]
                   const uint32_t C,
                   const torch::Tensor &elev_boundaries, // [M] 
                   const float tile_azim_resolution, // [1]
                   const float min_azim, // [1]
                   const bool sort, const bool double_buffer
) {
    DEVICE_GUARD(points2d);
    CHECK_INPUT(points2d);
    if (camera_ids.has_value()) {
        CHECK_INPUT(camera_ids.value());
    }
    CHECK_INPUT(elev_boundaries);
    bool packed = points2d.dim() == 2;

    uint32_t N, nnz, total_elems;
    int64_t *camera_ids_ptr;
    if (packed) {
        nnz = points2d.size(0);
        total_elems = nnz;
        assert(camera_ids.has_value());
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
    } else {
        N = points2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles_azim = (uint32_t)ceil(360.f / tile_azim_resolution);
    uint32_t n_tiles_elev = (uint32_t)(elev_boundaries.size(0) - 1);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor tile_ids =
        torch::empty({total_elems}, points2d.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({total_elems}, points2d.options().dtype(torch::kInt32));
    if (total_elems) {
        map_points_to_lidar_tiles<<<(total_elems + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                      stream>>>(
            packed, C, N, nnz, camera_ids_ptr,
            (float2 *)points2d.data_ptr<float>(), elev_boundaries.data_ptr<float>(),
            tile_azim_resolution, min_azim, n_tiles_azim, n_tiles_elev,
            tile_ids.data_ptr<int64_t>(), flatten_ids.data_ptr<int32_t>());
    }

    if (sort) {
        torch::Tensor tile_ids_sorted = torch::empty_like(tile_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(tile_ids.data_ptr<int64_t>(),
                                              tile_ids_sorted.data_ptr<int64_t>());
            cub::DoubleBuffer<int32_t> d_values(flatten_ids.data_ptr<int32_t>(),
                                                flatten_ids_sorted.data_ptr<int32_t>());
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, d_keys, d_values, total_elems, 0,
                        64, stream);
            switch (d_keys.selector) {
            case 0: // sorted items are stored in tile_ids
                tile_ids_sorted = tile_ids;
                break;
            case 1: // sorted items are stored in tile_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n", d_values.selector);
        } else {
            CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, tile_ids.data_ptr<int64_t>(),
                        tile_ids_sorted.data_ptr<int64_t>(),
                        flatten_ids.data_ptr<int32_t>(),
                        flatten_ids_sorted.data_ptr<int32_t>(), total_elems, 0,
                        64, stream);
        }
        return std::make_tuple(tile_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(tile_ids, flatten_ids);
    }
}

__global__ void populate_image_from_points_kernel(
    const uint32_t C, const uint32_t n_points,
    const float *__restrict__ points,    // [C, N, 5] or [nnz, 5]
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_width, const uint32_t tile_height,
    const uint32_t tile_grid_width, const uint32_t tile_grid_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    float *__restrict__ raster_pts // [C, image_height, image_width, 5]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    const int32_t camera_id = block.group_index().x;
    const int32_t tile_id = block.group_index().y * tile_grid_width + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_height + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_width + block.thread_index().x;
    tile_offsets += camera_id * tile_grid_height * tile_grid_width;
    raster_pts += camera_id * image_height * image_width * 5;
    const int32_t pix_idx =  i * image_width + j;

    // return if out of bounds
    if (i >= image_height || j >= image_width)
        return;

    const int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_grid_width * tile_grid_height - 1)
            ? n_points
            : tile_offsets[tile_id + 1];
    range_end = min(range_end, range_start + block.size());

    uint32_t tr = block.thread_rank();
    const int32_t flattened_point_idx = range_start + tr;
    if (flattened_point_idx >= range_end)
        return;

    const int32_t point_idx = flatten_ids[flattened_point_idx];
    // write out, points has 5 dims
    raster_pts[pix_idx * 5] = points[point_idx * 5];
    raster_pts[pix_idx * 5 + 1] = points[point_idx * 5 + 1];
    raster_pts[pix_idx * 5 + 2] = points[point_idx * 5 + 2];
    raster_pts[pix_idx * 5 + 3] = points[point_idx * 5 + 3];
    raster_pts[pix_idx * 5 + 4] = points[point_idx * 5 + 4];
}

torch::Tensor populate_image_from_points_tensor(
    // Points to fill the raster image with
    const torch::Tensor &points,   // [C, N, 5] or [nnz, 5]
    // image size
    const uint32_t image_width, const uint32_t image_height, 
    const uint32_t tile_width, const uint32_t tile_height,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(points);
    CHECK_INPUT(points);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    uint32_t C = tile_offsets.size(0);         // number of cameras

    uint32_t tile_grid_height = tile_offsets.size(1);
    uint32_t tile_grid_width = tile_offsets.size(2);
    uint32_t n_points = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_grid_height * tile_grid_width blocks.
    dim3 threads = {tile_width, tile_height, 1};
    dim3 blocks = {C, tile_grid_height, tile_grid_width};

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor raster_pts_image =
        torch::zeros({C, image_height, image_width, 5}, points.options().dtype(torch::kFloat32));
    populate_image_from_points_kernel<<<blocks, threads, 0, stream>>>(
        C, n_points, (float *)points.data_ptr<float>(),
        image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
        tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        (float *)raster_pts_image.data_ptr<float>());
        
    return raster_pts_image;
}

/****************************************************************************
 * Rasterization
 ****************************************************************************/

__global__ void rasterize_to_indices_in_range_kernel(
    const uint32_t range_start, const uint32_t range_end,
    const uint32_t C, const uint32_t N, const uint32_t n_isects,
    const float2 *__restrict__ means2d,             // [C, N, 2]
    const float3 *__restrict__ conics,              // [C, N, 3]
    const float *__restrict__ opacities,            // [C, N]
    const float2 *__restrict__ pix_vels,            // [C, N, 2] or [nnz, 2]
    const float *__restrict__ rolling_shutter_time, // [C]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const float *__restrict__ transmittances, // [C, image_height, image_width]
    const int32_t *__restrict__ chunk_starts, // [C, image_height, image_width]
    int32_t *__restrict__ chunk_cnts,         // [C, image_height, image_width]
    int64_t *__restrict__ gaussian_ids,       // [n_elems]
    int64_t *__restrict__ pixel_ids           // [n_elems]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // move pointers to the current camera
    tile_offsets += camera_id * tile_height * tile_width;
    transmittances += camera_id * image_height * image_width;

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;
    float roll_time = rolling_shutter_time[camera_id] * ((py - 0.5f) / (image_height - 1) - 0.5f);

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    bool first_pass = chunk_starts == nullptr;
    int32_t base;
    if (!first_pass && inside) {
        chunk_starts += camera_id * image_height * image_width;
        base = chunk_starts[pix_id];
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t isect_range_start = tile_offsets[tile_id];
    int32_t isect_range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (isect_range_end - isect_range_start + block_size - 1) / block_size;

    if (range_start >= num_batches) {
        // this entire tile has been processed in the previous iterations
        // so we don't need to do anything.
        return;
    }

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                              // [block_size]
    float3 *xy_opacity_batch = (float3 *)&id_batch[block_size];    // [block_size]
    float3 *conic_batch = (float3 *)&xy_opacity_batch[block_size]; // [block_size]
    float2 *pix_vel_batch = (float2 *)&conic_batch[block_size];    // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we (should) use double for it. However double make bwd
    // 1.5x slower so we stick with float for now.
    float T, next_T;
    if (inside) {
        T = transmittances[pix_id];
        next_T = T;
    }

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    int32_t cnt = 0;
    for (uint32_t b = range_start; b < min(range_end, num_batches); ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = isect_range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < isect_range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            const float2 pix_vel = pix_vels[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            pix_vel_batch[tr] = {pix_vel.x, pix_vel.y};
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, isect_range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 pix_vel = pix_vel_batch[t];
            const float2 delta = {
                xy_opac.x + roll_time * pix_vel.x - px,
                xy_opac.y + roll_time * pix_vel.y - py
            };
            const float sigma =
                0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
            float alpha = min(0.999f, opac * __expf(-sigma));

            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            if (first_pass) {
                // First pass of this function we count the number of gaussians
                // that contribute to each pixel
                cnt += 1;
            } else {
                // Second pass we write out the gaussian ids and pixel ids
                int32_t g = id_batch[t]; // flatten index in [C * N]
                gaussian_ids[base + cnt] = g % N;
                pixel_ids[base + cnt] = pix_id + camera_id * image_height * image_width;
                cnt += 1;
            }

            T = next_T;
        }
    }

    if (inside && first_pass) {
        chunk_cnts += camera_id * image_height * image_width;
        chunk_cnts[pix_id] = cnt;
    }
}

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_tensor(
    const uint32_t range_start, const uint32_t range_end, // iteration steps
    const torch::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const torch::Tensor &means2d,              // [C, N, 2]
    const torch::Tensor &conics,               // [C, N, 3]
    const torch::Tensor &opacities,            // [C, N]
    const torch::Tensor &pix_vels,             // [C, N, 2]
    const torch::Tensor &rolling_shutter_time, // [C]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(pix_vels);
    CHECK_INPUT(rolling_shutter_time);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    uint32_t C = means2d.size(0); // number of cameras
    uint32_t N = means2d.size(1); // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(float3) + sizeof(float3) + sizeof(float2));
    if (cudaFuncSetAttribute(rasterize_to_indices_in_range_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                 " bytes), try lowering tile_size.");
    }

    // First pass: count the number of gaussians that contribute to each pixel
    int64_t n_elems;
    torch::Tensor chunk_starts;
    if (n_isects) {
        torch::Tensor chunk_cnts = torch::zeros({C * image_height * image_width},
                                                means2d.options().dtype(torch::kInt32));
        rasterize_to_indices_in_range_kernel<<<blocks, threads, shared_mem, stream>>>(
            range_start, range_end, C, N, n_isects, (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(), opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            transmittances.data_ptr<float>(), nullptr, chunk_cnts.data_ptr<int32_t>(),
            nullptr, nullptr);

        torch::Tensor cumsum = torch::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
        n_elems = cumsum[-1].item<int64_t>();
        chunk_starts = cumsum - chunk_cnts;
    } else {
        n_elems = 0;
    }

    // Second pass: allocate memory and write out the gaussian and pixel ids.
    torch::Tensor gaussian_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    torch::Tensor pixel_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    if (n_elems) {
        rasterize_to_indices_in_range_kernel<<<blocks, threads, shared_mem, stream>>>(
            range_start, range_end, C, N, n_isects, (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(), opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            transmittances.data_ptr<float>(), chunk_starts.data_ptr<int32_t>(), nullptr,
             gaussian_ids.data_ptr<int64_t>(), pixel_ids.data_ptr<int64_t>());
    }
    return std::make_tuple(gaussian_ids, pixel_ids);
}

__global__ void rasterize_to_indices_in_range_lidar_kernel(
    const uint32_t range_start, const uint32_t range_end, const uint32_t C,
    const uint32_t N, const uint32_t n_isects,
    const float2 *__restrict__ means2d,  // [C, N, 2]
    const float3 *__restrict__ conics,   // [C, N, 3]
    const float *__restrict__ opacities, // [C, N]
    const float3 *__restrict__ pix_vels, // [C, N, 3] or [nnz, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_grid_width,
    const uint32_t tile_grid_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_grid_height, tile_grid_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const float *__restrict__ transmittances, // [C, image_height, image_width]
    const int32_t *__restrict__ chunk_starts, // [C, image_height, image_width]
    const float4 *__restrict__ raster_pts,    // [C, image_height, image_width, 4]
    int32_t *__restrict__ chunk_cnts,         // [C, image_height, image_width]
    int64_t *__restrict__ gaussian_ids,       // [n_elems]
    int64_t *__restrict__ pixel_ids           // [n_elems]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id = block.group_index().y * tile_grid_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_height + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_width + block.thread_index().x;

    // move pointers to the current camera
    tile_offsets += camera_id * tile_grid_height * tile_grid_width;
    transmittances += camera_id * image_height * image_width;

    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    bool first_pass = chunk_starts == nullptr;
    int32_t base;
    if (!first_pass && inside) {
        chunk_starts += camera_id * image_height * image_width;
        base = chunk_starts[pix_id];
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t isect_range_start = tile_offsets[tile_id];
    int32_t isect_range_end =
        (camera_id == C - 1) && (tile_id == tile_grid_width * tile_grid_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (isect_range_end - isect_range_start + block_size - 1) / block_size;

    if (range_start >= num_batches) {
        // this entire tile has been processed in the previous iterations
        // so we don't need to do anything.
        return;
    }

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                              // [block_size]
    float3 *xy_opacity_batch = (float3 *)&id_batch[block_size];    // [block_size]
    float3 *conic_batch = (float3 *)&xy_opacity_batch[block_size]; // [block_size]
    float3 *pix_vel_batch = (float3 *)&conic_batch[block_size];    // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we (should) use double for it. However double make bwd
    // 1.5x slower so we stick with float for now.
    float T, next_T;
    if (inside) {
        T = transmittances[pix_id];
        next_T = T;
    }

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    int32_t cnt = 0;
    for (uint32_t b = range_start; b < min(range_end, num_batches); ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = isect_range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < isect_range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            const float3 pix_vel = pix_vels[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            pix_vel_batch[tr] = {pix_vel.x, pix_vel.y, pix_vel.z};
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        if (!inside) {
            continue;
        }

        // process gaussians in the current batch for this pixel
        float px = raster_pts[(camera_id * image_height + i) * image_width + j].x; // azimuth
        float py = raster_pts[(camera_id * image_height + i) * image_width + j].y; // elevation
        float pz = raster_pts[(camera_id * image_height + i) * image_width + j].z; // range
        float roll_time = raster_pts[(camera_id * image_height + i) * image_width + j].w; // roll time

        if (pz <= 0.f) {
            inside = false;
            done = true;
            continue;
        }

        uint32_t batch_size = min(block_size, isect_range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float3 pix_vel = pix_vel_batch[t];
            const float2 delta = {
                angle_difference(xy_opac.x + roll_time * pix_vel.x, px),
                angle_difference(xy_opac.y + roll_time * pix_vel.y, py)
            };
            const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
            float alpha = min(0.999f, opac * __expf(-sigma));

            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            if (first_pass) {
                // First pass of this function we count the number of gaussians
                // that contribute to each pixel
                cnt += 1;
            } else {
                // Second pass we write out the gaussian ids and pixel ids
                int32_t g = id_batch[t]; // flatten index in [C * N]
                gaussian_ids[base + cnt] = g % N;
                pixel_ids[base + cnt] = pix_id + camera_id * image_height * image_width;
                cnt += 1;
            }

            T = next_T;
        }
    }

    if (inside && first_pass) {
        chunk_cnts += camera_id * image_height * image_width;
        chunk_cnts[pix_id] = cnt;
    }
}

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_lidar_tensor(
    const uint32_t range_start, const uint32_t range_end, // iteration steps
    const torch::Tensor transmittances, // [C, image_height, image_width]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &opacities, // [C, N]
    const torch::Tensor &pix_vels,             // [C, N, 3]
    // Points to rasterize
    const torch::Tensor &raster_pts,  // [C, image_height, image_width, 3]
    // image size
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_width, const uint32_t tile_height,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(pix_vels);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(raster_pts);

    uint32_t C = means2d.size(0); // number of cameras
    uint32_t N = means2d.size(1); // number of gaussians
    uint32_t tile_grid_height = tile_offsets.size(1);
    uint32_t tile_grid_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_grid_height * tile_grid_width blocks.
    dim3 threads = {tile_width, tile_height, 1};
    dim3 blocks = {C, tile_grid_height, tile_grid_width};

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_width * tile_height * (sizeof(int32_t) + sizeof(float3) + sizeof(float3) + sizeof(float3));
    if (cudaFuncSetAttribute(rasterize_to_indices_in_range_lidar_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                 " bytes), try lowering tile_size.");
    }

    // First pass: count the number of gaussians that contribute to each pixel
    int64_t n_elems;
    torch::Tensor chunk_starts;
    if (n_isects) {
        torch::Tensor chunk_cnts = torch::zeros({C * image_height * image_width},
                                                means2d.options().dtype(torch::kInt32));
        rasterize_to_indices_in_range_lidar_kernel<<<blocks, threads, shared_mem, stream>>>(
            range_start, range_end, C, N, n_isects, (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(), opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            transmittances.data_ptr<float>(), nullptr, (float4 *)raster_pts.data_ptr<float>(),
            chunk_cnts.data_ptr<int32_t>(), nullptr, nullptr);

        torch::Tensor cumsum = torch::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
        n_elems = cumsum[-1].item<int64_t>();
        chunk_starts = cumsum - chunk_cnts;
    } else {
        n_elems = 0;
    }

    // Second pass: allocate memory and write out the gaussian and pixel ids.
    torch::Tensor gaussian_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    torch::Tensor pixel_ids =
        torch::empty({n_elems}, means2d.options().dtype(torch::kInt64));
    if (n_elems) {
        rasterize_to_indices_in_range_lidar_kernel<<<blocks, threads, shared_mem, stream>>>(
            range_start, range_end, C, N, n_isects, (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(), opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            transmittances.data_ptr<float>(), chunk_starts.data_ptr<int32_t>(),
            (float4 *)raster_pts.data_ptr<float>(),
            nullptr, gaussian_ids.data_ptr<int64_t>(), pixel_ids.data_ptr<int64_t>());
    }
    return std::make_tuple(gaussian_ids, pixel_ids);
}

template <uint32_t COLOR_DIM>
__global__ void rasterize_to_pixels_fwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    const float2 *__restrict__ means2d,             // [C, N, 2] or [nnz, 2]
    const float3 *__restrict__ conics,              // [C, N, 3] or [nnz, 3]
    const float *__restrict__ colors,               // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const float *__restrict__ opacities,            // [C, N] or [nnz]
    const float2 *__restrict__ pix_vels,            // [C, N, 2] or [nnz, 2]
    const float *__restrict__ rolling_shutter_time, // [C]
    const float *__restrict__ backgrounds,          // [C, COLOR_DIM]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    float *__restrict__ render_colors,        // [C, image_height, image_width, COLOR_DIM]
    float *__restrict__ render_alphas,        // [C, image_height, image_width, 1]
    float *__restrict__ render_medians,       // [C, image_height, image_width, COLOR_DIM]
    int32_t *__restrict__ last_ids            // [C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    const int32_t camera_offset = camera_id * image_height * image_width;
    render_colors += camera_offset * COLOR_DIM;
    render_alphas += camera_offset;
    render_medians += camera_offset * COLOR_DIM;
    last_ids += camera_offset;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;
    float roll_time = rolling_shutter_time[camera_id] * ((py - 0.5f) / (image_height - 1) - 0.5f);

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                              // [block_size]
    float3 *xy_opacity_batch = (float3 *)&id_batch[block_size];    // [block_size]
    float3 *conic_batch = (float3 *)&xy_opacity_batch[block_size]; // [block_size]
    float2 *pix_vel_batch = (float2 *)&conic_batch[block_size];    // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x slower
    // so we stick with float for now.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[COLOR_DIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            const float2 pix_vel = pix_vels[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            pix_vel_batch[tr] = {pix_vel.x, pix_vel.y};
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float2 pix_vel = pix_vel_batch[t];
            const float2 delta = {
                xy_opac.x + roll_time * pix_vel.x - px,
                xy_opac.y + roll_time * pix_vel.y - py
            };
            const float sigma =
                0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
            
            float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * COLOR_DIM;
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }

            if (T > 0.5 && next_T <= 0.5) {
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    render_medians[pix_id * COLOR_DIM + k] = c_ptr[k];
                }
            }

            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward pass and
        // it can be very small and causing large diff in gradients with float32.
        // However, double precision makes the backward pass 1.5x slower so we stick
        // with float for now.
        render_alphas[pix_id] = 1.0f - T;
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_to_pixels_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const torch::Tensor &pix_vels,                  // [C, N, 2]
    const torch::Tensor &rolling_shutter_time,      // [C]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(pix_vels);
    CHECK_INPUT(rolling_shutter_time);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor renders = torch::empty({C, image_height, image_width, channels},
                                         means2d.options().dtype(torch::kFloat32));
    torch::Tensor alphas = torch::empty({C, image_height, image_width, 1},
                                        means2d.options().dtype(torch::kFloat32));
    torch::Tensor render_medians = torch::empty({C, image_height, image_width, channels},
                                         means2d.options().dtype(torch::kFloat32));
    torch::Tensor last_ids = torch::empty({C, image_height, image_width},
                                          means2d.options().dtype(torch::kInt32));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(float3) + sizeof(float3) + sizeof(float2));

    // TODO: an optimization can be done by passing the actual number of channels into
    // the kernel functions and avoid necessary global memory writes. This requires
    // moving the channel padding from python to C side.
    switch (channels) {
    case 1:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<1>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<1><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 2:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<2>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<2><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 3:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<3>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<3><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 4:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<4>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<4><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 5:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<5>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<5><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 8:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<8>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<8><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 9:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<9>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<9><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 16:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<16>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<16><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 17:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<17>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<17><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 32:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<32>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<32><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 33:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<33>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<33><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 64:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<64>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<64><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 65:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<65>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<65><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 128:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<128>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<128><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 129:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<129>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<129><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 256:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<256>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<256><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 257:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<257>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<257><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 512:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<512>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<512><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    case 513:
        if (cudaFuncSetAttribute(rasterize_to_pixels_fwd_kernel<513>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_kernel<513><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float2 *)pix_vels.data_ptr<float>(),
            rolling_shutter_time.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_medians.data_ptr<float>(),
            last_ids.data_ptr<int32_t>());
        break;
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
    return std::make_tuple(renders, alphas, render_medians, last_ids);
}

template <uint32_t COLOR_DIM>
__global__ void rasterize_to_points_fwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    const float2 *__restrict__ means2d,             // [C, N, 2] or [nnz, 2]
    const float3 *__restrict__ conics,              // [C, N, 3] or [nnz, 3]
    const float *__restrict__ colors,               // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const float *__restrict__ opacities,            // [C, N] or [nnz]
    const float3 *__restrict__ pix_vels,            // [C, N, 3] or [nnz, 3]
    const float2 *__restrict__ depth_compensations, // [C, N, 2] or [nnz, 2]
    const float *__restrict__ backgrounds,          // [C, COLOR_DIM]
    const float4 *__restrict__ raster_pts,          // [C, image_height, image_width, 4]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_grid_width,
    const uint32_t tile_grid_height,
    const bool compute_alpha_sum_until_points,
    const float compute_alpha_sum_until_points_threshold,
    const int32_t *__restrict__ tile_offsets,      // [C, tile_grid_height, tile_grid_width]
    const int32_t *__restrict__ flatten_ids,       // [n_isects]
    const uint32_t depth_channel_idx,
    float *__restrict__ render_colors,             // [C, image_height, image_width, COLOR_DIM]
    float *__restrict__ render_alphas,             // [C, image_height, image_width, 1]
    float *__restrict__ alpha_sum_until_points,    // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids,                 // [C, image_height, image_width]
    float *__restrict__ median_depths    // [C, image_height, image_width, 1]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * tile_grid_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_height + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_width + block.thread_index().x;

    tile_offsets += camera_id * tile_grid_height * tile_grid_width;
    const int32_t camera_offset = camera_id * image_height * image_width;
    render_colors += camera_offset * COLOR_DIM;
    render_alphas += camera_offset;
    median_depths += camera_offset;
    if (compute_alpha_sum_until_points && (alpha_sum_until_points != nullptr)) {
        alpha_sum_until_points += camera_offset;
    }
    last_ids += camera_offset;
    raster_pts += camera_offset + i * image_width + j;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }

    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    // read out point coordinates at raster_pts[camera_id, i, j]
    float px; // azimuth
    float py; // elevation
    float pz; // range
    float roll_time; // roll time
    if (inside) {
        px = raster_pts[0].x;
        py = raster_pts[0].y;
        pz = raster_pts[0].z;
        roll_time = raster_pts[0].w;
        if (pz <= 0.f) {
            inside = false;
        }
    }
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_grid_width * tile_grid_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                              // [block_size]
    float3 *xy_opacity_batch = (float3 *)&id_batch[block_size];    // [block_size]
    float3 *conic_batch = (float3 *)&xy_opacity_batch[block_size]; // [block_size]
    float3 *pix_vel_batch = (float3 *)&conic_batch[block_size];    // [block_size]
    float2 *depth_comp_batch = (float2 *)&pix_vel_batch[block_size]; // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x slower
    // so we stick with float for now.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[COLOR_DIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            const float3 pix_vel = pix_vels[g];
            const float2 depth_comp = depth_compensations[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            pix_vel_batch[tr] = {pix_vel.x, pix_vel.y, pix_vel.z};
            depth_comp_batch[tr] = {depth_comp.x, depth_comp.y};
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float3 pix_vel = pix_vel_batch[t];
            const float2 delta = {
                angle_difference(xy_opac.x + roll_time * pix_vel.x, px),
                angle_difference(xy_opac.y + roll_time * pix_vel.y, py)
            };
            const float sigma =
                0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
            float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * COLOR_DIM;
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            const float2 depth_comp = depth_comp_batch[t];
            pix_out[depth_channel_idx] += (depth_comp.x * delta.x + depth_comp.y * delta.y + pix_vel.z * roll_time) * vis;
            if (T > 0.5 && next_T <= 0.5) {
                // median depth of the gaussian
                median_depths[pix_id] = c_ptr[depth_channel_idx] + (depth_comp.x * delta.x + depth_comp.y * delta.y + pix_vel.z * roll_time);
            }
            // depth of the gaussian is assumed to be the last channel
            if (compute_alpha_sum_until_points) {
                const float g_depth = c_ptr[depth_channel_idx];
                // update alpha until we get into proximity of the point
                if (g_depth < (pz - compute_alpha_sum_until_points_threshold)) {
                    alpha_sum_until_points[pix_id] += alpha;
                }
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward pass and
        // it can be very small and causing large diff in gradients with float32.
        // However, double precision makes the backward pass 1.5x slower so we stick
        // with float for now.
        render_alphas[pix_id] = 1.0f - T;
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_to_points_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const torch::Tensor &pix_vels,                  // [C, N, 3]
    const torch::Tensor &depth_compensations,       // [C, N, 2]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    // Points to rasterize
    const torch::Tensor &raster_pts, // [C, image_height, image_width, 4]
    // image size
    const uint32_t image_width,
    const uint32_t image_height, 
    const uint32_t tile_width,
    const uint32_t tile_height,
    // compute alphas until point
    const bool compute_alpha_sum_until_points,
    const float compute_alpha_sum_until_points_threshold,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_grid_height, tile_grid_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    // depth channel index
    const uint32_t depth_channel_idx
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(pix_vels);
    CHECK_INPUT(depth_compensations);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(raster_pts);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = colors.size(-1);
    uint32_t tile_grid_height = tile_offsets.size(1);
    uint32_t tile_grid_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_grid_height * tile_grid_width blocks.
    dim3 threads = {tile_width, tile_height, 1};
    dim3 blocks = {C, tile_grid_height, tile_grid_width};

    torch::Tensor renders = torch::zeros({C, image_height, image_width, channels},
                                         means2d.options().dtype(torch::kFloat32));
    torch::Tensor alphas = torch::zeros({C, image_height, image_width, 1},
                                        means2d.options().dtype(torch::kFloat32));
    torch::Tensor median_depths = torch::zeros({C, image_height, image_width, 1},
                                        means2d.options().dtype(torch::kFloat32));
    torch::Tensor alpha_sum_until_points;
    if (compute_alpha_sum_until_points) {
        alpha_sum_until_points = torch::zeros({C, image_height, image_width, 1},
                                        means2d.options().dtype(torch::kFloat32));
    }
    torch::Tensor last_ids = torch::zeros({C, image_height, image_width},
                                          means2d.options().dtype(torch::kInt32));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_width * tile_height * (sizeof(int32_t) + sizeof(float3) + sizeof(float3) + sizeof(float3) + sizeof(float2));

    // TODO: an optimization can be done by passing the actual number of channels into
    // the kernel functions and avoid necessary global memory writes. This requires
    // moving the channel padding from python to C side.
    switch (channels) {
    case 1:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<1>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<1><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 2:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<2>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<2><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 3:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<3>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<3><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 4:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<4>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<4><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 5:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<5>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<5><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 8:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<8>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<8><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 9:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<9>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<9><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 16:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<16>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<16><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 17:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<17>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<17><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 32:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<32>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<32><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 33:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<33>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<33><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 64:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<64>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<64><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 65:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<65>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<65><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 128:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<128>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<128><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 129:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<129>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<129><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 256:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<256>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<256><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 257:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<257>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<257><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 512:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<512>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<512><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    case 513:
        if (cudaFuncSetAttribute(rasterize_to_points_fwd_kernel<513>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem,
                     " bytes), try lowering tile_size.");
        }
        rasterize_to_points_fwd_kernel<513><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects, packed,
            (float2 *)means2d.data_ptr<float>(),
            (float3 *)conics.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            (float3 *)pix_vels.data_ptr<float>(),
            (float2 *)depth_compensations.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            (float4 *)raster_pts.data_ptr<float>(),
            image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depth_channel_idx,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            compute_alpha_sum_until_points ? alpha_sum_until_points.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(), median_depths.data_ptr<float>());
        break;
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
    return std::make_tuple(renders, alphas, last_ids, alpha_sum_until_points, median_depths);
}

template <uint32_t COLOR_DIM>
__global__ void rasterize_to_pixels_bwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    // fwd inputs
    const float2 *__restrict__ means2d,             // [C, N, 2] or [nnz, 2]
    const float3 *__restrict__ conics,              // [C, N, 3] or [nnz, 3]
    const float *__restrict__ colors,               // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const float *__restrict__ opacities,            // [C, N] or [nnz]
    const float2 *__restrict__ pix_vels,            // [C, N, 2]
    const float *__restrict__ rolling_shutter_time, // [C]
    const float *__restrict__ backgrounds,          // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const float *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,    // [C, image_height, image_width]
    // grad outputs
    const float
        *__restrict__ v_render_colors, // [C, image_height, image_width, COLOR_DIM]
    const float *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    float2 *__restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
    float2 *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    float3 *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    float *__restrict__ v_colors,       // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    float *__restrict__ v_opacities,    // [C, N] or [nnz]
    float2 *__restrict__ v_pix_vels    // [C, N, 2]
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);
    float roll_time = rolling_shutter_time[camera_id] * ((py - 0.5f) / (image_height - 1) - 0.5f);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                              // [block_size]
    float3 *xy_opacity_batch = (float3 *)&id_batch[block_size];    // [block_size]
    float3 *conic_batch = (float3 *)&xy_opacity_batch[block_size]; // [block_size]
    float2 *pix_vel_batch = (float2 *)&conic_batch[block_size];    // [block_size]
    float *rgbs_batch = (float *)&pix_vel_batch[block_size]; // [block_size * COLOR_DIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[COLOR_DIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[COLOR_DIM];
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            const float2 pix_vel = pix_vels[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            pix_vel_batch[tr] = {pix_vel.x, pix_vel.y};
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 pix_vel;
            float2 delta;
            float3 conic;
            float vis;

            if (valid) {
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                pix_vel = pix_vel_batch[t];
                delta = {
                    xy_opac.x + roll_time * pix_vel.x - px,
                    xy_opac.y + roll_time * pix_vel.y - py
                };
                float sigma =
                    0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[COLOR_DIM] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float2 v_xy_abs_local = {0.f, 0.f};
            float2 v_pix_vel_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_alpha += (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                     v_sigma * delta.x * delta.y,
                                     0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                  v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    v_pix_vel_local.x = v_xy_local.x * roll_time;
                    v_pix_vel_local.y = v_xy_local.y * roll_time;
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }
            }
            warpSum<COLOR_DIM, float>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_pix_vel_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + COLOR_DIM * g;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                float* v_pix_vel_ptr = (float*)(v_pix_vels) + 2 * g;
                gpuAtomicAdd(v_pix_vel_ptr, v_pix_vel_local.x);
                gpuAtomicAdd(v_pix_vel_ptr + 1, v_pix_vel_local.y);

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const torch::Tensor &pix_vels,                  // [C, N, 2] or [nnz, 2]
    const torch::Tensor &rolling_shutter_time,      // [C]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
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
    bool absgrad) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(pix_vels);
    CHECK_INPUT(rolling_shutter_time);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_pix_vels = torch::zeros_like(pix_vels);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    if (n_isects) {
        const uint32_t shared_mem = tile_size * tile_size *
                                    (sizeof(int32_t) + sizeof(float3) + sizeof(float3) +
                                     sizeof(float2) + sizeof(float) * COLOR_DIM);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        switch (COLOR_DIM) {
        case 1:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<1>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<1><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 2:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<2>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<2><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 3:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<3>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<3><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 4:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<4>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<4><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 5:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<5>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<5><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 8:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<8>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<8><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 9:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<9>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<9><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 16:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<16>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<16><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 17:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<17>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<17><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 32:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<32>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<32><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 33:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<33>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<33><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 64:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<64>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<64><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 65:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<65>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<65><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float2 *)pix_vels.data_ptr<float>(),
                rolling_shutter_time.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                image_width, image_height, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 128:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<128>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<128>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float2 *)pix_vels.data_ptr<float>(),
                    rolling_shutter_time.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    image_width, image_height, tile_size, tile_width, tile_height,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 129:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<129>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<129>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float2 *)pix_vels.data_ptr<float>(),
                    rolling_shutter_time.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    image_width, image_height, tile_size, tile_width, tile_height,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 256:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<256>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<256>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float2 *)pix_vels.data_ptr<float>(),
                    rolling_shutter_time.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    image_width, image_height, tile_size, tile_width, tile_height,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 257:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<257>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<257>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float2 *)pix_vels.data_ptr<float>(),
                    rolling_shutter_time.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    image_width, image_height, tile_size, tile_width, tile_height,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 512:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<512>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<512>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float2 *)pix_vels.data_ptr<float>(),
                    rolling_shutter_time.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    image_width, image_height, tile_size, tile_width, tile_height,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        case 513:
            if (cudaFuncSetAttribute(rasterize_to_pixels_bwd_kernel<513>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_kernel<513>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float2 *)pix_vels.data_ptr<float>(),
                    rolling_shutter_time.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    image_width, image_height, tile_size, tile_width, tile_height,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    render_alphas.data_ptr<float>(), last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(), (float2 *)v_pix_vels.data_ptr<float>());
            break;
        default:
            AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
        }
    }

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities, v_pix_vels);
}

template <uint32_t COLOR_DIM>
__global__ void rasterize_to_points_bwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    // fwd inputs
    const float2 *__restrict__ means2d,    // [C, N, 2] or [nnz, 2]
    const float3 *__restrict__ conics,     // [C, N, 3] or [nnz, 3]
    const float *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const float *__restrict__ opacities,   // [C, N] or [nnz]
    const float3 *__restrict__ pix_vels,   // [C, N, 3]
    const float2 *__restrict__ depth_compensations, // [C, N, 2]
    const float *__restrict__ backgrounds, // [C, COLOR_DIM] or [nnz, COLOR_DIM]
    const float4 *__restrict__ raster_pts, // [C, image_height, image_width, 4]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_grid_width,
    const uint32_t tile_grid_height,
    const bool compute_alpha_sum_until_points,
    const float compute_alpha_sum_until_points_threshold,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const uint32_t depth_channel_idx,
    // fwd outputs
    const float *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,    // [C, image_height, image_width]
    // grad outputs
    const float *__restrict__ v_render_colors, // [C, image_height, image_width, COLOR_DIM]
    const float *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    const float *__restrict__ v_alpha_sum_until_points, // [C, image_height, image_width, 1]
    // grad inputs
    float2 *__restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
    float2 *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    float3 *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    float *__restrict__ v_colors,       // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    float *__restrict__ v_opacities,    // [C, N] or [nnz]
    float3 *__restrict__ v_pix_vels,    // [C, N, 3]
    float2 *__restrict__ v_depth_compensations // [C, N, 2]
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * tile_grid_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_height + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_width + block.thread_index().x;

    tile_offsets += camera_id * tile_grid_height * tile_grid_width;
    const int32_t camera_offset = camera_id * image_height * image_width;
    render_alphas += camera_offset;
    last_ids += camera_offset;
    raster_pts += camera_offset + i * image_width + j;
    v_render_colors += camera_offset * COLOR_DIM;
    v_render_alphas += camera_offset;
    if (v_alpha_sum_until_points != nullptr) {
        v_alpha_sum_until_points += camera_offset;
    }
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }

    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    float px;
    float py;
    float pz;
    float roll_time;
    if (inside) {
        px = raster_pts[0].x; // azimuth
        py = raster_pts[0].y; // elevation
        pz = raster_pts[0].z; // range
        roll_time = raster_pts[0].w;
        if (pz <= 0.f) {
            inside = false;
        }
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_grid_width * tile_grid_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;                              // [block_size]
    float3 *xy_opacity_batch = (float3 *)&id_batch[block_size];    // [block_size]
    float3 *conic_batch = (float3 *)&xy_opacity_batch[block_size]; // [block_size]
    float3 *pix_vel_batch = (float3 *)&conic_batch[block_size];    // [block_size]
    float2 *depth_comp_batch = (float2 *)&pix_vel_batch[block_size]; // [block_size]
    float *rgbs_batch = (float *)&depth_comp_batch[block_size]; // [block_size * COLOR_DIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[COLOR_DIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[COLOR_DIM];
    PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];
    float v_alpha_sum_until_points_current;
    if (v_alpha_sum_until_points != nullptr) {
        v_alpha_sum_until_points_current = v_alpha_sum_until_points[pix_id];
    }

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const float2 xy = means2d[g];
            const float opac = opacities[g];
            const float3 pix_vel = pix_vels[g];
            const float2 depth_comp = depth_compensations[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            pix_vel_batch[tr] = {pix_vel.x, pix_vel.y, pix_vel.z};
            depth_comp_batch[tr] = {depth_comp.x, depth_comp.y};
            PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float3 pix_vel;
            float2 depth_comp;
            float2 delta;
            float3 conic;
            float vis;

            if (valid) {
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                pix_vel = pix_vel_batch[t];
                depth_comp = depth_comp_batch[t];
                delta = {
                    angle_difference(xy_opac.x + roll_time * pix_vel.x, px),
                    angle_difference(xy_opac.y + roll_time * pix_vel.y, py)
                };
                float sigma =
                    0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                    conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[COLOR_DIM] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float2 v_xy_abs_local = {0.f, 0.f};
            float3 v_pix_vel_local = {0.f, 0.f, 0.f};
            float2 v_depth_comp_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
                // pragma unroll?
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_alpha += (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }
                v_alpha += (depth_comp.x * delta.x + depth_comp.y * delta.y + pix_vel.z * roll_time) * T * v_render_c[depth_channel_idx];

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    v_depth_comp_local = {v_rgb_local[depth_channel_idx] * delta.x, v_rgb_local[depth_channel_idx] * delta.y};
                    v_pix_vel_local.z = v_rgb_local[depth_channel_idx] * roll_time;
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                     v_sigma * delta.x * delta.y,
                                     0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                  v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    v_xy_local.x += v_rgb_local[depth_channel_idx] * depth_comp.x;
                    v_xy_local.y += v_rgb_local[depth_channel_idx] * depth_comp.y;
                    v_pix_vel_local.x = v_xy_local.x * roll_time;
                    v_pix_vel_local.y = v_xy_local.y * roll_time;
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }
                buffer[depth_channel_idx] += (depth_comp.x * delta.x + depth_comp.y * delta.y + pix_vel.z * roll_time) * fac;

                const float g_depth = rgbs_batch[t * COLOR_DIM + depth_channel_idx];
                const bool g_is_prior_point = g_depth < (pz - compute_alpha_sum_until_points_threshold);
                if (compute_alpha_sum_until_points && g_is_prior_point) {
                    const float v_sigma_until_point = v_alpha_sum_until_points_current * -opac * vis;
                    const float2 v_xy_local_until_point = {
                        v_sigma_until_point * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma_until_point * (conic.y * delta.x + conic.z * delta.y)
                    };
                    
                    v_xy_local.x += v_xy_local_until_point.x;
                    v_xy_local.y += v_xy_local_until_point.y;
                    v_pix_vel_local.x += v_xy_local_until_point.x * roll_time;
                    v_pix_vel_local.y += v_xy_local_until_point.y * roll_time;
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local.x += abs(v_xy_local_until_point.x);
                        v_xy_abs_local.y += abs(v_xy_local_until_point.y);
                    }
             
                    v_conic_local.x += v_sigma_until_point * 0.5f * delta.x * delta.x;
                    v_conic_local.y += v_sigma_until_point * delta.x * delta.y;
                    v_conic_local.z += v_sigma_until_point * 0.5f * delta.y * delta.y;
                    
                    v_opacity_local += v_alpha_sum_until_points_current * vis;
                }
            }
            warpSum<COLOR_DIM, float>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_pix_vel_local, warp);
            warpSum(v_depth_comp_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + COLOR_DIM * g;
                PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                float* v_pix_vel_ptr = (float*)(v_pix_vels) + 3 * g;
                gpuAtomicAdd(v_pix_vel_ptr, v_pix_vel_local.x);
                gpuAtomicAdd(v_pix_vel_ptr + 1, v_pix_vel_local.y);
                gpuAtomicAdd(v_pix_vel_ptr + 2, v_pix_vel_local.z);

                float* v_depth_comp_ptr = (float*)(v_depth_compensations) + 2 * g;
                gpuAtomicAdd(v_depth_comp_ptr, v_depth_comp_local.x);
                gpuAtomicAdd(v_depth_comp_ptr + 1, v_depth_comp_local.y);

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_points_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
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
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(pix_vels);
    CHECK_INPUT(depth_compensations);
    CHECK_INPUT(raster_pts);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    CHECK_INPUT(v_alpha_sum_until_points);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_grid_height = tile_offsets.size(1);
    uint32_t tile_grid_width = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_width, tile_height, 1};
    dim3 blocks = {C, tile_grid_height, tile_grid_width};

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_pix_vels = torch::zeros_like(pix_vels);
    torch::Tensor v_depth_compensations = torch::zeros_like(depth_compensations);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    if (n_isects) {
        const uint32_t shared_mem = tile_width * tile_height *
                                    (sizeof(int32_t) + sizeof(float3) + sizeof(float3) +
                                     sizeof(float3) + sizeof(float2) + sizeof(float) * COLOR_DIM);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        switch (COLOR_DIM) {
        case 1:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<1>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<1><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 2:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<2>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<2><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 3:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<3>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<3><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 4:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<4>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<4><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 5:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<5>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<5><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 8:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<8>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<8><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 9:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<9>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<9><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 16:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<16>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<16><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 17:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<17>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<17><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 32:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<32>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<32><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 33:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<33>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<33><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 64:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<64>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<64><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 65:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<65>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<65><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects, packed,
                (float2 *)means2d.data_ptr<float>(),
                (float3 *)conics.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                (float3 *)pix_vels.data_ptr<float>(),
                (float2 *)depth_compensations.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                (float4 *)raster_pts.data_ptr<float>(),
                image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                depth_channel_idx,
                render_alphas.data_ptr<float>(), 
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                v_alpha_sum_until_points.data_ptr<float>(),
                absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                (float2 *)v_means2d.data_ptr<float>(),
                (float3 *)v_conics.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                (float3 *)v_pix_vels.data_ptr<float>(),
                (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 128:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<128>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<128>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float3 *)pix_vels.data_ptr<float>(),
                    (float2 *)depth_compensations.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    (float4 *)raster_pts.data_ptr<float>(),
                    image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                    compute_alpha_sum_until_points,
                    compute_alpha_sum_until_points_threshold,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    depth_channel_idx,
                    render_alphas.data_ptr<float>(), 
                    last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    v_alpha_sum_until_points.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    (float3 *)v_pix_vels.data_ptr<float>(),
                    (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 129:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<129>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<129>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float3 *)pix_vels.data_ptr<float>(),
                    (float2 *)depth_compensations.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    (float4 *)raster_pts.data_ptr<float>(),
                    image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                    compute_alpha_sum_until_points,
                    compute_alpha_sum_until_points_threshold,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    depth_channel_idx,
                    render_alphas.data_ptr<float>(), 
                    last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    v_alpha_sum_until_points.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    (float3 *)v_pix_vels.data_ptr<float>(),
                    (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 256:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<256>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<256>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float3 *)pix_vels.data_ptr<float>(),
                    (float2 *)depth_compensations.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    (float4 *)raster_pts.data_ptr<float>(),
                    image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                    compute_alpha_sum_until_points,
                    compute_alpha_sum_until_points_threshold,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    depth_channel_idx,
                    render_alphas.data_ptr<float>(), 
                    last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    v_alpha_sum_until_points.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    (float3 *)v_pix_vels.data_ptr<float>(),
                    (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 257:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<257>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<257>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float3 *)pix_vels.data_ptr<float>(),
                    (float2 *)depth_compensations.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    (float4 *)raster_pts.data_ptr<float>(),
                    image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                    compute_alpha_sum_until_points,
                    compute_alpha_sum_until_points_threshold,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    depth_channel_idx,
                    render_alphas.data_ptr<float>(), 
                    last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    v_alpha_sum_until_points.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    (float3 *)v_pix_vels.data_ptr<float>(),
                    (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 512:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<512>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<512>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float3 *)pix_vels.data_ptr<float>(),
                    (float2 *)depth_compensations.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    (float4 *)raster_pts.data_ptr<float>(),
                    image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                    compute_alpha_sum_until_points,
                    compute_alpha_sum_until_points_threshold,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    depth_channel_idx,
                    render_alphas.data_ptr<float>(), 
                    last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    v_alpha_sum_until_points.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    (float3 *)v_pix_vels.data_ptr<float>(),
                    (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        case 513:
            if (cudaFuncSetAttribute(rasterize_to_points_bwd_kernel<513>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_points_bwd_kernel<513>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    (float2 *)means2d.data_ptr<float>(),
                    (float3 *)conics.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    (float3 *)pix_vels.data_ptr<float>(),
                    (float2 *)depth_compensations.data_ptr<float>(),
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                            : nullptr,
                    (float4 *)raster_pts.data_ptr<float>(),
                    image_width, image_height, tile_width, tile_height, tile_grid_width, tile_grid_height,
                    compute_alpha_sum_until_points,
                    compute_alpha_sum_until_points_threshold,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    depth_channel_idx,
                    render_alphas.data_ptr<float>(), 
                    last_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    v_alpha_sum_until_points.data_ptr<float>(),
                    absgrad ? (float2 *)v_means2d_abs.data_ptr<float>() : nullptr,
                    (float2 *)v_means2d.data_ptr<float>(),
                    (float3 *)v_conics.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    (float3 *)v_pix_vels.data_ptr<float>(),
                    (float2 *)v_depth_compensations.data_ptr<float>());
            break;
        default:
            AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
        }
    }

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities, v_pix_vels, v_depth_compensations);
}