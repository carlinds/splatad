from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,
) -> Tensor:
    """Computes spherical harmonics.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.

    Returns:
        Spherical harmonics. [..., 3]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    assert dirs.shape[:-1] == coeffs.shape[:-2], (dirs.shape, coeffs.shape)
    assert dirs.shape[-1] == 3, dirs.shape
    assert coeffs.shape[-1] == 3, coeffs.shape
    if masks is not None:
        assert masks.shape == dirs.shape[:-1], masks.shape
        masks = masks.contiguous()
    return _SphericalHarmonics.apply(
        degrees_to_use, dirs.contiguous(), coeffs.contiguous(), masks
    )


def quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Quaternions (No need to be normalized). [N, 4]
        scales: Scales. [N, 3]
        compute_covar: Whether to compute covariance matrices. Default: True. If False,
            the returned covariance matrices will be None.
        compute_preci: Whether to compute precision matrices. Default: True. If False,
            the returned precision matrices will be None.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple:

        - **Covariance matrices**. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
        - **Precision matrices**. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
    """
    assert quats.dim() == 2 and quats.size(1) == 4, quats.size()
    assert scales.dim() == 2 and scales.size(1) == 3, scales.size()
    quats = quats.contiguous()
    scales = scales.contiguous()
    covars, precis = _QuatScaleToCovarPreci.apply(
        quats, scales, compute_covar, compute_preci, triu
    )
    return covars if compute_covar else None, precis if compute_preci else None


def persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.

    Args:
        means: Gaussian means. [C, N, 3]
        covars: Gaussian covariances. [C, N, 3, 3]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [C, N, 2]
        - **Projected covariances**. [C, N, 2, 2]
    """
    C, N, _ = means.shape
    assert means.shape == (C, N, 3), means.size()
    assert covars.shape == (C, N, 3, 3), covars.size()
    assert Ks.shape == (C, 3, 3), Ks.size()
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()
    return _PerspProj.apply(means, covars, Ks, width, height)

def compute_pix_velocity(p_view: Tensor, # [C, N, 3]
                           lin_vel: Tensor, # [C, 3]
                           ang_vel: Tensor, # [C, 3]
                           velocities: Tensor, # [C, N, 3]
                           Ks: Tensor, #[C, 3, 3]
                           width: int,
                           height: int,
) -> Tensor:
    """Compute velocity of Gaussians in pixel coordinates.
    
    Args:
        p_view: Gaussian means in camera coordinates. [C, N, 3]
        lin_vel: Linear velocity of the camera. [C, 3]
        ang_vel: Angular velocity of the camera. [C, 3]
        velocities: Gaussian velocities in world coordinates. [C, N, 3]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        
    Returns:
        Gaussian velocity in pixel coordinates. [C, N, 2]
    """
    C, N, _ = p_view.shape
    assert p_view.shape == (C, N, 3), p_view.size()
    assert lin_vel.shape == (C, 3), lin_vel.size()
    assert ang_vel.shape == (C, 3), ang_vel.size()
    assert velocities.shape == (C, N, 3), velocities.size()
    assert Ks.shape == (C, 3, 3), Ks.size()
    p_view = p_view.contiguous()
    lin_vel = lin_vel.contiguous()
    ang_vel = ang_vel.contiguous()
    velocities = velocities.contiguous()
    Ks = Ks.contiguous()
    return _ComputePixVelocity.apply(p_view, lin_vel, ang_vel, velocities, Ks, width, height)


def world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera coordinate system.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances. [N, 3, 3]
        viewmats: World-to-camera transformation matrices. [C, 4, 4]

    Returns:
        A tuple:

        - **Gaussian means in camera coordinate system**. [C, N, 3]
        - **Gaussian covariances in camera coordinate system**. [C, N, 3, 3]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert covars.size() == (N, 3, 3), covars.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _WorldToCam.apply(means, covars, viewmats)

def lidar_proj(means: Tensor, # [C, N, 3]
               covars: Tensor, # [C, N, 3, 3]
               eps2d: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Transform Gaussians from Euclidean to spherical coordinates, only azimuth and elevation.
    Note that elevation is defined as the angle from the horizontal xy-plane.
    
    Args:
        means: Gaussian means. [C, N, 3]
        covars: Gaussian covariances. [C, N, 3, 3]
        eps2d: A epsilon added to the 2D covariance.
        
    Returns:
        A tuple:
        
        - **Gaussian means in spherical coordinate system**. [C, N, 2]
        - **Gaussian covariances in spherical coordinate system**. [C, N, 2, 2]
        - **Depth compensations**. [C, N, 2]
    """
    C, N, _ = means.shape
    assert means.shape == (C, N, 3), means.size()
    assert covars.shape == (C, N, 3, 3), covars.size()
    means = means.contiguous()
    covars = covars.contiguous()
    return _LidarProj.apply(means, covars, eps2d)

def compute_lidar_velocity(p_view: Tensor, # [C, N, 3]
                           lin_vel: Tensor, # [C, 3]
                           ang_vel: Tensor, # [C, 3]
                           velocities: Tensor # [C, N, 3]
) -> Tensor:
    """Compute velocity of Gaussians in spherical coordinates. Note that elevation is defined as the angle
    from the horizontal xy-plane.
    
    Args:
        p_view: Gaussian means in lidar coordinates. [C, N, 3]
        lin_vel: Linear velocity of the lidar. [C, 3]
        ang_vel: Angular velocity of the lidar. [C, 3]
        velocities: Gaussian velocities in world coordinates. [C, N, 3]
        
    Returns:
        Gaussian velocity in spherical coordinates. [C, N, 3]
    """
    C, N, _ = p_view.shape
    assert p_view.shape == (C, N, 3), p_view.size()
    assert lin_vel.shape == (C, 3), lin_vel.size()
    assert ang_vel.shape == (C, 3), ang_vel.size()
    assert velocities.shape == (C, N, 3), velocities.size()
    p_view = p_view.contiguous()
    lin_vel = lin_vel.contiguous()
    ang_vel = ang_vel.contiguous()
    velocities = velocities.contiguous()
    return _ComputeLidarVelocity.apply(p_view, lin_vel, ang_vel, velocities)


def fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    velocities: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    linear_velocity: Tensor, # [C, 3]
    angular_velocity: Tensor, # [C, 3]
    rolling_shutter_time: Tensor, # [C]
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and perspective projection (:func:`persp_proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `camera_ids` tensor and `gaussian_ids` tensor will be returned to indicate the
        row (camera) and column (Gaussian) indices of the packed flattened tensor, which is essentially
        following the COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6] Optional.
        quats: Quaternions (No need to be normalized). [N, 4] Optional.
        scales: Scales. [N, 3] Optional.
        velocities: Gaussian velocities. [N, 3] Optional.
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        linear_velocity: Linear velocity of the camera. [C, 3]
        angular_velocity: Angular velocity of the camera. [C, 3]
        rolling_shutter_time: Rolling shutter time of the camera. [C]
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]
        - **pix_vels**. The velocities of Gaussians in 2D coordiantes. [nnz, 2]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N, 2].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
        - **pix_vels**. The velocities of Gaussians in 2D coordiantes. [C, N, 2]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    assert linear_velocity.size() == (C, 3), linear_velocity.size()
    assert angular_velocity.size() == (C, 3), angular_velocity.size()
    assert rolling_shutter_time.size() == (C,), rolling_shutter_time.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if velocities is not None:
        assert velocities.size() == (N, 3), velocities.size()
        velocities = velocities.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _FullyFusedProjectionPacked.apply(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
            calc_compensations,
        )
    else:
        return _FullyFusedProjection.apply(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )


def fully_fused_lidar_projection(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    velocities: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    linear_velocity: Tensor, # [C, 3]
    angular_velocity: Tensor, # [C, 3]
    rolling_shutter_time: Tensor, # [C]
    min_elevation: float = -45,
    max_elevation: float = 45,
    min_azimuth: float = -180,
    max_azimuth: float = 180,
    eps2d: float = 0.01,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to lidar space (:func:`world_to_cam()`),
    and spherical projection (:func:`lidar_proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the lidar frustum.
        This frustum is defined by the elevation angle (angle from horizontal xy-plane) and azimuth
        angle (angle around the z-axis, counter-clockwise from the x-axis).
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `camera_ids` tensor and `gaussian_ids` tensor will be returned to indicate the
        row (lidar) and column (Gaussian) indices of the packed flattened tensor, which is essentially
        following the COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6] Optional.
        quats: Quaternions (No need to be normalized). [N, 4] Optional.
        scales: Scales. [N, 3] Optional.
        viewmats: Lidar-to-world matrices. [C, 4, 4]
        velocities: Gaussian velocities. [N, 3] Optional.
        linear_velocity: Linear velocity of the Lidar. [C, 3]
        angular_velocity: Angular velocity of the Lidar. [C, 3]
        rolling_shutter_time: Rolling shutter time of the Lidar. [C]
        min_elevation: Minimum elevation angle in degrees. Default: -45.
        max_elevation: Maximum elevation angle in degrees. Default: 45.
        min_azimuth: Minimum azimuth angle in degrees. Default: -180.
        max_azimuth: Maximum azimuth angle in degrees. Default: 180.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.01.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in spherical coords, degrees. Float32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The Euclidian distance of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]
        - **pix_vels**. The velocities of Gaussians in 2D coordiantes. [nnz, 2]
        - **depth_compensations**. The depth compensation factor, i.e., how the depth changes with angular coordinates due to Gaussian orientation. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in spherical coords, degrees. Float32 tensor of shape [C, N, 2].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The Euclidian distance of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
        - **pix_vels**. The velocities of Gaussians in 2D coordiantes. [C, N, 2]
        - **depth_compensations**. The depth compensation factor, i.e., how the depth changes with angular coordinates due to Gaussian orientation. [C, N]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if velocities is not None:
        assert velocities.size() == (N, 3), velocities.size()
        velocities = velocities.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    if packed:
        raise NotImplementedError("FullyFusedLidarProjectionPacked is not implemented yet")
    else:
        return _FullyFusedLidarProjection.apply(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )

@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    radii: Tensor,  # [C, N, 2] or [nnz, 2]
    depths: Tensor,  # [C, N] or [nnz]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        depths: Z-depth of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of cameras. Required if packed is True.
        camera_ids: The row indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the camera and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [C * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz, 2), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert camera_ids is not None, "camera_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        camera_ids = camera_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        C = n_cameras

    else:
        C, N, _ = means2d.shape
        assert means2d.shape == (C, N, 2), means2d.size()
        assert radii.shape == (C, N, 2), radii.size()
        assert depths.shape == (C, N), depths.size()

    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("isect_tiles")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        camera_ids,
        gaussian_ids,
        C,
        tile_size,
        tile_width,
        tile_height,
        sort,
        True,  # DoubleBuffer: memory efficient radixsort
    )
    return tiles_per_gauss, isect_ids, flatten_ids

@torch.no_grad()
def isect_lidar_tiles(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    radii: Tensor,  # [C, N, 2] or [nnz, 2]
    depths: Tensor,  # [C, N] or [nnz]
    elev_boundaries: Tensor,
    tile_azim_resolution: float,
    min_azim: float,
    sort: bool = True,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        depths: Euclidian distance of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        elev_boundaries: Elevation boundaries, defining borders between lidar channels. [n_elev]
        tile_azim_resolution: Tile azimuth resolution.
        min_azim: Minimum azimuth angle in degrees.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of lidars. Required if packed is True.
        camera_ids: The row indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the camera and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [C * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz, 2), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert camera_ids is not None, "camera_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        camera_ids = camera_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        C = n_cameras

    else:
        C, N, _ = means2d.shape
        assert means2d.shape == (C, N, 2), means2d.size()
        assert radii.shape == (C, N, 2), radii.size()
        assert depths.shape == (C, N), depths.size()

    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("isect_lidar_tiles")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        camera_ids,
        gaussian_ids,
        C,
        elev_boundaries.contiguous(),
        tile_azim_resolution,
        min_azim,
        sort,
        True,  # DoubleBuffer: memory efficient radixsort
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_cameras: Number of cameras.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [C, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("isect_offset_encode")(
        isect_ids.contiguous(), n_cameras, tile_width, tile_height
    )

@torch.no_grad()
def map_points_to_lidar_tiles(
    points2d: Tensor,  # [C, N, 2] or [nnz, 2]
    elev_boundaries: Tensor,
    tile_azim_resolution: float,
    min_azim: float,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    sort: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Maps projected points to intersecting tiles, similar to isect_lidar_tiles, but assume
    infinitesimal radius.
    This is used for the purpose of mapping points to tiles, e.g., for rendering.

    Args:
        points2d: Projected points. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        elev_boundaries: Elevation boundaries, defining borders between lidar channels. [n_elev]
        tile_azim_resolution: Tile azimuth resolution.
        min_azim: Minimum azimuth angle in degrees.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of cameras. Required if packed is True.
        camera_ids: The row indices of the projected points. Required if packed is True.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
    Returns:
        A tuple:

        - **Tiles per point**. The number of tiles intersected by each point.
          Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the camera and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [C * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = points2d.size(0)
        assert points2d.shape == (nnz, 2), points2d.size()
        assert camera_ids is not None, "camera_ids is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        camera_ids = camera_ids.contiguous()
        C = n_cameras

    else:
        C, N, _ = points2d.shape
        assert points2d.shape == (C, N, 2), points2d.size()
        
    tile_ids = _make_lazy_cuda_func("map_points_to_lidar_tiles")(
        points2d.contiguous(),
        camera_ids,
        C,
        elev_boundaries.contiguous(),
        tile_azim_resolution,
        min_azim,
        sort,
        True  # DoubleBuffer: memory efficient radixsort
    )
    return tile_ids

@torch.no_grad()
def points_mapping_offset_encode(
    point_ids: Tensor, n_lidars: int, tile_width: int, tile_height: int
) -> Tensor:
    """Encodes intersection ids to offsets, similar to isec_offset_encode, but for points.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_lidars: Number of lidars.
        tile_width: Number of tiles in the width direction.
        tile_height: Number of tiles in the height direction.

    Returns:
        Offsets. [C, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("points_mapping_offset_encode")(
        point_ids.contiguous(), n_lidars, tile_width, tile_height
    )

@torch.no_grad()
def populate_image_from_points(
    points: Tensor,
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    tile_offsets: Tensor, 
    flatten_id: Tensor,
) -> Tensor:
    """
    Populate an image from points. Sets pixel values to point properties. Useful to generate
    ground truth image from points.
    Args:
        points: Points to populate the image with. [C, N, 5] or [nnz, 5]
        image_width: Image width.
        image_height: Image height.
        tile_width: Tile width.
        tile_height: Tile height.
        tile_offsets: Offsets for the tiles. [C, tile_height, tile_width]
        flatten_id: Flattened ids for the points. [n_isects]

    Returns:
        Populated image. [C, image_height, image_width, 5]
    """
    return _make_lazy_cuda_func("populate_image_from_points")(
        points.contiguous(),
        image_width,
        image_height,
        tile_width,
        tile_height,
        tile_offsets.contiguous(),
        flatten_id.contiguous()
    )

def rasterize_to_pixels(
    means2d: Tensor,                               # [C, N, 2] or [nnz, 2]
    conics: Tensor,                                # [C, N, 3] or [nnz, 3]
    colors: Tensor,                                # [C, N, channels] or [nnz, channels]
    opacities: Tensor,                             # [C, N] or [nnz]
    pix_vels: Tensor,                              # [C, N, 2] or [nnz, 2]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,                         # [C, tile_height, tile_width]
    flatten_ids: Tensor,                           # [n_isects]
    rolling_shutter_time: Optional[Tensor] = None, # [C]
    backgrounds: Optional[Tensor] = None,          # [C, channels]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        pix_vels: Pixel velocities. [C, N, 2]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        rolling_shutter_time: Rolling shutter time. [C]. Default: None.
        backgrounds: Background colors. [C, channels]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
    """

    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
        assert pix_vels.shape == (nnz, 2), pix_vels.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
        assert pix_vels.shape == (C, N, 2), pix_vels.shape
    if rolling_shutter_time is not None:
        assert rolling_shutter_time.shape == (C,), rolling_shutter_time.shape
    else:
        rolling_shutter_time = torch.zeros(C, device=device)
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    render_colors, render_alphas, render_medians = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        pix_vels.contiguous(),
        rolling_shutter_time.contiguous(),
        backgrounds,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
        render_medians = render_medians[..., :-padded_channels]
    return render_colors, render_alphas, render_medians

def rasterize_to_points(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    lidar_features: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    pix_vels: Tensor,  # [C, N, 3] or [nnz, 3]
    depth_compensations: Tensor,  # [C, N, 2] or [nnz, 2]
    raster_pts: Tensor, # [C, H, W, 3]
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    compute_alpha_sum_until_points: bool,
    compute_alpha_sum_until_points_threshold: float,
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
    """Rasterizes Gaussians to points.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        lidar_features: Gaussian ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        pix_vels: Spherical velocities. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        depth_compensations: Depth compensation factors. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        raster_pts: Spherical coordinates of the points to rasterize. [C, H, W, 3]
        image_width: Image width.
        image_height: Image height.
        tile_width: Tile width.
        tile_height: Tile height.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        compute_alpha_sum_until_points: Whether to compute the alpha sum until provided observed points.
        compute_alpha_sum_until_points_threshold: Alpha sum is computed up until we are threshold away from the point.
        backgrounds: Background colors. [C, channels]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
        - **alpha_sum_until_points**. [C, image_height, image_width, 1].
        - **median_depths**. [C, image_height, image_width, 1].
    """

    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert lidar_features.shape[0] == nnz, lidar_features.shape
        assert opacities.shape == (nnz,), opacities.shape
        assert pix_vels.shape == (nnz, 3), pix_vels.shape
        assert depth_compensations.shape == (nnz, 2), depth_compensations.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert lidar_features.shape[:2] == (C, N), lidar_features.shape
        assert opacities.shape == (C, N), opacities.shape
        assert pix_vels.shape == (C, N, 3), pix_vels.shape
        assert depth_compensations.shape == (C, N, 2), depth_compensations.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, lidar_features.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = lidar_features.shape[-1]
    depth_channel_idx = channels - 1
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of lidar_features channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        lidar_features = torch.cat(
            [
                lidar_features,
                torch.zeros(*lidar_features.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_grid_height, tile_grid_width = isect_offsets.shape[1:3]
    assert (
        tile_grid_height * tile_height >= image_height
    ), f"Assert Failed: {tile_grid_height} * {tile_height} >= {image_height}"
    assert (
        tile_grid_width * tile_width >= image_width
    ), f"Assert Failed: {tile_grid_width} * {tile_width} >= {image_width}"
    assert raster_pts.shape == (
        C,
        image_height,
        image_width,
        4,
    ), "raster_pts does not have the correct shape"

    render_lidar_features, render_alphas, alpha_sum_until_points, median_depths = (
        _RasterizeToPoints.apply(
            means2d.contiguous(),
            conics.contiguous(),
            lidar_features.contiguous(),
            opacities.contiguous(),
            pix_vels.contiguous(),
            depth_compensations.contiguous(),
            backgrounds,
            raster_pts.contiguous(),
            image_width,
            image_height,
            tile_width,
            tile_height,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            absgrad,
            depth_channel_idx,
        )
    )

    if padded_channels > 0:
        render_lidar_features = render_lidar_features[..., :-padded_channels]
    return render_lidar_features, render_alphas, alpha_sum_until_points, median_depths


@torch.no_grad()
def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: Tensor,                         # [C, image_height, image_width]
    means2d: Tensor,                                # [C, N, 2]
    conics: Tensor,                                 # [C, N, 3]
    opacities: Tensor,                              # [C, N]
    pix_vels: Tensor,                               # [C, N, 2]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,                          # [C, tile_height, tile_width]
    flatten_ids: Tensor,                            # [n_isects]
    rolling_shutter_time: Optional[Tensor] = None,  # [C]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [C, image_height, image_width]
        means2d: Projected Gaussian means. [C, N, 2]
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3]
        opacities: Gaussian opacities that support per-view values. [C, N]
        pix_vels: Pixel velocities. [C, N, 2]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Camera ids**. Camera indices. A flattened list of shape [M].
    """

    C, N, _ = means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert opacities.shape == (C, N), opacities.shape
    assert isect_offsets.shape[0] == C, isect_offsets.shape
    assert pix_vels.shape == (C, N, 2), pix_vels.shape
    if rolling_shutter_time is not None:
        assert rolling_shutter_time.shape == (C,), rolling_shutter_time.shape
    else:
        rolling_shutter_time = torch.zeros(C, device=means2d.device)

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func("rasterize_to_indices_in_range")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        pix_vels.contiguous(),
        rolling_shutter_time.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_camera_ids = out_indices // (image_width * image_height)
    return out_gauss_ids, out_pixel_ids, out_camera_ids

@torch.no_grad()
def rasterize_to_indices_in_range_lidar(
    range_start: int,
    range_end: int,
    transmittances: Tensor,                         # [C, image_height, image_width]
    means2d: Tensor,                                # [C, N, 2]
    conics: Tensor,                                 # [C, N, 3]
    opacities: Tensor,                              # [C, N]
    pix_vels: Tensor,                               # [C, N, 3]
    raster_pts: Tensor,                             # [C, image_height, image_width, 4]
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    isect_offsets: Tensor,                          # [C, tile_height, tile_width]
    flatten_ids: Tensor,                            # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to lidar images and returns the indices of intersections.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [C, image_height, image_width]
        means2d: Projected Gaussian means. [C, N, 2]
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3]
        opacities: Gaussian opacities that support per-view values. [C, N]
        pix_vels: Spherical velocities. [C, N, 3]
        raster_pts: Spherical coordinates + time for points to rasterize. [C, image_height, image_width, 4]
        image_width: Image width.
        image_height: Image height.
        tile_width: Tile width.
        tile_height: Tile height.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Camera ids**. Camera indices. A flattened list of shape [M].
    """

    C, N, _ = means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert opacities.shape == (C, N), opacities.shape
    assert isect_offsets.shape[0] == C, isect_offsets.shape
    assert pix_vels.shape == (C, N, 3), pix_vels.shape

    tile_grid_height, tile_grid_width = isect_offsets.shape[1:3]
    assert (
        tile_grid_height * tile_height >= image_height
    ), f"Assert Failed: {tile_grid_height} * {tile_height} >= {image_height}"
    assert (
        tile_grid_width * tile_width >= image_width
    ), f"Assert Failed: {tile_grid_width} * {tile_width} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func("rasterize_to_indices_in_range_lidar")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        pix_vels.contiguous(),
        raster_pts.contiguous(),
        image_width,
        image_height,
        tile_width,
        tile_height,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_camera_ids = out_indices // (image_width * image_height)
    return out_gauss_ids, out_pixel_ids, out_camera_ids


class _QuatScaleToCovarPreci(torch.autograd.Function):
    """Converts quaternions and scales to covariance and precision matrices."""

    @staticmethod
    def forward(
        ctx,
        quats: Tensor,  # [N, 4],
        scales: Tensor,  # [N, 3],
        compute_covar: bool = True,
        compute_preci: bool = True,
        triu: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        covars, precis = _make_lazy_cuda_func("quat_scale_to_covar_preci_fwd")(
            quats, scales, compute_covar, compute_preci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_precis: Tensor):
        quats, scales = ctx.saved_tensors
        compute_covar = ctx.compute_covar
        compute_preci = ctx.compute_preci
        triu = ctx.triu
        if compute_covar and v_covars.is_sparse:
            v_covars = v_covars.to_dense()
        if compute_preci and v_precis.is_sparse:
            v_precis = v_precis.to_dense()
        v_quats, v_scales = _make_lazy_cuda_func("quat_scale_to_covar_preci_bwd")(
            quats,
            scales,
            v_covars.contiguous() if compute_covar else None,
            v_precis.contiguous() if compute_preci else None,
            triu,
        )
        return v_quats, v_scales, None, None, None

class _PerspProj(torch.autograd.Function):
    """Perspective fully_fused_projection on Gaussians."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [C, N, 3]
        covars: Tensor,  # [C, N, 3, 3]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
    ) -> Tuple[Tensor, Tensor]:
        means2d, covars2d = _make_lazy_cuda_func("persp_proj_fwd")(
            means, covars, Ks, width, height
        )
        ctx.save_for_backward(means, covars, Ks)
        ctx.width = width
        ctx.height = height
        return means2d, covars2d

    @staticmethod
    def backward(ctx, v_means2d: Tensor, v_covars2d: Tensor):
        means, covars, Ks = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_means, v_covars = _make_lazy_cuda_func("persp_proj_bwd")(
            means,
            covars,
            Ks,
            width,
            height,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
        )
        return v_means, v_covars, None, None, None

class _ComputePixVelocity(torch.autograd.Function):
    """Compute velocity of Gaussians in image coordinates."""

    @staticmethod
    def forward(
        ctx,
        p_view,  # [C, N, 3]
        lin_vel, # [C, 3]
        ang_vel,  # [C, 3]
        velocities, # [C, N, 3]
        Ks, # [C, 3, 3]
        width, 
        height,
    ) -> Tensor:
        vel = _make_lazy_cuda_func("compute_pix_velocity_fwd")(
            p_view, lin_vel, ang_vel, velocities, Ks, width, height
        )
        ctx.save_for_backward(p_view, lin_vel, ang_vel, velocities, Ks)
        ctx.width = width
        ctx.height = height
        return vel
    
    @staticmethod
    def backward(ctx, v_vel: Tensor):
        p_view, lin_vel, ang_vel, velocities, Ks = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_p_view, v_velocities = _make_lazy_cuda_func("compute_pix_velocity_bwd")(
            p_view,
            lin_vel,
            ang_vel,
            velocities,
            Ks,
            width,
            height,
            v_vel.contiguous()
        )
        return v_p_view, None, None, v_velocities, None, None, None

class _WorldToCam(torch.autograd.Function):
    """Transforms Gaussians from world to camera space."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 3, 3]
        viewmats: Tensor,  # [C, 4, 4]
    ) -> Tuple[Tensor, Tensor]:
        means_c, covars_c = _make_lazy_cuda_func("world_to_cam_fwd")(
            means, covars, viewmats
        )
        ctx.save_for_backward(means, covars, viewmats)
        return means_c, covars_c

    @staticmethod
    def backward(ctx, v_means_c: Tensor, v_covars_c: Tensor):
        means, covars, viewmats = ctx.saved_tensors
        v_means, v_covars, v_viewmats = _make_lazy_cuda_func("world_to_cam_bwd")(
            means,
            covars,
            viewmats,
            v_means_c.contiguous(),
            v_covars_c.contiguous(),
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_viewmats = None
        return v_means, v_covars, v_viewmats

class _LidarProj(torch.autograd.Function):
    """Transform Gaussians from Euclidean to spherical coordinates, azimuth and elevation only."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,   # [C, N, 3]
        covars: Tensor,  # [C, N, 3, 3]
        eps2d: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        means2d, covars2d, depth_compensations = _make_lazy_cuda_func("lidar_proj_fwd")(
            means, covars, eps2d
        )
        ctx.save_for_backward(means, covars)
        ctx.eps2d = eps2d
        return means2d, covars2d, depth_compensations
    
    @staticmethod
    def backward(ctx, v_means2d: Tensor, v_covars2d: Tensor, v_depth_compensations: Tensor):
        means, covars = ctx.saved_tensors
        eps2d = ctx.eps2d
        v_means, v_covars = _make_lazy_cuda_func("lidar_proj_bwd")(
            means,
            covars,
            eps2d,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
            v_depth_compensations.contiguous()
        )
        return v_means, v_covars, None
    
class _ComputeLidarVelocity(torch.autograd.Function):
    """Compute velocity of Gaussians in spherical coordinates."""

    @staticmethod
    def forward(
        ctx,
        p_view,  # [C, N, 3]
        lin_vel, # [C, 3]
        ang_vel,  # [C, 3]
        velocities, # [C, N, 3]
    ) -> Tensor:
        vel = _make_lazy_cuda_func("compute_lidar_velocity_fwd")(
            p_view, lin_vel, ang_vel, velocities
        )
        ctx.save_for_backward(p_view, lin_vel, ang_vel, velocities)
        return vel
    
    @staticmethod
    def backward(ctx, v_vel: Tensor):
        p_view, lin_vel, ang_vel, velocities = ctx.saved_tensors
        v_p_view, v_velocities = _make_lazy_cuda_func("compute_lidar_velocity_bwd")(
            p_view,
            lin_vel,
            ang_vel,
            velocities,
            v_vel.contiguous()
        )
        return v_p_view, None, None, v_velocities

class _FullyFusedProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        velocities: Tensor, # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        linear_velocity: Tensor, # [C, 3]
        angular_velocity: Tensor, # [C, 3]
        rolling_shutter_time: Tensor, # [C]
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations, pix_vels = _make_lazy_cuda_func(
            "fully_fused_projection_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            radii,
            conics,
            compensations
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d

        return radii, means2d, depths, conics, compensations, pix_vels

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations, v_pix_vels):
        (
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_projection_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            v_pix_vels.contiguous(),
            ctx.needs_input_grad[5],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[5]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            None,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    
class _FullyFusedLidarProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        velocities: Tensor, # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        min_elevation: float,
        max_elevation: float,
        min_azimuth: float,
        max_azimuth: float,
        linear_velocity: Tensor, # [C, 3]
        angular_velocity: Tensor, # [C, 3]
        rolling_shutter_time: Tensor, # [C]
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations, pix_vels, depth_compensation = _make_lazy_cuda_func(
            "fully_fused_lidar_projection_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            radii,
            conics,
            compensations,
        )
        ctx.min_elevation = min_elevation
        ctx.max_elevation = max_elevation
        ctx.min_azimuth = min_azimuth
        ctx.max_azimuth = max_azimuth
        ctx.eps2d = eps2d

        return radii, means2d, depths, conics, compensations, pix_vels, depth_compensation

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations, v_pix_vels, v_depth_compensations):
        (
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        min_elevation = ctx.min_elevation
        max_elevation = ctx.max_elevation
        min_azimuth = ctx.min_azimuth
        max_azimuth = ctx.max_azimuth
        eps2d = ctx.eps2d
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_lidar_projection_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            v_pix_vels.contiguous(),
            v_depth_compensations.contiguous(),
            ctx.needs_input_grad[5],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[5]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            None,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,              # [C, N, 2]
        conics: Tensor,               # [C, N, 3]
        colors: Tensor,               # [C, N, D]
        opacities: Tensor,            # [C, N]
        pix_vels: Tensor,             # [C, N, 2]
        rolling_shutter_time: Tensor, # [C]
        backgrounds: Tensor,          # [C, D], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,        # [C, tile_height, tile_width]
        flatten_ids: Tensor,          # [n_isects]
        absgrad: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        render_colors, render_alphas, render_medians, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            rolling_shutter_time,
            backgrounds,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            rolling_shutter_time,
            backgrounds,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas, render_medians

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
        v_render_medians: Tensor,  # [C, H, W, 3]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            rolling_shutter_time,
            backgrounds,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_pix_vels,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            rolling_shutter_time,
            backgrounds,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_pix_vels,
            None,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class _RasterizeToPoints(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        pix_vels: Tensor,  # [C, N, 3]
        depth_compensations: Tensor,  # [C, N, 2]
        backgrounds: Tensor,  # [C, D], Optional
        raster_pts: Tensor,
        width: int,
        height: int,
        tile_width: int,
        tile_height: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        compute_alpha_sum_until_points: bool,
        compute_alpha_sum_until_points_threshold: float,
        absgrad: bool,
        depth_channel_idx: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        render_colors, render_alphas, last_ids, alpha_sum_until_points, median_depths = (
            _make_lazy_cuda_func("rasterize_to_points_fwd")(
                means2d,
                conics,
                colors,
                opacities,
                pix_vels,
                depth_compensations,
                backgrounds,
                raster_pts,
                width,
                height,
                tile_width,
                tile_height,
                compute_alpha_sum_until_points,
                compute_alpha_sum_until_points_threshold,
                isect_offsets,
                flatten_ids,
                depth_channel_idx,
            )
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            depth_compensations,
            backgrounds,
            raster_pts,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_width = tile_width
        ctx.tile_height = tile_height
        ctx.absgrad = absgrad
        ctx.compute_alpha_sum_until_points = compute_alpha_sum_until_points
        ctx.compute_alpha_sum_until_points_threshold = compute_alpha_sum_until_points_threshold
        ctx.depth_channel_idx = depth_channel_idx

        # double to float
        render_alphas = render_alphas.float()
        alpha_sum_until_points = (
            alpha_sum_until_points.float() if compute_alpha_sum_until_points else None
        )
        return render_colors, render_alphas, alpha_sum_until_points, median_depths

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
        v_alpha_sum_until_points: Tensor,  # [C, H, W, 1]
        v_median_depths: Tensor,  # [C, H, W, 1]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            depth_compensations,
            backgrounds,
            raster_pts,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_width = ctx.tile_width
        tile_height = ctx.tile_height
        absgrad = ctx.absgrad
        compute_alpha_sum_until_points = ctx.compute_alpha_sum_until_points
        compute_alpha_sum_until_points_threshold = ctx.compute_alpha_sum_until_points_threshold
        depth_channel_idx = ctx.depth_channel_idx

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_pix_vels,
            v_depth_compensations,
        ) = _make_lazy_cuda_func("rasterize_to_points_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            pix_vels,
            depth_compensations,
            backgrounds,
            raster_pts,
            width,
            height,
            tile_width,
            tile_height,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_alpha_sum_until_points.contiguous()
            if compute_alpha_sum_until_points
            else torch.zeros_like(v_render_alphas),
            absgrad,
            compute_alpha_sum_until_points,
            compute_alpha_sum_until_points_threshold,
            depth_channel_idx,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_pix_vels,
            v_depth_compensations,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class _FullyFusedProjectionPacked(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        velocities: Tensor, # [N, 3]
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        linear_velocity: Tensor, # [C, 3]
        angular_velocity: Tensor, # [C, 3]
        rolling_shutter_time: Tensor, # [C]
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
        calc_compensations: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            indptr,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
            pix_vels,
        ) = _make_lazy_cuda_func("fully_fused_projection_packed_fwd")(
            means,
            covars,  # optional
            quats,  # optional
            scales,  # optional
            velocities, 
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            conics,
            compensations,
            pix_vels
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.sparse_grad = sparse_grad

        return camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations, pix_vels

    @staticmethod
    def backward(
        ctx,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
        v_pix_vels,
    ):
        (
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            conics,
            compensations,
            pix_vels,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        sparse_grad = ctx.sparse_grad

        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_projection_packed_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            velocities,
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            eps2d,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            pix_vels,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            v_pix_vels.contiguous(),
            ctx.needs_input_grad[5],  # viewmats_requires_grad
            sparse_grad,
        )

        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_means,  # [nnz, 3]
                    size=means.size(),  # [N, 3]
                    #is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_covars = None
        else:
            if sparse_grad:
                v_covars = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_covars,  # [nnz, 6]
                    size=covars.size(),  # [N, 6]
                    #is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_quats,  # [nnz, 4]
                    size=quats.size(),  # [N, 4]
                    #is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[3]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_scales,  # [nnz, 3]
                    size=scales.size(),  # [N, 3]
                    #is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[5]:
            v_viewmats = None

        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            None,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class _SphericalHarmonics(torch.autograd.Function):
    """Spherical Harmonics"""

    @staticmethod
    def forward(
        ctx, sh_degree: int, dirs: Tensor, coeffs: Tensor, masks: Tensor
    ) -> Tensor:
        colors = _make_lazy_cuda_func("compute_sh_fwd")(sh_degree, dirs, coeffs, masks)
        ctx.save_for_backward(dirs, coeffs, masks)
        ctx.sh_degree = sh_degree
        ctx.num_bases = coeffs.shape[-2]
        return colors

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        num_bases = ctx.num_bases
        compute_v_dirs = ctx.needs_input_grad[1]
        v_coeffs, v_dirs = _make_lazy_cuda_func("compute_sh_bwd")(
            num_bases,
            sh_degree,
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )
        if not compute_v_dirs:
            v_dirs = None
        return None, v_dirs, v_coeffs, None
