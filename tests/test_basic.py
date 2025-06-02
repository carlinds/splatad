"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math

import pytest
import torch
from gsplat._helper import load_test_data

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(device=device)
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
        "min_azimuth": -180,
        "max_azimuth": 180,
        "min_elevation": -45 - 1e-2,
        "max_elevation": 45 + 1e-2,
        "velocities": torch.randn_like(means, requires_grad=True, device=device) * 0.1,
        "linear_velocity": torch.randn(viewmats.shape[0], 3, device=device) * 0.5,
        "angular_velocity": torch.randn(viewmats.shape[0], 3, device=device) * 0.1,
        "rolling_shutter_time": torch.ones(viewmats.shape[0], device=device) * 0.01,
        "n_elev_channels": 32,
        "tile_height": 4,
        "tile_width": 64,
        "azim_resolution": 0.2,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("triu", [False, True])
def test_quat_scale_to_covar_preci(test_data, triu: bool):
    from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    torch.manual_seed(42)

    quats = test_data["quats"]
    scales = test_data["scales"]
    quats.requires_grad = True
    scales.requires_grad = True

    # forward
    covars, precis = quat_scale_to_covar_preci(quats, scales, triu=triu)
    _covars, _precis = _quat_scale_to_covar_preci(quats, scales, triu=triu)
    torch.testing.assert_close(covars, _covars)
    # This test is disabled because the numerical instability.
    # torch.testing.assert_close(precis, _precis, rtol=2e-2, atol=1e-2)
    # if not triu:
    #     I = torch.eye(3, device=device).expand(len(covars), 3, 3)
    #     torch.testing.assert_close(torch.bmm(covars, precis), I)
    #     torch.testing.assert_close(torch.bmm(precis, covars), I)

    # backward
    v_covars = torch.randn_like(covars)
    v_precis = torch.randn_like(precis) * 0.01
    v_quats, v_scales = torch.autograd.grad(
        (covars * v_covars + precis * v_precis).sum(),
        (quats, scales),
    )
    _v_quats, _v_scales = torch.autograd.grad(
        (_covars * v_covars + _precis * v_precis).sum(),
        (quats, scales),
    )
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e-1, atol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_world_to_cam(test_data):
    from gsplat.cuda._torch_impl import _world_to_cam
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci, world_to_cam

    torch.manual_seed(42)

    viewmats = test_data["viewmats"]
    means = test_data["means"]
    scales = test_data["scales"]
    quats = test_data["quats"]
    covars, _ = quat_scale_to_covar_preci(quats, scales)
    means.requires_grad = True
    covars.requires_grad = True
    viewmats.requires_grad = True

    # forward
    means_c, covars_c = world_to_cam(means, covars, viewmats)
    _means_c, _covars_c = _world_to_cam(means, covars, viewmats)
    torch.testing.assert_close(means_c, _means_c)
    torch.testing.assert_close(covars_c, _covars_c)

    # backward
    v_means_c = torch.randn_like(means_c)
    v_covars_c = torch.randn_like(covars_c)
    v_means, v_covars, v_viewmats = torch.autograd.grad(
        (means_c * v_means_c).sum() + (covars_c * v_covars_c).sum(),
        (means, covars, viewmats),
    )
    _v_means, _v_covars, _v_viewmats = torch.autograd.grad(
        (_means_c * v_means_c).sum() + (_covars_c * v_covars_c).sum(),
        (means, covars, viewmats),
    )
    torch.testing.assert_close(v_means, _v_means)
    torch.testing.assert_close(v_covars, _v_covars)
    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_persp_proj(test_data):
    from gsplat.cuda._torch_impl import _persp_proj
    from gsplat.cuda._wrapper import persp_proj, quat_scale_to_covar_preci, world_to_cam

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    covars, _ = quat_scale_to_covar_preci(test_data["quats"], test_data["scales"])
    means, covars = world_to_cam(test_data["means"], covars, viewmats)
    means.requires_grad = True
    covars.requires_grad = True

    # forward
    means2d, covars2d = persp_proj(means, covars, Ks, width, height)
    _means2d, _covars2d = _persp_proj(means, covars, Ks, width, height)
    torch.testing.assert_close(means2d, _means2d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(covars2d, _covars2d, rtol=1e-1, atol=3e-2)

    # backward
    v_means2d = torch.randn_like(means2d)
    v_covars2d = torch.randn_like(covars2d)
    v_means, v_covars = torch.autograd.grad(
        (means2d * v_means2d).sum() + (covars2d * v_covars2d).sum(),
        (means, covars),
    )
    _v_means, _v_covars = torch.autograd.grad(
        (_means2d * v_means2d).sum() + (_covars2d * v_covars2d).sum(),
        (means, covars),
    )
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_covars, _v_covars, rtol=1e-1, atol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("calc_depth_comp_grads", [False, True])
def test_lidar_proj(test_data, calc_depth_comp_grads: bool):
    from gsplat.cuda._torch_impl import _lidar_proj
    from gsplat.cuda._wrapper import lidar_proj, quat_scale_to_covar_preci, world_to_cam

    torch.manual_seed(42)

    viewmats = test_data["viewmats"]
    covars, _ = quat_scale_to_covar_preci(test_data["quats"], test_data["scales"])
    means, covars = world_to_cam(test_data["means"], covars, viewmats)
    means.requires_grad = True
    covars.requires_grad = True

    # forward
    means2d, covars2d, depth_comp = lidar_proj(means, covars, eps2d=1e-3)
    _means2d, _covars2d, _r, _depth_comp = _lidar_proj(means, covars, eps2d=1e-3)
    well_behaved = (means2d[..., -1] > -70) & (means2d[..., -1] < 70)
    torch.testing.assert_close(means2d, _means2d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(covars2d, _covars2d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depth_comp[well_behaved], _depth_comp[well_behaved], rtol=1e-2, atol=1e-2)

    # backward
    v_means2d = torch.randn_like(means2d) * well_behaved[..., None] * (0 if calc_depth_comp_grads else 1)
    v_covars2d = torch.randn_like(covars2d) * well_behaved[..., None, None] * (0 if calc_depth_comp_grads else 1)
    v_depth_comp = torch.randn_like(depth_comp) * well_behaved[..., None] * (1 if calc_depth_comp_grads else 0)
    v_means, v_covars = torch.autograd.grad(
        (means2d * v_means2d).sum() + (covars2d * v_covars2d).sum() + (depth_comp * v_depth_comp).sum(),
        (means, covars),
    )
    _v_means, _v_covars = torch.autograd.grad(
        (_means2d * v_means2d).sum() + (_covars2d * v_covars2d).sum() + (_depth_comp * v_depth_comp).sum(),
        (means, covars),
    )
    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_covars, _v_covars, rtol=5e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_compute_lidar_velocity(test_data):
    from gsplat.cuda._torch_impl import _compute_lidar_velocity
    from gsplat.cuda._wrapper import compute_lidar_velocity, quat_scale_to_covar_preci, world_to_cam

    torch.manual_seed(42)

    viewmats = test_data["viewmats"]
    linear_velocity = test_data["linear_velocity"]
    angular_velocity = test_data["angular_velocity"]
    covars, _ = quat_scale_to_covar_preci(test_data["quats"], test_data["scales"])
    means, _ = world_to_cam(test_data["means"], covars, viewmats)
    velocities = test_data["velocities"].repeat(len(viewmats), 1, 1)
    means.requires_grad = True

    # forward
    spherical_vel = compute_lidar_velocity(means, linear_velocity, angular_velocity, velocities)
    _spherical_vel = _compute_lidar_velocity(means, linear_velocity, angular_velocity, velocities)
    torch.testing.assert_close(spherical_vel, _spherical_vel, rtol=1e-4, atol=1e-4)

    # backward
    v_spherical_vel = torch.randn_like(spherical_vel)
    v_means, v_velocities = torch.autograd.grad(
        (spherical_vel * v_spherical_vel).sum(),
        (means, velocities),
    )
    _v_means, _v_velocities = torch.autograd.grad(
        (_spherical_vel * v_spherical_vel).sum(),
        (means, velocities),
    )
    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_velocities, _v_velocities, rtol=1e-5, atol=1e-3)


pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")


def test_compute_pix_velocity(test_data):
    from gsplat.cuda._torch_impl import _compute_pix_velocity
    from gsplat.cuda._wrapper import compute_pix_velocity, quat_scale_to_covar_preci, world_to_cam

    torch.manual_seed(42)

    viewmats = test_data["viewmats"]
    Ks = test_data["Ks"]
    width = test_data["width"]
    height = test_data["height"]
    linear_velocity = test_data["linear_velocity"]
    angular_velocity = test_data["angular_velocity"]
    covars, _ = quat_scale_to_covar_preci(test_data["quats"], test_data["scales"])
    means, _ = world_to_cam(test_data["means"], covars, viewmats)
    velocities = test_data["velocities"].repeat(len(viewmats), 1, 1)
    means.requires_grad = True
    valid = means[..., -1] > 1e-2

    # forward
    pix_vel = compute_pix_velocity(means, linear_velocity, angular_velocity, velocities, Ks, width, height)
    _pix_vel = _compute_pix_velocity(means, linear_velocity, angular_velocity, velocities, Ks, width, height)
    torch.testing.assert_close(pix_vel[valid], _pix_vel[valid], rtol=1e-3, atol=1e-3)

    # backward
    v_pix_vel = torch.randn_like(pix_vel) * valid[..., None]
    v_means, v_velocities = torch.autograd.grad(
        (pix_vel * v_pix_vel).sum(),
        (means, velocities),
    )
    _v_means, _v_velocities = torch.autograd.grad(
        (_pix_vel * v_pix_vel).sum(),
        (means, velocities),
    )
    torch.testing.assert_close(v_means, _v_means, rtol=2e-3, atol=1e-3)
    torch.testing.assert_close(v_velocities, _v_velocities, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("calc_compensations", [False, True])
@pytest.mark.parametrize("use_velocities", [False, True])
def test_projection(test_data, fused: bool, calc_compensations: bool, use_velocities: bool):
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.cuda._wrapper import fully_fused_projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    linear_velocity = test_data["linear_velocity"]
    angular_velocity = test_data["angular_velocity"]
    rolling_shutter_time = test_data["rolling_shutter_time"]
    velocities = test_data["velocities"] if use_velocities else None
    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    if fused:
        radii, means2d, depths, conics, compensations, pix_vels = fully_fused_projection(
            means,
            None,
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
            calc_compensations=calc_compensations,
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [N, 6]
        radii, means2d, depths, conics, compensations, pix_vels = fully_fused_projection(
            means,
            covars,
            None,
            None,
            velocities,
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            calc_compensations=calc_compensations,
        )
    _covars, _ = quat_scale_to_covar_preci(quats, scales, triu=False)  # [N, 3, 3]
    _radii, _means2d, _depths, _conics, _compensations, _pix_vels = _fully_fused_projection(
        means,
        _covars,
        velocities,
        viewmats,
        Ks,
        width,
        height,
        linear_velocity,
        angular_velocity,
        rolling_shutter_time,
        calc_compensations=calc_compensations,
    )

    # radii is integer so we allow for 1 unit difference
    valid = (radii[..., 0] > 0) & (_radii[..., 0] > 0)
    torch.testing.assert_close(radii[valid], _radii[valid], rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], _conics[valid], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(compensations[valid], _compensations[valid], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(pix_vels[valid], _pix_vels[valid], rtol=1e-4, atol=1e-4)

    # backward
    v_means2d = torch.randn_like(means2d) * valid[..., None]
    v_depths = torch.randn_like(depths) * valid
    v_conics = torch.randn_like(conics) * valid[..., None]
    if calc_compensations:
        v_compensations = torch.randn_like(compensations) * valid
    v_pix_vels = torch.randn_like(pix_vels) * valid[..., None]
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (conics * v_conics).sum()
        + ((compensations * v_compensations).sum() if calc_compensations else 0)
        + (pix_vels * v_pix_vels).sum(),
        (viewmats, quats, scales, means),
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum()
        + ((_compensations * v_compensations).sum() if calc_compensations else 0)
        + (_pix_vels * v_pix_vels).sum(),
        (viewmats, quats, scales, means),
    )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=2e-1, atol=1e-2)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("sparse_grad", [False, True])
@pytest.mark.parametrize("calc_compensations", [False, True])
def test_fully_fused_projection_packed(test_data, fused: bool, sparse_grad: bool, calc_compensations: bool):
    from gsplat.cuda._wrapper import fully_fused_projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    linear_velocity = test_data["linear_velocity"]
    angular_velocity = test_data["angular_velocity"]
    rolling_shutter_time = test_data["rolling_shutter_time"]
    velocities = test_data["velocities"]
    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    if fused:
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
            pix_vels,
        ) = fully_fused_projection(
            means,
            None,
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
            packed=True,
            sparse_grad=sparse_grad,
            calc_compensations=calc_compensations,
        )
        _radii, _means2d, _depths, _conics, _compensations, _pix_vels = fully_fused_projection(
            means,
            None,
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
            packed=False,
            calc_compensations=calc_compensations,
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [N, 6]
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
            pix_vels,
        ) = fully_fused_projection(
            means,
            covars,
            None,
            None,
            velocities,
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            packed=True,
            sparse_grad=sparse_grad,
            calc_compensations=calc_compensations,
        )
        _radii, _means2d, _depths, _conics, _compensations, _pix_vels = fully_fused_projection(
            means,
            covars,
            None,
            None,
            velocities,
            viewmats,
            Ks,
            width,
            height,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            packed=False,
            calc_compensations=calc_compensations,
        )

    # recover packed tensors to full matrices for testing
    __radii = torch.sparse_coo_tensor(torch.stack([camera_ids, gaussian_ids]), radii, _radii.shape).to_dense()
    __means2d = torch.sparse_coo_tensor(torch.stack([camera_ids, gaussian_ids]), means2d, _means2d.shape).to_dense()
    __depths = torch.sparse_coo_tensor(torch.stack([camera_ids, gaussian_ids]), depths, _depths.shape).to_dense()
    __conics = torch.sparse_coo_tensor(torch.stack([camera_ids, gaussian_ids]), conics, _conics.shape).to_dense()
    __pix_vels = torch.sparse_coo_tensor(torch.stack([camera_ids, gaussian_ids]), pix_vels, _pix_vels.shape).to_dense()
    if calc_compensations:
        __compensations = torch.sparse_coo_tensor(
            torch.stack([camera_ids, gaussian_ids]), compensations, _compensations.shape
        ).to_dense()
    sel = (__radii[..., 0] > 0) & (_radii[..., 0] > 0)
    torch.testing.assert_close(__radii[sel], _radii[sel], rtol=0, atol=1)
    torch.testing.assert_close(__means2d[sel], _means2d[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__depths[sel], _depths[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__conics[sel], _conics[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__pix_vels[sel], _pix_vels[sel], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(__compensations[sel], _compensations[sel], rtol=1e-4, atol=1e-3)

    # backward
    v_means2d = torch.randn_like(_means2d) * sel[..., None]
    v_depths = torch.randn_like(_depths) * sel
    v_conics = torch.randn_like(_conics) * sel[..., None]
    v_pix_vels = torch.randn_like(_pix_vels) * sel[..., None]
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum()
        + (_pix_vels * v_pix_vels).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d[__radii[..., 0] > 0]).sum()
        + (depths * v_depths[__radii[..., 0] > 0]).sum()
        + (conics * v_conics[__radii[..., 0] > 0]).sum()
        + (pix_vels * v_pix_vels[__radii[..., 0] > 0]).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    if sparse_grad:
        v_quats = v_quats.to_dense()
        v_scales = v_scales.to_dense()
        v_means = v_means.to_dense()

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("calc_compensations", [False, True])
@pytest.mark.parametrize("use_velocities", [True, False])
def test_lidar_projection(test_data, fused: bool, calc_compensations: bool, use_velocities: bool):
    from gsplat.cuda._torch_impl import _fully_fused_lidar_projection
    from gsplat.cuda._wrapper import fully_fused_lidar_projection, quat_scale_to_covar_preci

    torch.manual_seed(42)
    rescaling_factor = 100

    viewmats = test_data["viewmats"]
    world2cam = viewmats
    lidar2cam = torch.eye(4, device=device).expand(len(viewmats), -1, -1).clone()
    rot = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    lidar2cam[:, :3, :3] = rot
    cam2lidar = lidar2cam.inverse()
    world2lidar = cam2lidar @ world2cam
    lidar2world = world2lidar.inverse()
    lidar2world[:, :3, 3] *= rescaling_factor
    viewmats = lidar2world.inverse()
    quats = test_data["quats"]
    scales = test_data["scales"] * rescaling_factor
    means = test_data["means"] * rescaling_factor
    min_azimuth = test_data["min_azimuth"]
    max_azimuth = test_data["max_azimuth"]
    min_elevation = test_data["min_elevation"]
    max_elevation = test_data["max_elevation"]
    linear_velocity = test_data["linear_velocity"]
    angular_velocity = test_data["angular_velocity"]
    rolling_shutter_time = test_data["rolling_shutter_time"]
    velocities = test_data["velocities"] if use_velocities else None

    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    if fused:
        radii, means2d, depths, conics, compensations, pix_vels, depth_compensations = fully_fused_lidar_projection(
            means,
            None,
            quats,
            scales,
            velocities,
            viewmats,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            calc_compensations=calc_compensations,
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [N, 6]
        radii, means2d, depths, conics, compensations, pix_vels, depth_compensations = fully_fused_lidar_projection(
            means,
            covars,
            None,
            None,
            velocities,
            viewmats,
            linear_velocity,
            angular_velocity,
            rolling_shutter_time,
            min_elevation,
            max_elevation,
            min_azimuth,
            max_azimuth,
            calc_compensations=calc_compensations,
        )
    _covars, _ = quat_scale_to_covar_preci(quats, scales, triu=False)  # [N, 3, 3]
    _radii, _means2d, _depths, _conics, _compensations, _pix_vels, _depth_compensations = _fully_fused_lidar_projection(
        means,
        _covars,
        velocities,
        viewmats,
        linear_velocity,
        angular_velocity,
        rolling_shutter_time,
        min_elevation,
        max_elevation,
        min_azimuth,
        max_azimuth,
        calc_compensations=calc_compensations,
    )

    valid = (radii[..., 0] > 0) & (_radii[..., 0] > 0)
    torch.testing.assert_close(radii[valid], _radii[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], _conics[valid], rtol=1e-3, atol=1e-3)
    if calc_compensations:
        torch.testing.assert_close(compensations[valid], _compensations[valid], rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(pix_vels[valid], _pix_vels[valid], rtol=2e-4, atol=1e-4)
    torch.testing.assert_close(depth_compensations[valid], _depth_compensations[valid], rtol=1e-1, atol=1e-2)

    # # backward
    v_means2d = torch.randn_like(means2d) * valid[..., None]
    v_depths = torch.randn_like(depths) * valid
    v_conics = torch.randn_like(conics) * valid[..., None]
    if calc_compensations:
        v_compensations = torch.randn_like(compensations) * valid
    v_pix_vels = torch.randn_like(pix_vels) * valid[..., None]
    v_depth_compensations = torch.randn_like(depth_compensations) * valid[..., None]
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (conics * v_conics).sum()
        + ((compensations * v_compensations).sum() if calc_compensations else 0)
        + (pix_vels * v_pix_vels).sum()
        + (depth_compensations * v_depth_compensations).sum(),
        (viewmats, quats, scales, means),
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum()
        + ((_compensations * v_compensations).sum() if calc_compensations else 0)
        + (_pix_vels * v_pix_vels).sum()
        + (_depth_compensations * v_depth_compensations).sum(),
        (viewmats, quats, scales, means),
    )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=2e-1, atol=1e-1)
    torch.testing.assert_close(v_scales, _v_scales, rtol=2e-1, atol=5e-1)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_isect(test_data):
    from gsplat.cuda._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    torch.manual_seed(42)

    C, N = 3, 1000
    width, height = 40, 60
    means2d = torch.randn(C, N, 2, device=device) * width
    radii = torch.randint(0, width, (C, N, 2), device=device, dtype=torch.int32)
    depths = torch.rand(C, N, device=device)

    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    _tiles_per_gauss, _isect_ids, _gauss_ids = _isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    _isect_offsets = _isect_offset_encode(_isect_ids, C, tile_width, tile_height)

    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(flatten_ids, _gauss_ids)
    torch.testing.assert_close(isect_offsets, _isect_offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_isect_lidar(test_data):
    from gsplat.cuda._torch_impl import _isect_lidar_tiles, _isect_offset_encode
    from gsplat.cuda._wrapper import isect_lidar_tiles, isect_offset_encode

    torch.manual_seed(42)

    C, N = 3, 100
    width, height = 360.0, 90.0
    min_azimuth = -180

    min_elev = -height / 2.0
    tile_elev_res = height / 16

    azim_resolution = 0.2
    tile_azim_resolution = azim_resolution * 16
    n_azim_tiles = math.ceil(360 / tile_azim_resolution)

    n_elev_tiles = 16
    elev_boundaries = torch.linspace(-height / 2.0, height / 2.0, n_elev_tiles + 1, device=device)

    means2d = torch.zeros((C, N, 2), device=device)
    means2d[:, :, 0] = torch.rand(C, N, device=device) * width - width / 2
    means2d[:, :, 1] = torch.rand(C, N, device=device) * height - height / 2
    radii = torch.rand(C, N, 2, device=device, dtype=torch.float32) / 2
    depths = torch.rand(C, N, device=device)

    tiles_per_gauss, isect_ids, flatten_ids = isect_lidar_tiles(
        means2d, radii, depths, elev_boundaries, tile_azim_resolution, min_azimuth
    )
    isect_offsets = isect_offset_encode(isect_ids, C, n_azim_tiles, n_elev_tiles)

    _tiles_per_gauss, _isect_ids, _gauss_ids = _isect_lidar_tiles(
        means2d, radii, depths, elev_boundaries, tile_azim_resolution, min_azimuth
    )
    _isect_offsets = _isect_offset_encode(_isect_ids, C, n_azim_tiles, n_elev_tiles)

    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(flatten_ids, _gauss_ids)
    torch.testing.assert_close(isect_offsets, _isect_offsets)

    first_tile_ids = _gauss_ids[: _isect_offsets[0, 0, 1]]
    gauss_azim_lb = means2d.view(-1, 2)[first_tile_ids, 0] - radii.view(-1)[first_tile_ids] + min_azimuth
    gauss_azim_ub = means2d.view(-1, 2)[first_tile_ids, 0] + radii.view(-1)[first_tile_ids] + min_azimuth
    azim_lb_inside = ((gauss_azim_lb) > 0) & ((gauss_azim_lb) < (0 + tile_azim_resolution))
    azim_ub_inside = ((gauss_azim_ub) > 0) & ((gauss_azim_ub) < (0 + tile_azim_resolution))
    azim_bounds_include_tile = ((gauss_azim_lb) < 0) & ((gauss_azim_ub) > (0 + tile_elev_res))
    inside_azim_bounds = azim_lb_inside | azim_ub_inside | azim_bounds_include_tile
    torch.testing.assert_close(inside_azim_bounds, torch.ones_like(inside_azim_bounds), atol=0, rtol=0)

    gauss_elev_lb = means2d.view(-1, 2)[first_tile_ids, 1] - radii[..., 1].view(-1)[first_tile_ids]
    gauss_elev_ub = means2d.view(-1, 2)[first_tile_ids, 1] + radii[..., 1].view(-1)[first_tile_ids]
    elev_lb_inside = ((gauss_elev_lb) > min_elev) & ((gauss_elev_lb) < (min_elev + tile_elev_res))
    elev_ub_inside = ((gauss_elev_ub) > min_elev) & ((gauss_elev_ub) < (min_elev + tile_elev_res))
    elev_bounds_include_tile = ((gauss_elev_lb) < min_elev) & ((gauss_elev_ub) > (min_elev + tile_elev_res))
    inside_elev_bounds = elev_lb_inside | elev_ub_inside | elev_bounds_include_tile
    torch.testing.assert_close(inside_elev_bounds, torch.ones_like(inside_elev_bounds), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_map_points_to_lidar_tiles(test_data):
    from gsplat.cuda._wrapper import map_points_to_lidar_tiles, points_mapping_offset_encode

    torch.manual_seed(42)

    C, N = 3, 100_000
    width, height = 360.0, 90.0
    min_azimuth = -180

    azim_resolution = 0.2
    tile_azim_resolution = azim_resolution * 16
    n_azim_tiles = math.ceil(360 / tile_azim_resolution)

    n_elev_tiles = 16
    elev_boundaries = torch.linspace(-height / 2.0, height / 2.0, n_elev_tiles + 1, device=device)

    points2d = torch.zeros((C, N, 2), device=device)
    points2d[:, :, 0] = torch.rand(C, N, device=device) * width - width / 2
    points2d[:, :, 1] = torch.rand(C, N, device=device) * height - height / 2

    points_tile_ids, flatten_ids = map_points_to_lidar_tiles(
        points2d, elev_boundaries, tile_azim_resolution, min_azimuth
    )
    tile_offsets = points_mapping_offset_encode(points_tile_ids, C, n_azim_tiles, n_elev_tiles)

    camera_ids = points_tile_ids >> 32
    tile_ids = points_tile_ids & 0xFFFFFFFF

    assert camera_ids.shape[0] == C * N
    assert camera_ids.unique().shape[0] == C
    for i in range(C):
        assert camera_ids[i * N : (i + 1) * N].unique()[0] == i

    n_max_tiles = n_elev_tiles * n_azim_tiles
    assert tile_ids.shape[0] == C * N
    assert tile_ids.unique().shape[0] <= n_max_tiles
    for i in range(C):
        tile_ids_i = tile_ids[i * N : (i + 1) * N].unique()
        assert (tile_ids_i[:-1] <= tile_ids_i[1:]).all()

    assert flatten_ids.shape[0] == C * N
    for i in range(C):
        points_in_last_tile = points2d.view(-1, 2)[flatten_ids[i * N : (i + 1) * N][-10:]]
        assert (points_in_last_tile[:, 0] <= (width / 2)).all()
        assert (points_in_last_tile[:, 0] >= (width / 2 - tile_azim_resolution)).all()
        assert (points_in_last_tile[:, 1] <= elev_boundaries[-1]).all()
        assert (points_in_last_tile[:, 1] >= elev_boundaries[-2]).all()

    assert tile_offsets.shape == (C, n_elev_tiles, n_azim_tiles)
    for i in range(C):
        tile_offsets_i = tile_offsets[i]
        # Check that we have points in all tiles
        assert ((tile_offsets_i.flatten()[1:] - tile_offsets_i.flatten()[:-1]) > 0).all()
        assert (tile_offsets_i.flatten()[1:] >= tile_offsets_i.flatten()[:-1]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_populate_image_from_points(test_data):
    from gsplat.cuda._wrapper import map_points_to_lidar_tiles, points_mapping_offset_encode, populate_image_from_points

    torch.manual_seed(42)

    C, N = 3, 1_000_000
    min_azimuth = test_data["min_azimuth"]
    max_azimuth = test_data["max_azimuth"]
    min_elevation = test_data["min_elevation"]
    max_elevation = test_data["max_elevation"]
    tile_height = test_data["tile_height"]
    tile_width = test_data["tile_width"]
    n_elev_channels = test_data["n_elev_channels"]
    n_elev_tiles = n_elev_channels // tile_height
    azim_resolution = test_data["azim_resolution"]

    tile_azim_resolution = azim_resolution * tile_width
    n_azim_tiles = math.ceil(360 / tile_azim_resolution)

    elev_boundaries = torch.linspace(min_elevation - 1e-2, max_elevation + 1e-2, n_elev_tiles + 1, device=device)

    points = torch.zeros((C, N, 5), device=device)
    points[:, :, 0] = torch.rand(C, N, device=device) * (max_azimuth - min_azimuth) - max_azimuth
    points[:, :, 1] = torch.rand(C, N, device=device) * (max_elevation - min_elevation) - max_elevation
    points[:, :, 2] = torch.rand(C, N, device=device) * 100
    points[:, :, 3] = torch.rand(C, N, device=device)
    points[:, :, 4] = torch.rand(C, N, device=device)

    points_tile_ids, flatten_ids = map_points_to_lidar_tiles(
        points[:, :, :2], elev_boundaries, tile_azim_resolution, min_azimuth
    )
    tile_offsets = points_mapping_offset_encode(points_tile_ids, C, n_azim_tiles, n_elev_tiles)

    raster_pts_image = populate_image_from_points(
        points,
        image_width=n_azim_tiles * tile_width,
        image_height=n_elev_channels,
        tile_width=tile_width,
        tile_height=tile_height,
        tile_offsets=tile_offsets,
        flatten_id=flatten_ids,
    )

    assert raster_pts_image.shape == (C, n_elev_channels, n_azim_tiles * tile_width, 5)
    assert raster_pts_image.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3, 32, 64])
def test_rasterize_to_pixels(test_data, channels: int):
    from gsplat.cuda._torch_impl import _rasterize_to_pixels
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        quat_scale_to_covar_preci,
        rasterize_to_pixels,
    )

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    C = len(Ks)
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    velocities = test_data["velocities"].detach()
    colors = torch.randn(C, len(means), channels, device=device)
    backgrounds = torch.rand((C, colors.shape[-1]), device=device)
    linear_velocity = test_data["linear_velocity"]
    angular_velocity = test_data["angular_velocity"]
    rolling_shutter_time = test_data["rolling_shutter_time"]

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    # project Gaussians to 2D
    radii, means2d, depths, conics, _, pix_vels = fully_fused_projection(
        means,
        covars,
        velocities,
        None,
        None,
        viewmats,
        Ks,
        width,
        height,
        linear_velocity,
        angular_velocity,
        rolling_shutter_time,
    )
    opacities = opacities.repeat(C, 1)

    # identify intersecting tiles
    tile_size = 16 if channels <= 32 else 4
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    pix_vels.requires_grad = True
    backgrounds.requires_grad = True

    # forward
    render_colors, render_alphas, render_medians = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        pix_vels,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        rolling_shutter_time=rolling_shutter_time,
        backgrounds=backgrounds,
    )
    _render_colors, _render_alphas = _rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        pix_vels,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        rolling_shutter_time=rolling_shutter_time,
        backgrounds=backgrounds,
    )
    torch.testing.assert_close(render_colors, _render_colors, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(render_alphas, _render_alphas, rtol=1e-4, atol=1e-4)

    # backward
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

    v_means2d, v_conics, v_colors, v_opacities, v_pix_vels, v_backgrounds = torch.autograd.grad(
        (render_colors * v_render_colors).sum() + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, pix_vels, backgrounds),
    )
    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_opacities,
        _v_pix_vels,
        _v_backgrounds,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum() + (_render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, pix_vels, backgrounds),
    )
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(v_conics, _v_conics, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_pix_vels, _v_pix_vels, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("compute_alpha_sum_until_points", [False, True])
@pytest.mark.parametrize("use_depth_compensations", [False, True])
def test_rasterize_to_points(test_data, compute_alpha_sum_until_points: bool, use_depth_compensations: bool):
    from gsplat.cuda._torch_impl import _rasterize_to_points
    from gsplat.cuda._wrapper import (
        fully_fused_lidar_projection,
        isect_lidar_tiles,
        isect_offset_encode,
        quat_scale_to_covar_preci,
        rasterize_to_points,
    )

    torch.manual_seed(42)
    Ks = test_data["Ks"]
    C = len(Ks)
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    opacities = test_data["opacities"]
    velocities = test_data["velocities"].detach()
    num_sensors = viewmats.shape[0]
    min_azimuth = test_data["min_azimuth"]
    max_azimuth = test_data["max_azimuth"]
    min_elevation = test_data["min_elevation"]
    max_elevation = test_data["max_elevation"]
    linear_velocity = test_data["linear_velocity"]
    angular_velocity = test_data["angular_velocity"]
    rolling_shutter_time = test_data["rolling_shutter_time"]
    min_azimuth = test_data["min_azimuth"]
    max_azimuth = test_data["max_azimuth"]
    min_elevation = test_data["min_elevation"]
    max_elevation = test_data["max_elevation"]
    tile_height = test_data["tile_height"]
    tile_width = test_data["tile_width"]
    n_elev_channels = test_data["n_elev_channels"]
    n_elev_tiles = n_elev_channels // tile_height
    azim_resolution = test_data["azim_resolution"]

    # setup raster points
    elevations = torch.linspace(min_elevation, max_elevation, n_elev_channels, device=device)
    n_elev_tiles = n_elev_channels // tile_height
    elev_boundaries = torch.linspace(min_elevation - 1e-2, max_elevation + 1e-2, n_elev_tiles + 1, device=device)
    height = n_elev_channels

    width = math.ceil(360 / azim_resolution)
    azimuths = torch.linspace(
        min_azimuth + azim_resolution / 2, max_azimuth - azim_resolution / 2, width, device=device
    )
    tile_azim_resolution = azim_resolution * tile_width
    n_azim_tiles = math.ceil(360 / tile_azim_resolution)

    azimuth_elevation = torch.stack(torch.meshgrid(elevations, azimuths), dim=-1)[
        ..., [1, 0]
    ]  # [height, width, 2] (azim, elev)
    ranges = torch.rand(height, width, 1, device=device) * 10
    keep_range_mask = torch.rand(height, width, device=device) > 0.1
    raster_pts = torch.cat([azimuth_elevation, ranges], dim=-1)
    raster_pts = raster_pts.unsqueeze(0).repeat(
        num_sensors, 1, 1, 1
    )  # [num_sensors, height, width, 3] (azim, elev, range)
    # add randomness
    raster_pts += torch.randn_like(raster_pts) * 0.01 * keep_range_mask[None, ..., None]

    # add timestamps
    raster_pts = torch.cat(
        [raster_pts, rolling_shutter_time.max() * torch.randn(raster_pts[..., 0:1].shape, device=raster_pts.device)],
        dim=-1,
    )

    lidar_features = torch.randn(C, len(means), 5, device=device) * ranges.max() * 1.5
    lidar_features[..., -1] = torch.rand_like(lidar_features[..., -1]) * ranges.max() * 1.5

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    # project Gaussians to 2D
    radii, means2d, depths, conics, _, pix_vels, depth_compensations = fully_fused_lidar_projection(
        means,
        covars,
        None,
        None,
        velocities,
        viewmats,
        linear_velocity,
        angular_velocity,
        rolling_shutter_time,
        min_elevation,
        max_elevation,
        min_azimuth,
        max_azimuth,
        eps2d=0.0,
    )
    opacities = opacities.repeat(C, 1)

    # identify intersecting tiles
    _, isect_ids, flatten_ids = isect_lidar_tiles(
        means2d, radii, depths, elev_boundaries, tile_azim_resolution, min_azimuth
    )
    isect_offsets = isect_offset_encode(isect_ids, C, n_azim_tiles, n_elev_tiles)

    means2d.requires_grad = True
    conics.requires_grad = True
    lidar_features.requires_grad = True
    opacities.requires_grad = True
    pix_vels.requires_grad = True
    depth_compensations = depth_compensations * use_depth_compensations
    depth_compensations.requires_grad = True

    # forward
    render_lidar_features, render_alphas, alpha_sum_until_points, _ = rasterize_to_points(
        means2d,
        conics,
        lidar_features,
        opacities,
        pix_vels,
        depth_compensations,
        raster_pts,
        width,
        height,
        tile_width,
        tile_height,
        isect_offsets,
        flatten_ids,
        compute_alpha_sum_until_points=compute_alpha_sum_until_points,
        compute_alpha_sum_until_points_threshold=0.5,
        # backgrounds=backgrounds,
    )
    _render_lidar_features, _render_alphas, _alpha_sum_until_points = _rasterize_to_points(
        means2d,
        conics,
        lidar_features,
        opacities,
        pix_vels,
        depth_compensations,
        raster_pts,
        width,
        height,
        tile_width,
        tile_height,
        isect_offsets,
        flatten_ids,
        compute_alpha_sum_until_points=compute_alpha_sum_until_points,
        compute_alpha_sum_until_points_threshold=0.5,
        # backgrounds=backgrounds,
    )
    torch.testing.assert_close(render_lidar_features, _render_lidar_features, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(render_alphas, _render_alphas, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(alpha_sum_until_points, _alpha_sum_until_points, rtol=1e-3, atol=1e-3)

    # backward
    v_render_colors = torch.randn_like(render_lidar_features)
    v_render_alphas = torch.randn_like(render_alphas)
    v_alpha_sum_until_points = torch.randn_like(alpha_sum_until_points) if compute_alpha_sum_until_points else None

    losses = (render_lidar_features * v_render_colors).sum() + (render_alphas * v_render_alphas).sum()
    if compute_alpha_sum_until_points:
        losses += (alpha_sum_until_points * v_alpha_sum_until_points).sum()
    v_means2d, v_conics, v_colors, v_opacities, v_pix_vels, v_depth_compensations = torch.autograd.grad(
        losses,
        (means2d, conics, lidar_features, opacities, pix_vels, depth_compensations),
    )

    _losses = (_render_lidar_features * v_render_colors).sum() + (_render_alphas * v_render_alphas).sum()
    if compute_alpha_sum_until_points:
        _losses += (_alpha_sum_until_points * v_alpha_sum_until_points).sum()
    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_opacities,
        _v_pix_vels,
        _v_depth_compensations,
    ) = torch.autograd.grad(
        _losses,
        (means2d, conics, lidar_features, opacities, pix_vels, depth_compensations),
    )
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_conics, _v_conics, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_colors, _v_colors, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_pix_vels, _v_pix_vels, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_depth_compensations, _v_depth_compensations, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("use_random_test_points", [False, True])
@pytest.mark.parametrize("compute_alpha_sum_until_points", [False, True])
def test_accumulate_until_points(test_data, use_random_test_points: bool, compute_alpha_sum_until_points: bool):
    from gsplat.cuda._torch_impl import _rasterize_to_points
    from gsplat.cuda._wrapper import isect_lidar_tiles, isect_offset_encode, rasterize_to_points

    torch.manual_seed(42)
    min_azimuth = test_data["min_azimuth"]
    max_azimuth = test_data["max_azimuth"]
    min_elevation = test_data["min_elevation"]
    max_elevation = test_data["max_elevation"]
    tile_height = test_data["tile_height"]
    tile_width = test_data["tile_width"]
    n_elev_channels = test_data["n_elev_channels"]
    n_elev_tiles = n_elev_channels // tile_height
    azim_resolution = test_data["azim_resolution"]

    # setup lidar raster points
    elevations = torch.linspace(min_elevation, max_elevation, n_elev_channels, device=device)
    n_elev_tiles = n_elev_channels // tile_height
    elev_boundaries = torch.linspace(min_elevation - 1e-2, max_elevation + 1e-2, n_elev_tiles + 1, device=device)
    height = n_elev_channels

    width = math.ceil(360 / azim_resolution)
    azimuths = torch.linspace(
        min_azimuth + azim_resolution / 2, max_azimuth - azim_resolution / 2, width, device=device
    )
    tile_azim_resolution = azim_resolution * tile_width
    n_azim_tiles = math.ceil(360 / tile_azim_resolution)

    azimuth_elevation = torch.stack(torch.meshgrid(elevations, azimuths), dim=-1)[
        ..., [1, 0]
    ]  # [height, width, 2] (azim, elev)
    raster_pts = torch.cat([azimuth_elevation, torch.zeros_like(azimuth_elevation)], dim=-1).unsqueeze(0)

    if use_random_test_points:
        n_gaussians = 100_000
        raster_pts[..., 2] = torch.rand_like(raster_pts[..., 2]) * 150.0

        means2d = torch.cat(
            [
                torch.rand((1, n_gaussians, 1), device=device) * 2 * 180 - 180.0,
                torch.rand((1, n_gaussians, 1), device=device) * 2 * 90 - 90.0,
            ],
            dim=-1,
        )
        radii = torch.rand((1, n_gaussians, 2), device=device) * 10.0
        sigma_x = torch.rand(1, n_gaussians, device=device) * radii[..., 0]
        sigma_y = torch.rand(1, n_gaussians, device=device) * radii[..., 1]
        sigma_xy = torch.rand(1, n_gaussians, device=device) * radii[..., 0] * radii[..., 1]
        conics = torch.linalg.inv(
            torch.stack([torch.stack([sigma_x, sigma_xy], dim=-1), torch.stack([sigma_xy, sigma_y], dim=-1)], dim=-1)
        ).reshape(1, n_gaussians, -1)[..., [0, 1, 3]]
        depths = torch.rand((1, n_gaussians), device=device) * 150.0
        opacities = torch.rand_like(depths)
        pix_vels = (torch.rand_like(conics) - 0.5) * 10.0
    else:
        n_gaussians = 10
        # testing accumulation against a single point for simplicity
        test_point = (16, 900, 85.0)
        raster_pts[0, test_point[0], test_point[1], 2] = test_point[2]

        # all gaussians are located at the test point, such that only their range will determine if they should be accumulated or not
        means2d = (
            (
                raster_pts[0, test_point[0], test_point[1], :2].clone()
                + 0.1 * torch.ones_like(raster_pts[0, test_point[0], test_point[1], :2])
            )
            .repeat(n_gaussians, 1)
            .unsqueeze(0)
        )
        radii = torch.tensor([1.0, 1.0], device=device).repeat(n_gaussians).reshape(1, n_gaussians, 2)
        sigma_x = torch.ones(1, n_gaussians, device=device)
        sigma_y = torch.ones(1, n_gaussians, device=device)
        sigma_xy = torch.zeros(1, n_gaussians, device=device)
        conics = torch.linalg.inv(
            torch.stack([torch.stack([sigma_x, sigma_xy], dim=-1), torch.stack([sigma_xy, sigma_y], dim=-1)], dim=-1)
        ).reshape(1, n_gaussians, -1)[..., [0, 1, 3]]
        depths = torch.linspace(20.0, 120.0, n_gaussians, device=device).unsqueeze(0)
        opacities = torch.rand_like(depths)
        pix_vels = torch.zeros_like(conics)

    lidar_features = torch.randn((*means2d.shape[:2], 5), device=device)
    lidar_features = torch.cat([lidar_features, depths.unsqueeze(-1)], dim=-1)

    # identify intersecting tiles
    _, isect_ids, flatten_ids = isect_lidar_tiles(
        means2d, radii, depths, elev_boundaries, tile_azim_resolution, min_azimuth
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, n_azim_tiles, n_elev_tiles)

    means2d.requires_grad = True
    conics.requires_grad = True
    lidar_features.requires_grad = True
    opacities.requires_grad = True
    pix_vels.requires_grad = True

    # forward
    point_epsilon = 0.5
    depth_compensations = torch.zeros_like(means2d)
    render_lidar_features, render_alphas, alpha_sum_until_points, _ = rasterize_to_points(
        means2d,
        conics,
        lidar_features,
        opacities,
        pix_vels,
        depth_compensations,
        raster_pts,
        width,
        height,
        tile_width,
        tile_height,
        isect_offsets,
        flatten_ids,
        compute_alpha_sum_until_points=compute_alpha_sum_until_points,
        compute_alpha_sum_until_points_threshold=point_epsilon,
    )
    _render_lidar_features, _render_alphas, _alpha_sum_until_points = _rasterize_to_points(
        means2d,
        conics,
        lidar_features,
        opacities,
        pix_vels,
        depth_compensations,
        raster_pts,
        width,
        height,
        tile_width,
        tile_height,
        isect_offsets,
        flatten_ids,
        compute_alpha_sum_until_points=compute_alpha_sum_until_points,
        compute_alpha_sum_until_points_threshold=point_epsilon,
    )

    # forward
    torch.testing.assert_close(render_lidar_features, _render_lidar_features, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(render_alphas, _render_alphas, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(alpha_sum_until_points, _alpha_sum_until_points, rtol=1e-3, atol=1e-3)

    # backward
    v_render_colors = torch.rand_like(render_lidar_features)
    v_render_alphas = torch.rand_like(render_alphas)
    v_alpha_sum_until_points = torch.rand_like(alpha_sum_until_points) if compute_alpha_sum_until_points else None

    losses = (render_lidar_features * v_render_colors).sum() + (render_alphas * v_render_alphas).sum()
    if compute_alpha_sum_until_points:
        losses += (alpha_sum_until_points * v_alpha_sum_until_points).sum()
    v_means2d, v_conics, v_colors, v_opacities, v_pix_vels = torch.autograd.grad(
        losses,
        (means2d, conics, lidar_features, opacities, pix_vels),
    )

    _losses = (_render_lidar_features * v_render_colors).sum() + (_render_alphas * v_render_alphas).sum()
    if compute_alpha_sum_until_points:
        _losses += (_alpha_sum_until_points * v_alpha_sum_until_points).sum()
    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_opacities,
        _v_pix_vels,
    ) = torch.autograd.grad(
        _losses,
        (means2d, conics, lidar_features, opacities, pix_vels),
    )
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(v_conics, _v_conics, rtol=5e-2, atol=1e-2)
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=5e-2, atol=1e-2)
    torch.testing.assert_close(v_pix_vels, _v_pix_vels, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3, 4])
def test_sh(test_data, sh_degree: int):
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.cuda._wrapper import spherical_harmonics

    torch.manual_seed(42)

    N = 1000
    coeffs = torch.randn(N, (4 + 1) ** 2, 3, device=device)
    dirs = torch.randn(N, 3, device=device)
    coeffs.requires_grad = True
    dirs.requires_grad = True

    colors = spherical_harmonics(sh_degree, dirs, coeffs)
    _colors = _spherical_harmonics(sh_degree, dirs, coeffs)
    torch.testing.assert_close(colors, _colors, rtol=1e-4, atol=1e-4)

    v_colors = torch.randn_like(colors)

    v_coeffs, v_dirs = torch.autograd.grad(
        (colors * v_colors).sum(), (coeffs, dirs), retain_graph=True, allow_unused=True
    )
    _v_coeffs, _v_dirs = torch.autograd.grad(
        (_colors * v_colors).sum(), (coeffs, dirs), retain_graph=True, allow_unused=True
    )
    torch.testing.assert_close(v_coeffs, _v_coeffs, rtol=1e-4, atol=1e-4)
    if sh_degree > 0:
        torch.testing.assert_close(v_dirs, _v_dirs, rtol=1e-4, atol=1e-4)
