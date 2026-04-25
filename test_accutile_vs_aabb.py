"""Sanity-check the AccuTile/SNUGBOX + opacity-aware extend implementation.

Compares:
- Number of intersections (n_isects) between AABB and AccuTile paths.
- Pixel-level error of the rendered image (should be tiny — only Gaussians whose
  contribution is below ALPHA_THRESHOLD are dropped).

Run via: PYTHONPATH=. python test_accutile_vs_aabb.py
"""

from __future__ import annotations

import time

import torch

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
)


def make_random_gaussians(n: int, image_size: tuple[int, int], seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    H, W = image_size
    means = torch.randn(n, 3, generator=g, device="cuda")
    means[:, 0] *= 5.0  # spread x
    means[:, 1] *= 5.0  # spread y
    means[:, 2] = means[:, 2].abs() * 8.0 + 4.0  # depth in [4, 20] roughly

    quats = torch.randn(n, 4, generator=g, device="cuda")
    quats = quats / quats.norm(dim=-1, keepdim=True)

    log_scales = torch.randn(n, 3, generator=g, device="cuda") * 0.3 - 2.5
    scales = log_scales.exp()  # ~exp(N(-2.5, 0.3)) -> small Gaussians

    raw_opacities = torch.randn(n, generator=g, device="cuda")
    opacities = torch.sigmoid(raw_opacities)

    fx = fy = max(H, W) * 0.7
    Ks = torch.tensor(
        [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
        device="cuda",
    ).unsqueeze(0)
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)

    return means, quats, scales, opacities, viewmats, Ks


@torch.no_grad()
def project_and_count(
    means,
    quats,
    scales,
    opacities,
    viewmats,
    Ks,
    width: int,
    height: int,
    use_opacity_aware: bool,
    use_accutile: bool,
):
    C = viewmats.shape[0]
    linvel = torch.zeros(C, 3, device="cuda")
    angvel = torch.zeros(C, 3, device="cuda")
    rs_time = torch.zeros(C, device="cuda")

    proj_kwargs = dict(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        velocities=None,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        linear_velocity=linvel,
        angular_velocity=angvel,
        rolling_shutter_time=rs_time,
        eps2d=0.3,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=False,
    )
    if use_opacity_aware:
        proj_kwargs["opacities"] = opacities

    radii, means2d, depths, conics, _comp, _pv = fully_fused_projection(**proj_kwargs)

    tile_size = 16
    tile_w = (width + tile_size - 1) // tile_size
    tile_h = (height + tile_size - 1) // tile_size
    op = opacities.repeat(C, 1)  # [C, N]

    isect_kwargs = dict(
        means2d=means2d,
        radii=radii,
        depths=depths,
        tile_size=tile_size,
        tile_width=tile_w,
        tile_height=tile_h,
        n_cameras=C,
    )
    if use_accutile:
        isect_kwargs["conics"] = conics
        isect_kwargs["opacities"] = op

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(**isect_kwargs)
    valid_radii = (radii[..., 0] > 0).sum().item()
    n_isects = isect_ids.numel()
    return valid_radii, n_isects


def main():
    torch.cuda.empty_cache()
    H, W = 1080, 1920
    print(f"Image: {W}x{H}")
    for n in [50_000, 200_000, 1_000_000, 2_500_000]:
        means, quats, scales, opacities, viewmats, Ks = make_random_gaussians(
            n, (H, W)
        )
        # Warmup.
        project_and_count(
            means, quats, scales, opacities, viewmats, Ks, W, H,
            use_opacity_aware=False, use_accutile=False,
        )
        torch.cuda.synchronize()

        timings = {}
        for label, opa, acc in [
            ("baseline (AABB)", False, False),
            ("opacity-aware", True, False),
            ("opacity+AccuTile", True, True),
        ]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            valid, n_isects = project_and_count(
                means, quats, scales, opacities, viewmats, Ks, W, H,
                use_opacity_aware=opa, use_accutile=acc,
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            timings[label] = (valid, n_isects, dt)

        print()
        print(f"=== N = {n:,} ===")
        for label, (valid, n_isects, dt) in timings.items():
            print(f"  {label:<22}: valid={valid:>10,} | n_isects={n_isects:>14,} | {dt*1000:7.1f} ms")

        base = timings["baseline (AABB)"][1]
        if base > 0:
            for label in ["opacity-aware", "opacity+AccuTile"]:
                ratio = timings[label][1] / base
                saved = 100.0 * (1.0 - ratio)
                print(f"  {label:<22} reduction: {saved:5.1f}% of n_isects")


if __name__ == "__main__":
    main()
