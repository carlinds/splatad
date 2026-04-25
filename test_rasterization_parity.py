"""End-to-end rasterization parity check between AABB and AccuTile paths.

Exercises gsplat.rendering.rasterization (the high-level entry point) twice:
once with the new (opacity-aware + AccuTile) path forced on (default), and once
with conics+opacities zeroed out at the isect_tiles boundary so we get the AABB
fallback. Compares pixel-level output to ensure correctness.
"""

from __future__ import annotations

import math

import torch

import gsplat
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)


def make_random_gaussians(n: int, seed: int = 7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    means = torch.stack(
        [
            (torch.rand(n, generator=g, device="cuda") - 0.5) * 6.0,  # x in [-3, 3]
            (torch.rand(n, generator=g, device="cuda") - 0.5) * 4.0,  # y in [-2, 2]
            torch.rand(n, generator=g, device="cuda") * 8.0 + 4.0,    # z in [4, 12]
        ],
        dim=-1,
    )
    quats = torch.randn(n, 4, generator=g, device="cuda")
    quats = quats / quats.norm(dim=-1, keepdim=True)
    log_scales = torch.randn(n, 3, generator=g, device="cuda") * 0.4 - 2.0
    scales = log_scales.exp()
    raw_opacities = torch.randn(n, generator=g, device="cuda")
    opacities = torch.sigmoid(raw_opacities)
    colors = torch.rand(n, 3, generator=g, device="cuda")
    return means, quats, scales, opacities, colors


@torch.no_grad()
def render(
    means,
    quats,
    scales,
    opacities,
    colors,
    width: int,
    height: int,
    enable_optimizations: bool,
):
    C = 1
    fx = fy = max(width, height) * 0.7
    Ks = torch.tensor(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        device="cuda",
    ).unsqueeze(0)
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)
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
    if enable_optimizations:
        proj_kwargs["opacities"] = opacities
    radii, means2d, depths, conics, _comp, pix_vels = fully_fused_projection(
        **proj_kwargs
    )

    op_per_cam = opacities.repeat(C, 1)  # [C, N]
    tile_size = 16
    tile_w = (width + tile_size - 1) // tile_size
    tile_h = (height + tile_size - 1) // tile_size
    isect_kwargs = dict(
        means2d=means2d,
        radii=radii,
        depths=depths,
        tile_size=tile_size,
        tile_width=tile_w,
        tile_height=tile_h,
        n_cameras=C,
    )
    if enable_optimizations:
        isect_kwargs["conics"] = conics
        isect_kwargs["opacities"] = op_per_cam
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(**isect_kwargs)
    isect_offsets = isect_offset_encode(isect_ids, C, tile_w, tile_h)

    colors_per_cam = colors.expand(C, -1, -1).contiguous()
    bg = torch.zeros(C, 3, device="cuda")
    rendered, alphas, _ = rasterize_to_pixels(
        means2d=means2d,
        conics=conics,
        colors=colors_per_cam,
        opacities=op_per_cam,
        pix_vels=pix_vels,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        rolling_shutter_time=rs_time,
        rolling_shutter_direction=5,
        backgrounds=bg,
    )
    return rendered, isect_ids.numel()


def main():
    H, W = 540, 960  # smaller image for quick iteration
    means, quats, scales, opacities, colors = make_random_gaussians(100_000)

    img_aabb, n_aabb = render(means, quats, scales, opacities, colors, W, H, False)
    img_accu, n_accu = render(means, quats, scales, opacities, colors, W, H, True)

    diff = (img_accu - img_aabb).abs()
    l1 = diff.mean().item()
    linf = diff.max().item()
    psnr = 20 * math.log10(1.0 / (diff.pow(2).mean().clamp_min(1e-12).sqrt()).item())

    print(f"Image: {W}x{H}, N=100,000")
    print(f"  AABB        n_isects = {n_aabb:,}")
    print(f"  AccuTile    n_isects = {n_accu:,}  ({100.0*(1.0 - n_accu/n_aabb):.1f}% reduction)")
    print()
    print("Pixel parity (AABB vs AccuTile):")
    print(f"  mean L1 diff = {l1:.6f}")
    print(f"  max L1 diff  = {linf:.6f}")
    print(f"  PSNR         = {psnr:.2f} dB")
    print()
    if linf < 0.05 and psnr > 35.0:
        print("PASS: AccuTile output is visually equivalent to AABB.")
    else:
        print("WARN: large difference; review the kernel implementation.")


if __name__ == "__main__":
    main()
