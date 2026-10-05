"""FastGS multi-view consistent densification for 2DGS.

FastGS (https://github.com/fastgs/FastGS) scores each Gaussian by how many of a few sampled
training views it covers high-error pixels in, and only densifies Gaussians that several
views agree are under-fit; that keeps the model several times smaller than 3DGS / 2DGS
densification. The score needs per-Gaussian pixel counts from FastGS's rasterizer
(diff_gaussian_rasterization_fastgs). As in Fast-PGSR, the surfels are rasterized there as
thin 3D Gaussians just for scoring; training still renders with the 2DGS surfel rasterizer.
"""
import math
import random

import torch
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

from utils.loss_utils import l1_loss, ssim


def sampling_cameras(cameras, num_cams=10):
    """num_cams distinct random training cameras."""
    return random.sample(list(cameras), min(num_cams, len(cameras)))


def render_fastgs(cam, pc, pipe, bg, mult, metric_map=None):
    """Render the surfels with the FastGS rasterizer; with metric_map, also count for every
    Gaussian how many flagged pixels it contributes to."""
    H, W = int(cam.image_height), int(cam.image_width)
    get_flag = metric_map is not None
    if metric_map is None:
        metric_map = torch.zeros(H * W, dtype=torch.int, device="cuda")
    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
        bg=bg, scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=pc.active_sh_degree, campos=cam.camera_center, mult=mult,
        prefiltered=False, debug=pipe.debug, get_flag=get_flag, metric_map=metric_map.reshape(-1))
    s = pc.get_scaling  # N x 2: surfel extent; give it a negligible third axis
    scales = torch.cat([s, 1e-3 * s.min(dim=1, keepdim=True).values], dim=1)
    means2D = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, device="cuda")
    image, radii, counts = GaussianRasterizer(raster_settings=settings)(
        means3D=pc.get_xyz, means2D=means2D, dc=pc._features_dc, shs=pc._features_rest,
        colors_precomp=None, opacities=pc.get_opacity, scales=scales, rotations=pc.get_rotation,
        cov3D_precomp=None)
    return image, counts


def _normalized_l1(image, gt):
    e = (image - gt).abs().mean(0)
    return (e - e.min()) / (e.max() - e.min())


@torch.no_grad()
def compute_gaussian_score_fastgs(camlist, pc, pipe, bg, opt, densify=False):
    """FastGS scores over camlist: importance = floor(mean over views of the number of
    high-error pixels each Gaussian touches) (densify only), pruning = per-Gaussian
    view-loss-weighted count, normalized to 0..1."""
    counts_sum, score = None, None
    for cam in camlist:
        gt = cam.original_image.cuda()
        image, _ = render_fastgs(cam, pc, pipe, bg, opt.mult)
        view_loss = 0.8 * l1_loss(image, gt) + 0.2 * (1.0 - ssim(image, gt))
        metric_map = (_normalized_l1(image, gt) > opt.loss_thresh).int()
        _, counts = render_fastgs(cam, pc, pipe, bg, opt.mult, metric_map=metric_map)
        counts_sum = counts.clone() if counts_sum is None else counts_sum + counts
        score = view_loss * counts if score is None else score + view_loss * counts
    pruning = (score - score.min()) / (score.max() - score.min())
    importance = torch.div(counts_sum, len(camlist), rounding_mode="floor") if densify else None
    return importance, pruning
