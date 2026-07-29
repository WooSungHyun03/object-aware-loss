#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    need_viewspace_grad=True,
    need_alpha=True,
    need_depth=True,
    need_normal=True,
    need_dist=True,
    need_objectmark=False,
    detach_objectmark_geometry=True,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Keep 2D mean gradients only when a caller will actually consume them.
    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        device="cuda",
        requires_grad=need_viewspace_grad,
    )
    if need_viewspace_grad:
        screenspace_points = screenspace_points + 0
        try:
            screenspace_points.retain_grad()
        except RuntimeError:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    aux_flags = 0
    if need_alpha or need_depth or need_normal:
        aux_flags |= 1
    if need_depth or need_normal:
        aux_flags |= 2
    if need_normal:
        aux_flags |= 4
    if need_dist:
        aux_flags |= 8

    raster_settings_kwargs = dict(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )
    # The release remains compatible with the upstream 2DGS rasterizer. Local
    # optimized builds may expose ``aux_flags`` to skip unused auxiliary maps.
    if "aux_flags" in getattr(GaussianRasterizationSettings, "_fields", ()):
        raster_settings_kwargs["aux_flags"] = aux_flags
    raster_settings = GaussianRasterizationSettings(**raster_settings_kwargs)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W - 1) / 2],
            [0, H / 2, 0, (H - 1) / 2],
            [0, 0, far - near, near],
            [0, 0, 0, 1],
        ]).float().cuda().T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]])
            .permute(0, 2, 1)
            .reshape(-1, 9)
        )  # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # Upstream 2DGS expects an auxiliary-output gradient tensor in backward,
    # even when a caller only uses RGB. This zero-valued dependency preserves
    # the rendered result while keeping that public extension compatible.
    rendered_image = rendered_image + allmap.sum() * 0.0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    need_expected_depth = need_depth or need_normal
    need_median_depth = need_depth or need_normal

    if need_alpha or need_expected_depth or need_median_depth or need_normal or need_dist:
        polarization_alpha = allmap[1:2]
        if need_alpha:
            rets.update({
                "polarization_alpha": polarization_alpha,
                "rend_alpha": polarization_alpha,
            })

        surf_depth = None
        if need_expected_depth or need_median_depth:
            render_depth_expected = allmap[0:1] / polarization_alpha
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

            if need_median_depth:
                render_depth_median = torch.nan_to_num(allmap[5:6], 0, 0)
                surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + pipe.depth_ratio * render_depth_median
            else:
                surf_depth = render_depth_expected

            if need_depth:
                rets["surf_depth"] = surf_depth

        if need_normal:
            render_normal = allmap[2:5]
            render_normal = (
                render_normal.permute(1, 2, 0) @ viewpoint_camera.world_view_transform[:3, :3].T
            ).permute(2, 0, 1)
            surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1)
            surf_normal = surf_normal * polarization_alpha.detach()
            rets.update({
                "rend_normal": render_normal,
                "surf_normal": surf_normal,
            })

        if need_dist:
            rets["rend_dist"] = allmap[6:7]

    if need_objectmark:
        # Dedicated ObjectMark pass: O_j is used as the rasterizer opacity and
        # unit color makes the CUDA alpha-compositing output sum_j T^O_ij O_ij.
        objectmark_opacity = pc.get_objectmark_score_prob
        objectmark_color = torch.ones_like(objectmark_opacity).expand(-1, 3)
        objectmark_bg = torch.zeros_like(bg_color)
        objectmark_settings_updates = {"bg": objectmark_bg}
        if "aux_flags" in getattr(GaussianRasterizationSettings, "_fields", ()):
            objectmark_settings_updates["aux_flags"] = 0
        objectmark_settings = raster_settings._replace(**objectmark_settings_updates)
        objectmark_rasterizer = GaussianRasterizer(raster_settings=objectmark_settings)

        if detach_objectmark_geometry:
            objectmark_means3D = means3D.detach()
            objectmark_means2D = means2D.detach()
            objectmark_scales = scales.detach() if scales is not None else None
            objectmark_rotations = rotations.detach() if rotations is not None else None
            objectmark_cov3D_precomp = cov3D_precomp.detach() if cov3D_precomp is not None else None
        else:
            objectmark_means3D = means3D
            objectmark_means2D = means2D
            objectmark_scales = scales
            objectmark_rotations = rotations
            objectmark_cov3D_precomp = cov3D_precomp

        objectmark_image, _, objectmark_aux = objectmark_rasterizer(
            means3D=objectmark_means3D,
            means2D=objectmark_means2D,
            shs=None,
            colors_precomp=objectmark_color,
            opacities=objectmark_opacity,
            scales=objectmark_scales,
            rotations=objectmark_rotations,
            cov3D_precomp=objectmark_cov3D_precomp,
        )
        objectmark_image = objectmark_image + objectmark_aux.sum() * 0.0
        rendered_object_mark = objectmark_image[0:1].clamp(0.0, 1.0)
        rets["rend_object_mark"] = rendered_object_mark
        # Compatibility alias used by earlier internal checkpoints and viewers.
        rets["objectmark_response"] = rendered_object_mark

    return rets
