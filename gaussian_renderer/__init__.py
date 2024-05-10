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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.cameras import align_cameras
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.pose_utils import Quaternion, Lie, interpolation_linear, interpolation_spline


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, mode="train", interp_alpha=0.0):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if mode == "train":
        # pose = viewpoint_camera.get_train_pose(interp_alpha)
        pose = viewpoint_camera.pose
        # trans_offset = viewpoint_camera.get_trans_offset(interp_alpha)
        # scale_offset = viewpoint_camera.get_scale_offset(interp_alpha)
        # rot_offset = viewpoint_camera.get_rot_offset(interp_alpha)
    elif mode == "test":
        with torch.no_grad():
            # pose = aligned_pose
            pose = viewpoint_camera.pose
            # trans_offset = 0
            # scale_offset = 1
            # rot_offset = torch.tensor([0, 0, 0, 1]).float().cuda()
        # pose = viewpoint_camera.pose_gt

    world_view_transform = torch.zeros((4, 4)).cuda()
    T = pose[:3, 3:].transpose(0, 1)
    R = pose[:3, :3]
    world_view_transform[:3, :3] = R
    world_view_transform[3, :3] = T
    world_view_transform[3, 3] = 1.0

    camera_center = world_view_transform.inverse()[3, :3]

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        projmatrix=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    if mode == "train":
        gaussian_trans = viewpoint_camera.get_gaussian_trans(interp_alpha)
        means3D = torch.cat([means3D, torch.ones_like(
            means3D[..., :1])], dim=-1)@gaussian_trans.transpose(-1, -2)
    elif mode == "test":
        with torch.no_grad():
            means3D = torch.cat([means3D, torch.ones_like(
                means3D[..., :1])], dim=-1)@Lie().se3_to_SE3(viewpoint_camera.gaussian_trans[0]).transpose(-1, -2)
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(
                1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (
                pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, depth_image, alpha_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        viewmatrix=world_view_transform)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth_image,
            "alpha": alpha_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
