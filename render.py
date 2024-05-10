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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, visualize_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import l1_loss
from utils.pose_utils import Pose, Lie
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.loss_utils import ssim
import sys


def render_with_optim_cam(view, gaussians, pipeline, background):
    gt_image = view.test_image.cuda()
    pose = torch.nn.Parameter(
        (torch.zeros_like(view.gaussian_trans)).cuda().requires_grad_(True))
    optimizer = torch.optim.Adam([{'params': [pose],
                                  'lr': 1e-3, "name": "camera pose refine"}])
    for iter in range(1400):
        # view.pose = Pose().compose([Lie().se3_to_SE3(pose), view.pose_gt])
        view.gaussian_trans = pose
        image = render(view, gaussians, pipeline, background)["render"]
        loss = l1_loss(image, gt_image)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(
            f"{iter} loss={loss:03f}", end='\r')

    view.gaussian_trans = pose
    # view.pose = Pose().compose(Lie().se3_to_SE3(pose), view.pose_gt)
    result = render(view, gaussians, pipeline, background)
    rgb = result["render"]
    depth = result["depth"]
    print(f"\n")
    print(f"psnr={psnr(rgb, gt_image).mean().double()}")
    print(f"ssim={ssim(rgb, gt_image).mean().double()}")
    print(f"lpips={lpips(rgb, gt_image).mean().double()}")
    return rgb, visualize_depth(depth)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, optim_pose):
    render_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt")
    refs_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "ref")
    depth_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(refs_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if optim_pose:
            result = render(view, gaussians, pipeline, background)
            rgb = result["render"]
            depth = result["depth"]
        else:
            rgb, depth = render_with_optim_cam(
                view, gaussians, pipeline, background)
        ref = view.original_image[0:3, :, :]
        gt = view.test_image[0:3, :, :]
        torchvision.utils.save_image(rgb, os.path.join(
            render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(ref, os.path.join(
            refs_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(
            gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(
            depth_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, optim_pose: bool):
    # with True:
    # with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,
                  load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
        render_set(dataset.model_path, "train", scene.loaded_iter,
                   scene.getTrainCameras(), gaussians, pipeline, background, optim_pose)

    if not skip_test:
        render_set(dataset.model_path, "test", scene.loaded_iter,
                   scene.getTestCameras(), gaussians, pipeline, background, optim_pose)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--optim_pose", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.device)

    render_sets(model.extract(args), args.iteration,
                pipeline.extract(args), args.skip_train, args.skip_test, args.optim_pose)
