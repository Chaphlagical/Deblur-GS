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
import numpy as np


def angle_to_rotation_matrix(a, axis):
    # get the rotation matrix from Euler angle around specific axis
    roll = dict(X=1, Y=2, Z=0)[axis]
    O = torch.zeros_like(a)
    I = torch.ones_like(a)
    M = torch.stack(
        [
            torch.stack([a.cos(), -a.sin(), O], dim=-1),
            torch.stack([a.sin(), a.cos(), O], dim=-1),
            torch.stack([O, O, I], dim=-1),
        ],
        dim=-2,
    )
    M = M.roll((roll, roll), dims=(-2, -1))
    return M


def get_novel_view_poses(pose_anchor, N=60, scale=1):
    # create circular viewpoints (small oscillations)
    theta = torch.arange(N) / N * 4 * np.pi
    R_x = angle_to_rotation_matrix((-theta.sin() * 0.05 / 3).asin(), "X")
    R_y = angle_to_rotation_matrix((-theta.cos() * 0.05).asin(), "Y")
    pose_rot = Pose()(R=R_y @ R_x)
    pose_shift = Pose()(t=[0, 0, 4.0 * scale])
    pose_shift2 = Pose()(t=[0, 0, -4.0 * scale])
    pose_oscil = Pose().compose([pose_shift, pose_rot, pose_shift2])
    pose_novel = Pose().compose([pose_oscil, pose_anchor.cpu()[None]])
    return pose_novel


def render_video(model_path, iteration, scene, gaussians, pipeline, background):
    render_path = os.path.join(
        model_path, "novel_view", "rgb")
    depth_path = os.path.join(
        model_path, "novel_view", "depth")
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    train_cameras = scene.getTrainCameras()
    train_pose = torch.stack([cam.pose_gt for cam in train_cameras])
    idx_center = (train_pose-train_pose.mean(dim=0, keepdim=True)
                  )[..., 3].norm(dim=-1).argmin()
    pose_novel = get_novel_view_poses(
        train_pose[idx_center], N=120, scale=1).cuda()
    view = train_cameras[0]
    for idx, pose in enumerate(pose_novel):
        view.pose = pose
        result = render(view, gaussians, pipeline, background)
        torchvision.utils.save_image(result["render"], os.path.join(
            render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(visualize_depth(result["depth"]), os.path.join(
            depth_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,
                  load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_video(dataset.model_path, iteration, scene,
                 gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.device)

    render_sets(model.extract(args), args.iteration,
                pipeline.extract(args))
