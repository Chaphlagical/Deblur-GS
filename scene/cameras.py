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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from easydict import EasyDict as edict
from utils.pose_utils import Pose, Lie, interpolation_linear, interpolation_spline, interpolation_bezier, Quaternion
from utils.depth_utils import estimate_depth


class Camera(nn.Module):
    def __init__(self, midas, colmap_id, R, T, FoVx, FoVy, image, test_image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", control_pts_num=2, mode="Linear"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.is_train = True
        self.mode = mode

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.test_image = test_image.clamp(0.0, 1.0).to(self.data_device)
        self.predict_depth = estimate_depth(midas, self.original_image)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.pose = Pose()(R, T).cuda()
        self.pose_gt = Pose()(R, T).cuda()
        # self.pose_0 = torch.nn.Parameter(
        #     (torch.zeros(6)).cuda().requires_grad_(True))
        # self.pose_1 = torch.nn.Parameter(
        #     (torch.zeros(6)).cuda().requires_grad_(True))
        # self.pose_2 = torch.nn.Parameter(
        #     (torch.zeros(6)).cuda().requires_grad_(True))
        # self.pose_3 = torch.nn.Parameter(
        #     (torch.zeros(6)).cuda().requires_grad_(True))
        self.depth_factor = torch.nn.Parameter(
            torch.tensor([1, 0]).float().cuda().requires_grad_(True))

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()

        # self.trans_offset = torch.nn.Parameter(
        #     torch.zeros(6).cuda().requires_grad_(True))
        # self.scale_offset = torch.nn.Parameter(
        #     torch.ones(6).cuda().requires_grad_(True))
        # self.rot_offset = torch.nn.Parameter(
        #     torch.tensor([0, 0, 0, 1]*2).float().cuda().requires_grad_(True))

        lr_pose = 1.e-3
        lr_pose_end = 1.e-6
        max_iter = 90_000

        self.gaussian_trans = torch.nn.Parameter(
            (torch.zeros([control_pts_num, 6])).cuda().requires_grad_(True))

        self.pose_optimizer = torch.optim.Adam([
            {'params': [self.gaussian_trans],
             'lr': 1.e-3, "name": "translation offset"},
            # {'params': [self.gaussian_trans_2],
            #  'lr': 1e-3, "name": "translation offset"},
            # {'params': [self.scale_offset],
            #  'lr': 1e-3, "name": "scaling offset"},
            # {'params': [self.rot_offset],
            #  'lr': 1e-3, "name": "rotation offset"},
            # {'params': [self.pose_0],
            #  'lr': lr_pose, "name": "camera pose refine 0"},
            # {'params': [self.pose_1],
            #  'lr': lr_pose, "name": "camera pose refine 1"},
            # {'params': [self.pose_2],
            #  'lr': lr_pose, "name": "camera pose refine 2"},
            # {'params': [self.pose_3],
            #  'lr': lr_pose, "name": "camera pose refine 3"},
        ], lr=0.0, eps=1e-15)
        self.depth_optimizer = torch.optim.Adam([
            {'params': [self.depth_factor],
             'lr': 1e-3, "name": "depth factor"},
        ], lr=0.0, eps=1e-15)

        # self.pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.pose_optimizer, gamma=(lr_pose_end/lr_pose)**(1./max_iter))
        # self.depth_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.depth_optimizer, gamma=(1e-5/1e-3)**(1./max_iter))

    def update(self, global_step):
        self.pose_optimizer.step()
        self.depth_optimizer.step()
        self.pose_optimizer.zero_grad(set_to_none=True)
        self.depth_optimizer.zero_grad(set_to_none=True)
        decay_rate_pose = 0.01
        pose_lrate = 1e-3
        decay_rate = 0.1
        lrate_decay = 200
        decay_steps = lrate_decay * 1000
        new_lrate_pose = pose_lrate * \
            (decay_rate_pose ** (global_step / decay_steps))
        for param_group in self.pose_optimizer.param_groups:
            param_group['lr'] = new_lrate_pose
        for param_group in self.depth_optimizer.param_groups:
            param_group['lr'] = new_lrate_pose
        # self.pose_scheduler.step()
        # self.depth_scheduler.step()

    def get_gaussian_trans(self, alpha=0):
        if self.mode == "Linear":
            return interpolation_linear(
                self.gaussian_trans[0], self.gaussian_trans[1], alpha)
        elif self.mode == "Spline":
            return interpolation_spline(
                self.gaussian_trans[0], self.gaussian_trans[1], self.gaussian_trans[2], self.gaussian_trans[3], alpha)
        elif self.mode == "Bezier":
            return interpolation_bezier(
                self.gaussian_trans, alpha)
    # def get_trans_offset(self, alpha=0):
    #     offset_0, offset_1 = self.trans_offset.split([3, 3], dim=-1)
    #     return offset_0*(1.0-alpha)+offset_1*alpha

    # def get_scale_offset(self, alpha=0):
    #     offset_0, offset_1 = self.scale_offset.split([3, 3], dim=-1)
    #     return offset_0*(1.0-alpha)+offset_1*alpha

    # def get_rot_offset(self, alpha=0):
    #     offset_0, offset_1 = self.rot_offset.split([4, 4], dim=-1)
    #     q_tau_0 = Quaternion().q_to_Q(
    #         Quaternion().conjugate(offset_0)) @ offset_1
    #     r = alpha * Quaternion().log_q2r_parallel(q_tau_0.squeeze(-1))
    #     q_t_0 = Quaternion().exp_r2q_parallel(r)
    #     return Quaternion().q_to_Q(offset_0) @ q_t_0

    def get_train_pose(self, interp_alpha=0):
        return self.pose_gt
        # pose_refine = Lie().se3_to_SE3(self.pose_0)
        pose_refine = interpolation_linear(
            self.pose_0, self.pose_1, interp_alpha)
        # pose_refine = interpolation_spline(
        #     self.pose_0, self.pose_1, self.pose_2, self.pose_3, interp_alpha)
        return Pose().compose([pose_refine, self.pose])

    @torch.no_grad()
    def get_test_pose(self):
        pose_refine = interpolation_linear(
            self.pose_0, self.pose_1, 0.5)
        return Pose().compose([Lie().se3_to_SE3(pose_refine, self.pose)])

    def rescale_depth(self, depth):
        return depth*self.depth_factor[0]+self.depth_factor[1]

    def get_pose(self):
        if self.is_train:
            return self.get_train_pose()
        else:
            return self.pose

    @property
    def world_view_transform(self):
        pose = self.get_pose()
        V = torch.zeros((4, 4)).cuda()
        T = pose[:3, 3:].transpose(0, 1)
        R = pose[:3, :3]
        V[:3, :3] = R
        V[3, :3] = T
        V[3, 3] = 1.0
        return V

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    @property
    def intr(self):
        return torch.tensor([[fov2focal(self.FoVx, self.image_width), 0, self.image_width/2],
                             [0, fov2focal(
                                 self.FoVy, self.image_height), self.image_height/2],
                             [0, 0, 1]]).float().cuda()


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def world2cam(X, pose):  # [B,N,3]
    X_hom = to_hom(X)
    return X_hom @ pose.transpose(-1, -2)


def cam2img(X, cam_intr):
    return X @ cam_intr.transpose(-1, -2)


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom @ pose_inv.transpose(-1, -2)


def procrustes_analysis(X0, X1):  # [N,3]
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    assert (s0 != 0 and s1 != 0)
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)
    U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0:
        R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0], t1=t1[0], s0=s0, s1=s1, R=R)
    return sim3


@torch.no_grad()
def prealign_cameras(pose, pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1, 1, 3).cuda()
    center_pred = cam2world(center, pose)[:, 0]  # [N,3]
    center_GT = cam2world(center, pose_GT)[:, 0]  # [N,3]
    try:
        sim3 = procrustes_analysis(center_GT, center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0, t1=0, s0=1, s1=1,
                     R=torch.eye(3).cuda())
    # align the camera poses
    center_aligned = (center_pred-sim3.t1) / \
        sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose[..., :3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[..., None])[..., 0]
    pose_aligned = Pose()(R=R_aligned, t=t_aligned)
    return pose_aligned, sim3


@torch.no_grad()
def align_cameras(sim3, pose):
    center = torch.zeros(1, 1, 3).cuda()
    center = cam2world(center, pose)[:, 0]  # [N,3]
    center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1
    R_aligned = pose[..., :3]@sim3.R
    t_aligned = (-R_aligned@center_aligned[..., None])[..., 0]
    pose_aligned = Pose()(R=R_aligned, t=t_aligned)
    return pose_aligned


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = (
        ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
    )  # numerical stability near -1/+1
    return angle


@torch.no_grad()
def evaluate_camera_alignment(pose_aligned, pose_GT):
    # measure errors in rotation and translation
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned-t_GT)[..., 0].norm(dim=-1)
    error = edict(R=R_error, t=t_error)
    return error


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
