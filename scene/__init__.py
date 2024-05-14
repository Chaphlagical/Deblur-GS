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

import os
import random
import json
from utils.system_utils import searchForMaxIteration, mkdir_p
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.cameras import Pose, world2cam, cam2img
import torch.nn.functional as torch_F
import torch
import numpy as np
from utils.general_utils import visualize_depth
import torchvision
from plyfile import PlyData, PlyElement


class Scene:

    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            # Multi-res consistent random shuffling
            random.shuffle(scene_info.train_cameras)
            # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # midas = torch.hub.load(
        #     "/home/cwb/.cache/torch/hub/intel-isl_MiDaS_master", "DPT_Hybrid", source="local")
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        midas.cuda()
        midas.eval()
        for param in midas.parameters():
            param.requires_grad = False

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                midas, scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                midas, scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" +
                                                 str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            #             # Generate sparse depth map
            #             #
            #             points = torch.tensor(scene_info.point_cloud.points).cuda()
            #             points = torch.cat(
            #                 [points, torch.ones_like(points[..., :1])], dim=-1)
            #             for cam in self.train_cameras[resolution_scale]:
            #                 img = torch.zeros(
            #                     (cam.image_height, cam.image_width)).float().cuda()
            #                 cam_pt = points@cam.full_proj_transform
            #                 cam_pt = cam_pt[..., :3] / \
            #                     (cam_pt[..., 3]+0.0000001).unsqueeze(-1)
            #                 depth = (points@cam.world_view_transform)[..., 2]
            #                 X = (((cam_pt[..., 0]+1.0)*cam.image_width-1.0)*0.5).int()
            #                 Y = (((cam_pt[..., 1]+1.0)*cam.image_height-1.0)*0.5).int()
            #                 img_pt = torch.stack([X, Y, 1.0/depth], dim=-1)
            #                 img_pt = img_pt[img_pt[..., 0] > 0]
            #                 img_pt = img_pt[img_pt[..., 0] < cam.image_width]
            #                 img_pt = img_pt[img_pt[..., 1] > 0]
            #                 img_pt = img_pt[img_pt[..., 1] < cam.image_height]

            #                 predict_depth = cam.predict_depth[img_pt[...,
            #                                                          1].int(), img_pt[..., 0].int()].cpu().numpy()
            #                 img_depth = img_pt[..., 2].cpu().numpy()
            #                 A = np.vstack([predict_depth, np.ones(len(predict_depth))]).T
            #                 m, c = np.linalg.lstsq(A, img_depth, rcond=None)[0]
            #                 predict_depth = 1/(m*cam.predict_depth+c)
            #                 # print(1/(m*predict_depth+c))
            #                 # print(1/img_depth)

            #                 y_range = torch.arange(
            #                     cam.image_height, dtype=torch.float32).cuda().add_(0.5)
            #                 x_range = torch.arange(
            #                     cam.image_width, dtype=torch.float32).cuda().add_(0.5)
            #                 Y, X = torch.meshgrid(y_range, x_range)
            #                 screen_pos = torch.stack(
            #                     [(X/cam.image_width)*2-1, (Y/cam.image_height)*2-1, predict_depth], dim=-1)
            #                 # print(torch.cat(
            #                 #     [points[..., :2]*points[..., 2].unsqueeze(-1), points[..., 2].unsqueeze(-1), torch.ones_like(points[..., :1])], dim=-1).shape)
            #                 # print(cam.full_proj_transform.inverse().shape)
            #                 # print(screen_pos)
            #                 # print(points)
            #                 # print(screen_pos[..., :2].shape)
            #                 # print((screen_pos[..., :2] *
            #                 #       screen_pos[..., 2].unsqueeze(-1)).shape)
            #                 # print(torch.cat([screen_pos[..., :2]*screen_pos[...,
            #                 #       2].unsqueeze(-1), screen_pos[..., 2]], dim=-1))
            #                 HomogeneousWorldPosition = torch.cat(
            #                     [screen_pos[..., :2]*screen_pos[..., 2].unsqueeze(-1), screen_pos[..., 2].unsqueeze(-1), torch.ones_like(screen_pos[..., :1])], dim=-1)  # @ cam.full_proj_transform.inverse()
            #                 AbsoluteWorldPos = HomogeneousWorldPosition[..., :3]
            #                 AbsoluteWorldPos[..., :2] = HomogeneousWorldPosition[...,
            #                                                                      :2]/HomogeneousWorldPosition[..., 3].unsqueeze(-1)
            #                 print(AbsoluteWorldPos.shape)
            #                 # mkdir_p(os.path.dirname("./test.ply"))
            #                 AbsoluteWorldPos = AbsoluteWorldPos.reshape(-1, 3)
            #                 xyz = AbsoluteWorldPos.cpu().numpy()
            #                 # normals = np.zeros_like(xyz)
            #                 # f_dc = self._features_dc.detach().transpose(
            #                 #     1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            #                 # f_rest = self._features_rest.detach().transpose(
            #                 #     1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            #                 # opacities = self._opacity.detach().cpu().numpy()
            #                 # scale = self._scaling.detach().cpu().numpy()
            #                 # rotation = self._rotation.detach().cpu().numpy()

            #                 dtype_full = [(attribute, 'f4')
            #                               for attribute in ['x', 'y', 'z']]

            #                 elements = np.empty(xyz.shape[0], dtype=dtype_full)
            #                 attributes = xyz
            #                 elements[:] = list(map(tuple, attributes))
            #                 el = PlyElement.describe(elements, 'vertex')
            #                 PlyData([el]).write("test.ply")
            # # float4 HomogeneousWorldPosition = mul(float4(ScreenPos * LinearDepth, LinearDepth, 1), View.ScreenToWorld);
            # # AbsoluteWorldPos = HomogeneousWorldPosition.xyz / HomogeneousWorldPosition.w;
            #                 # torchvision.utils.save_image((
            #                 #     cam.predict_depth*m+c), "depth_align.png")
            #                 # torchvision.utils.save_image((
            #                 #     cam.predict_depth), "depth_predict.png")
            #                 # print(m)
            #                 # print(c)
            #                 exit(0)
            #                 print(depth)
            #                 for idx in range(len(depth)):
            #                     if X[idx] > 0 and X[idx] < cam.image_width and Y[idx] > 0 and Y[idx] < cam.image_height:
            #                         img[int(Y[idx]), int(X[idx])] = 1/depth[idx]
            #                 depth = visualize_depth(img)
            #                 # print(img)
            #                 torchvision.utils.save_image(depth, "depth_sparse.png")
            #                 torchvision.utils.save_image(visualize_depth(
            #                     cam.predict_depth), "depth_dense.png")
            #                 exit(0)
            #                 # print(img)
            #             # print(scene_info.train_cameras)
            #             # print(points)
            #             exit(0)

            self.gaussians.create_from_pcd(
                scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(
            point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
