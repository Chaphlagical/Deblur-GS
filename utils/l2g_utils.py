import torch
import torch.nn.functional as torch_F

local_cam_num = 1024
embedding_dim = 128
skip_warp = [4]


class LocalWarp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # point-wise se3 prediction
        input_2D_dim = 2
        self.mlp_warp = torch.nn.ModuleList()
        layers = [None, 256, 256, 256, 256, 256, 256, 6]
        L = list(zip(layers[:-1], layers[1:]))
        for li, (k_in, k_out) in enumerate(L):
            if li == 0:
                k_in = input_2D_dim+embedding_dim
            if li in skip_warp:
                k_in += input_2D_dim+embedding_dim
            linear = torch.nn.Linear(k_in, k_out)
            self.mlp_warp.append(linear)

    def forward(self, uvf):
        feat = uvf
        for li, layer in enumerate(self.mlp_warp):
            if li in skip_warp:
                feat = torch.cat([feat, uvf], dim=-1)
            feat = layer(feat)
            if li != len(self.mlp_warp)-1:
                feat = torch_F.relu(feat)
        warp = feat
        return warp


class L2GContext(torch.nn.Module):
    def __init__(self, cameras):
        self.warp_embedding = torch.nn.Embedding(
            len(cameras), embedding_dim).cuda()
        self.warp_mlp = LocalWarp().cuda()
        poses = torch.stack([cam.pose for cam in cameras])
        self.global_poses = torch.nn.Embedding(
            len(cameras), 12, _weight=poses.view(-1, 12)).cuda()
        self.W = cameras[0].image_width
        self.H = cameras[0].image_height

    def gather_camera_cords_grid_3D(self):
        with torch.no_grad():
            # compute image coordinate grid
            y_range = torch.arange(
                self.H, dtype=torch.float32).cuda().add_(0.5)
            x_range = torch.arange(
                self.W, dtype=torch.float32).cuda().add_(0.5)
            Y, X = torch.meshgrid(y_range, x_range)  # [H,W]
            xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
            # compute grid_3D
            xy_grid = xy_grid.repeat(1, 1)  # [HW,2]
            grid_3D = img2cam(to_hom(xy_grid), intr)  # [B,HW,3]
            if ray_idx is not None:
                # consider only subset of rays
                grid_3D = torch.gather(
                    grid_3D, 1, ray_idx[..., None].expand(-1, -1, 3))
