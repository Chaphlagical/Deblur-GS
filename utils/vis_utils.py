import torch
from scene.cameras import cam2world
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@torch.no_grad()
def vis_cameras(vis, step, poses=[], colors=["blue", "magenta"], plot_dist=True, cam_depth=0.5):
    data = []
    # set up plots
    centers = []
    for pose, color in zip(poses, colors):
        pose = pose.detach().cpu()
        vertices, faces, wireframe = get_camera_mesh(
            pose, depth=cam_depth)
        center = vertices[:, -1]
        centers.append(center)
        # camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:, 0]],
            y=[float(n) for n in center[:, 1]],
            z=[float(n) for n in center[:, 2]],
            mode="markers",
            marker=dict(color=color, size=3),
        ))
        # colored camera mesh
        vertices_merged, faces_merged = merge_meshes(vertices, faces)
        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:, 0]],
            y=[float(n) for n in vertices_merged[:, 1]],
            z=[float(n) for n in vertices_merged[:, 2]],
            i=[int(n) for n in faces_merged[:, 0]],
            j=[int(n) for n in faces_merged[:, 1]],
            k=[int(n) for n in faces_merged[:, 2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))
        # camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        data.append(dict(
            type="scatter3d",
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode="lines",
            line=dict(color=color,),
            opacity=0.3,
        ))
    if plot_dist:
        # distance between two poses (camera centers)
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red", width=4,),
        ))
        if len(centers) == 4:
            center_merged = merge_centers(centers[2:4])
            data.append(dict(
                type="scatter3d",
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode="lines",
                line=dict(color="red", width=4,),
            ))
    # send data to visdom
    vis._send(dict(
        data=data,
        win="poses",
        layout=dict(
            title="({})".format(step),
            autosize=True,
            margin=dict(l=30, r=30, b=30, t=30,),
            showlegend=False,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
        opts=dict(title="poses ({})".format(step),),
    ))


def get_camera_mesh(pose, depth=1):
    vertices = torch.tensor([[-0.5, -0.5, 1],
                             [0.5, -0.5, 1],
                             [0.5, 0.5, 1],
                             [-0.5, 0.5, 1],
                             [0, 0, 0]])*depth
    faces = torch.tensor([[0, 1, 2],
                          [0, 2, 3],
                          [0, 1, 4],
                          [1, 2, 4],
                          [2, 3, 4],
                          [3, 0, 4]])
    vertices = cam2world(vertices[None], pose)
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe


def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]+[None]
    return wireframe_merged


def merge_meshes(vertices, faces):
    mesh_N, vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)], dim=0)
    vertices_merged = vertices.view(-1, vertices.shape[-1])
    return vertices_merged, faces_merged


def merge_centers(centers):
    center_merged = [[], [], []]
    for c1, c2 in zip(*centers):
        center_merged[0] += [float(c1[0]), float(c2[0]), None]
        center_merged[1] += [float(c1[1]), float(c2[1]), None]
        center_merged[2] += [float(c1[2]), float(c2[2]), None]
    return center_merged


def plot_save_poses(opt, fig, pose, pose_ref=None, path=None, ep=None):
    # get the camera meshes
    _, _, cam = get_camera_mesh(pose, depth=opt.visdom.cam_depth)
    cam = cam.numpy()
    if pose_ref is not None:
        _, _, cam_ref = get_camera_mesh(pose_ref, depth=opt.visdom.cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    plt.title("epoch {}".format(ep))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    setup_3D_plot(ax1, elev=-90, azim=-90,
                  lim=edict(x=(-1, 1), y=(-1, 1), z=(-1, 1)))
    setup_3D_plot(ax2, elev=0, azim=-90,
                  lim=edict(x=(-1, 1), y=(-1, 1), z=(-1, 1)))
    ax1.set_title("forward-facing view", pad=0)
    ax2.set_title("top-down view", pad=0)
    plt.subplots_adjust(left=0, right=1, bottom=0,
                        top=0.95, wspace=0, hspace=0)
    plt.margins(tight=True, x=0, y=0)
    # plot the cameras
    N = len(cam)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):
        if pose_ref is not None:
            ax1.plot(cam_ref[i, :, 0], cam_ref[i, :, 1],
                     cam_ref[i, :, 2], color=(0.3, 0.3, 0.3), linewidth=1)
            ax2.plot(cam_ref[i, :, 0], cam_ref[i, :, 1],
                     cam_ref[i, :, 2], color=(0.3, 0.3, 0.3), linewidth=1)
            ax1.scatter(cam_ref[i, 5, 0], cam_ref[i, 5, 1],
                        cam_ref[i, 5, 2], color=(0.3, 0.3, 0.3), s=40)
            ax2.scatter(cam_ref[i, 5, 0], cam_ref[i, 5, 1],
                        cam_ref[i, 5, 2], color=(0.3, 0.3, 0.3), s=40)
        c = np.array(color(float(i)/N))*0.8
        ax1.plot(cam[i, :, 0], cam[i, :, 1], cam[i, :, 2], color=c)
        ax2.plot(cam[i, :, 0], cam[i, :, 1], cam[i, :, 2], color=c)
        ax1.scatter(cam[i, 5, 0], cam[i, 5, 1], cam[i, 5, 2], color=c, s=40)
        ax2.scatter(cam[i, 5, 0], cam[i, 5, 1], cam[i, 5, 2], color=c, s=40)
    png_fname = "{}/{}.png".format(path, ep)
    plt.savefig(png_fname, dpi=75)
    # clean up
    plt.clf()


def plot_save_poses_blender(opt, fig, pose, pose_ref=None, path=None, ep=None):
    # get the camera meshes
    _, _, cam = get_camera_mesh(pose, depth=opt.visdom.cam_depth)
    cam = cam.numpy()
    if pose_ref is not None:
        _, _, cam_ref = get_camera_mesh(pose_ref, depth=opt.visdom.cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("epoch {}".format(ep), pad=0)
    setup_3D_plot(ax, elev=45, azim=35, lim=edict(
        x=(-3, 3), y=(-3, 3), z=(-3, 2.4)))
    plt.subplots_adjust(left=0, right=1, bottom=0,
                        top=0.95, wspace=0, hspace=0)
    plt.margins(tight=True, x=0, y=0)
    # plot the cameras
    N = len(cam)
    ref_color = (0.7, 0.2, 0.7)
    pred_color = (0, 0.6, 0.7)
    ax.add_collection3d(Poly3DCollection(
        [v[:4] for v in cam_ref], alpha=0.2, facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i, :, 0], cam_ref[i, :, 1],
                cam_ref[i, :, 2], color=ref_color, linewidth=0.5)
        ax.scatter(cam_ref[i, 5, 0], cam_ref[i, 5, 1],
                   cam_ref[i, 5, 2], color=ref_color, s=20)
    if ep == 0:
        png_fname = "{}/GT.png".format(path)
        plt.savefig(png_fname, dpi=75)
    ax.add_collection3d(Poly3DCollection(
        [v[:4] for v in cam], alpha=0.2, facecolor=pred_color))
    for i in range(N):
        ax.plot(cam[i, :, 0], cam[i, :, 1], cam[i, :, 2],
                color=pred_color, linewidth=1)
        ax.scatter(cam[i, 5, 0], cam[i, 5, 1],
                   cam[i, 5, 2], color=pred_color, s=20)
    for i in range(N):
        ax.plot([cam[i, 5, 0], cam_ref[i, 5, 0]],
                [cam[i, 5, 1], cam_ref[i, 5, 1]],
                [cam[i, 5, 2], cam_ref[i, 5, 2]], color=(1, 0, 0), linewidth=3)
    png_fname = "{}/{}.png".format(path, ep)
    plt.savefig(png_fname, dpi=75)
    # clean up
    plt.clf()


def setup_3D_plot(ax, elev, azim, lim=None):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.set_zlabel("Z", fontsize=16)
    ax.set_xlim(lim.x[0], lim.x[1])
    ax.set_ylim(lim.y[0], lim.y[1])
    ax.set_zlim(lim.z[0], lim.z[1])
    ax.view_init(elev=elev, azim=azim)
