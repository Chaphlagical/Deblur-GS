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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_depth_loss
from utils.vis_utils import vis_cameras
from utils import depth_utils
from torchmetrics.functional.regression import pearson_corrcoef
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, visualize_depth, check_socket_open
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import Lie, evaluate_camera_alignment, prealign_cameras, align_cameras
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from lpipsPyTorch import lpips
import visdom
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0

    if not opt.deblur:
        opt.blur_sample_num = 1

    # init tensorboard
    tb_writer = prepare_output_and_logger(dataset)

    # init visdom
    is_open = check_socket_open(dataset.visdom_server, dataset.visdom_port)
    retry = None
    while not is_open:
        retry = input(
            "visdom port ({}:{}) not open, retry? (y/n) ".format(dataset.visdom_server, dataset.visdom_port))
        if retry not in ["y", "n"]:
            continue
        if retry == "y":
            is_open = check_socket_open(
                dataset.visdom_server, dataset.visdom_port)
        else:
            break
    vis = visdom.Visdom(
        server=dataset.visdom_server, port=dataset.visdom_port)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    blur_blend_embedding = torch.nn.Embedding(
        len(scene.getTrainCameras()), opt.blur_sample_num).cuda()
    blur_blend_embedding.weight = torch.nn.Parameter(torch.ones(
        len(scene.getTrainCameras()), opt.blur_sample_num).cuda())
    optimizer = torch.optim.Adam([
        {'params': blur_blend_embedding.parameters(),
         'lr': 1e-3, "name": "blur blend parameters"},
    ], lr=0.0, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(1e-6/1e-3)**(1./opt.iterations))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations),
                        desc="Training progress")
    first_iter += 1

    # train_cameras = scene.getTrainCameras()
    # test_cameras = scene.getTestCameras()
    # train_pose = torch.stack([cam.get_pose() for cam in train_cameras])
    # train_pose_GT = torch.stack([cam.pose_gt for cam in train_cameras])
    # test_pose_gt = torch.stack([cam.pose_gt for cam in test_cameras])
    # aligned_train_pose, sim3 = prealign_cameras(train_pose, train_pose_GT)
    # aligned_test_pose = align_cameras(sim3, test_pose_gt)

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(
                        net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % opt.sh_up_degree_interval == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        blur_weight = blur_blend_embedding(
            torch.tensor(viewpoint_cam.uid).cuda())
        blur_weight /= torch.sum(blur_weight)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand(
            (3), device="cuda") if opt.random_background else background

        image = 0
        depth = 0
        radii = None
        viewspace_point_tensors = []
        viewspace_point_tensor_data = 0
        visibility_filter = None
        per_rgb_loss = 0

        # Loss
        if opt.ground_truth:
            gt_image = viewpoint_cam.test_image.cuda()
        else:
            gt_image = viewpoint_cam.original_image.cuda()
        predict_depth = viewpoint_cam.predict_depth.cuda()

        if not opt.non_uniform:
            blur_weight = 1.0/opt.blur_sample_num

        for idx in range(opt.blur_sample_num):
            alpha = idx / (max(1, opt.blur_sample_num-1))
            render_pkg = render(viewpoint_cam, gaussians,
                                pipe, bg, interp_alpha=alpha)
            image_, depth_, viewspace_point_tensor_, visibility_filter_, radii_ = render_pkg["render"], render_pkg[
                "depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            image += image_*blur_weight
            depth += depth_*blur_weight
            radii = radii_ if radii is None else torch.max(radii_, radii)
            per_rgb_loss += l1_loss(image_, gt_image)*blur_weight
            visibility_filter = visibility_filter_ if visibility_filter is None else torch.logical_or(
                visibility_filter, visibility_filter_)
            viewspace_point_tensors.append(viewspace_point_tensor_)
            viewspace_point_tensor_data += viewspace_point_tensor_ * \
                blur_weight

        image_loss = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * image_loss + \
            opt.lambda_dssim * (1.0 - ssim(image, gt_image))  # + \

        if opt.depth_reg:
            loss += opt.depth_weight * min(
                (1 - pearson_corrcoef(- predict_depth, depth)),
                (1 - pearson_corrcoef(1 / (predict_depth + 200.), depth))
            )

        loss.backward()

        viewspace_point_tensor = viewspace_point_tensor_data.clone().detach().requires_grad_(True)
        viewspace_point_tensor.grad = None
        for viewspace_point_tensor_ in viewspace_point_tensors:
            if viewspace_point_tensor.grad is None:
                viewspace_point_tensor.grad = viewspace_point_tensor_.grad
            else:
                viewspace_point_tensor.grad = torch.max(
                    viewspace_point_tensor.grad, viewspace_point_tensor_.grad)
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:
                print(f"blur_weight={blur_weight}")
            # Log and save
            training_report(tb_writer, vis, iteration, image_loss, loss, l1_loss, iter_start.elapsed_time(
                iter_end), testing_iterations, scene, pipe, background, opt)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if opt.deblur:
                    viewpoint_cam.update(iteration)
                #     viewpoint_cam.pose_optimizer.step()
                #     viewpoint_cam.pose_scheduler.step()
                #     viewpoint_cam.pose_optimizer.zero_grad(set_to_none=True)

                # if opt.depth_reg:
                #     viewpoint_cam.depth_optimizer.step()
                #     viewpoint_cam.depth_scheduler.step()
                #     viewpoint_cam.depth_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # if iteration % 2 == 0:
            #     train_cameras = scene.getTrainCameras()
            #     test_cameras = scene.getTestCameras()
            #     train_pose = torch.stack([cam.get_pose()
            #                              for cam in train_cameras])
            #     train_pose_GT = torch.stack(
            #         [cam.pose_gt for cam in train_cameras])
            #     test_pose_gt = torch.stack(
            #         [cam.pose_gt for cam in test_cameras])
            #     aligned_train_pose, sim3 = prealign_cameras(
            #         train_pose, train_pose_GT)

                # vis_cameras(vis, step=iteration, poses=[
                #             aligned_train_pose, train_pose_GT])


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, vis, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, pipe, background, opt):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss',
                             Ll1.item(), iteration)
        tb_writer.add_scalar(
            'train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        train_pose = torch.stack([cam.get_pose() for cam in train_cameras])
        train_pose_GT = torch.stack([cam.pose_gt for cam in train_cameras])
        test_pose_gt = torch.stack([cam.pose_gt for cam in test_cameras])
        aligned_train_pose, sim3 = prealign_cameras(train_pose, train_pose_GT)
        aligned_test_pose = align_cameras(sim3, test_pose_gt)

        vis_cameras(vis, step=iteration, poses=[
                    aligned_train_pose, train_pose_GT])

        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                rgb_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                depth_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name'] == 'test':
                        render_result = render(
                            viewpoint, scene.gaussians, pipe, background, mode="test")
                    else:
                        render_result = render(
                            viewpoint, scene.gaussians, pipe, background, mode="test")
                        print(
                            f"{idx} gaussian_trans={viewpoint.gaussian_trans}")

                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(
                        viewpoint.test_image.cuda(), 0.0, 1.0)
                    ref_image = torch.clamp(
                        viewpoint.original_image.cuda(), 0.0, 1.0)
                    predict_depth = viewpoint.rescale_depth(
                        viewpoint.predict_depth.cuda())
                    vis_predict_depth = visualize_depth(predict_depth)
                    vis_depth = visualize_depth(render_result["depth"])
                    depth_loss = compute_depth_loss(
                        predict_depth, render_result["depth"])
                    if tb_writer:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(
                            viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(
                            viewpoint.image_name), vis_depth[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/predict_depth".format(
                                viewpoint.image_name), vis_predict_depth[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(
                                viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/reference".format(
                                viewpoint.image_name), ref_image[None], global_step=iteration)
                    rgb_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image,
                                        net_type='vgg').mean().double()
                    depth_test += depth_loss.mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                rgb_test /= len(config['cameras'])
                depth_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                    iteration, config['name'], rgb_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - image_loss', rgb_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - depth_loss', depth_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    if config['name'] == 'train':
                        cam_error = evaluate_camera_alignment(
                            aligned_train_pose, train_pose_GT)
                        tb_writer.add_scalar(
                            config['name'] + '/loss_viewpoint - R_error', cam_error.R.mean(), iteration)
                        tb_writer.add_scalar(
                            config['name'] + '/loss_viewpoint - t_error', cam_error.t.mean(), iteration)

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(
                'total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+",
                        type=int, default=list(range(1000, 90_000, 1000)))
    parser.add_argument("--save_iterations", nargs="+",
                        type=int, default=list(range(1000, 90_000, 1000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations",
                        nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.device)

    print(torch.cuda.current_device())

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
