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
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from utils.image_utils import psnr, render_net_image
from utils.loss_utils import l1_loss, masked_l1_loss, polarization_loss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

LAMBDA_ALPHA_POLARIZATION = 0.1
LAMBDA_OBJECTMARK_POLARIZATION = 1.0
OBJECTMARK_PRUNE_FROM_ITER = 2_500
OBJECTMARK_PRUNE_UNTIL_ITER = 30_000
OBJECTMARK_PRUNE_INTERVAL = 500
OBJECTMARK_PRUNE_THRESHOLD = 0.5


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    from gaussian_renderer import network_gui, render
    from scene import GaussianModel, Scene

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    dataset.require_masks = True
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_pa_loss_for_log = 0.0
    ema_po_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            need_objectmark=True,
        )
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.to(background.device, non_blocking=True)
        ground_truth_mask = viewpoint_cam.gt_alpha_mask.to(
            background.device,
            dtype=background.dtype,
            non_blocking=True,
        )
        rgb_reconstruction_loss = masked_l1_loss(image, gt_image, ground_truth_mask)

        pa_loss = polarization_loss(render_pkg["rend_alpha"], ground_truth_mask)
        po_loss = polarization_loss(render_pkg["rend_object_mark"], ground_truth_mask)
        loss = (
            rgb_reconstruction_loss
            + LAMBDA_ALPHA_POLARIZATION * pa_loss
            + LAMBDA_OBJECTMARK_POLARIZATION * po_loss
        )

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        normal_loss = lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0)).mean()
        dist_loss = lambda_dist * render_pkg["rend_dist"].mean()

        # loss
        total_loss = loss + dist_loss + normal_loss

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_pa_loss_for_log = 0.4 * pa_loss.item() + 0.6 * ema_pa_loss_for_log
            ema_po_loss_for_log = 0.4 * po_loss.item() + 0.6 * ema_po_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Lpa": f"{ema_pa_loss_for_log:.{5}f}",
                    "Lpo": f"{ema_po_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', dist_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/Lpa', pa_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/Lpo', po_loss.item(), iteration)

            training_report(
                tb_writer,
                iteration,
                rgb_reconstruction_loss,
                total_loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if (
                OBJECTMARK_PRUNE_FROM_ITER <= iteration < OBJECTMARK_PRUNE_UNTIL_ITER
                and iteration % OBJECTMARK_PRUNE_INTERVAL == 0
            ):
                n_pruned = gaussians.prune_background_by_objectmark_score(
                    OBJECTMARK_PRUNE_THRESHOLD
                )
                if tb_writer is not None:
                    tb_writer.add_scalar('objectmark/pruned_gaussians', n_pruned, iteration)
                if n_pruned > 0:
                    print("\n[ITER {}] ObjectMark Pruning removed {} gaussians".format(iteration, n_pruned))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    (
                        custom_cam,
                        do_training,
                        keep_alive,
                        scaling_modifier,
                        render_mode,
                    ) = network_gui.receive()
                    if custom_cam is not None:
                        render_pkg = render(
                            custom_cam, gaussians, pipe, background, scaling_modifier
                        )
                        net_image = render_net_image(
                            render_pkg, dataset.render_items, render_mode, custom_cam
                        )
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, min=0, max=1.0) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                # Network input is an external process boundary; a malformed
                # or closed connection should not terminate optimization.
                except Exception:
                    network_gui.conn = None


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer,
    iteration,
    rgb_reconstruction_loss,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar(
            'train_loss_patches/rgb_reconstruction_loss',
            rgb_reconstruction_loss.item(),
            iteration,
        )
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    should_log_aux = tb_writer and (idx < 5)
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(image.device, non_blocking=True), 0.0, 1.0)
                    if should_log_aux:
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), render_pkg["rend_alpha"][None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except (KeyError, RuntimeError, ValueError) as exc:
                            print(
                                "[WARN] Skipping auxiliary validation images for "
                                f"{viewpoint.image_name}: {exc}"
                            )

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(
        description="Train mask-guided object-aware 2D Gaussian Splatting."
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1", help="Network GUI bind address.")
    parser.add_argument('--port', type=int, default=6009, help="Network GUI port.")
    parser.add_argument(
        '--detect_anomaly',
        action='store_true',
        default=False,
        help="Enable PyTorch autograd anomaly detection.",
    )
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30_000],
        help="Iterations at which to evaluate held-out and sample training views.",
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[7_000, 30_000],
        help="Iterations at which to save point-cloud PLY files.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress RNG initialization output.")
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[],
        help="Iterations at which to save resumable .pth checkpoints.",
    )
    parser.add_argument(
        "--start_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint from which to resume training.",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    from gaussian_renderer import network_gui

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("\nTraining complete.")
