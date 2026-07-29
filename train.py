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
import time
import uuid
from argparse import ArgumentParser, Namespace, SUPPRESS
from random import randint

from arguments import ModelParams, PipelineParams, OptimizationParams


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
    enable_polarization_alpha_loss = not getattr(
        opt, "disable_polarization_alpha_loss", False
    )
    enable_objectmark_filtering = not getattr(opt, "disable_objectmark_filtering", False)
    masks_required = (
        not opt.disable_mask_l1
        or (enable_polarization_alpha_loss and opt.lambda_pa > 0.0)
        or (enable_objectmark_filtering and opt.lambda_po > 0.0)
    )
    prepare_dataset_masks(dataset, require_masks=masks_required)
    training_start_time = time.time()
    objectmark_start_iter = max(0, int(getattr(opt, "objectmark_start_iter", 0)))
    objectmark_end_iter = max(0, int(getattr(opt, "objectmark_end_iter", opt.iterations)))
    initial_objectmark_active = object_mark_active_at_iteration(
        enable_objectmark_filtering,
        objectmark_start_iter,
        objectmark_end_iter,
        0,
    )
    gaussians = GaussianModel(dataset.sh_degree, use_objectmark=initial_objectmark_active)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.use_objectmark = object_mark_active_at_iteration(
            enable_objectmark_filtering,
            objectmark_start_iter,
            objectmark_end_iter,
            first_iter,
        )
        gaussians.restore(model_params, opt)

    if object_mark_active_at_iteration(
        enable_objectmark_filtering,
        objectmark_start_iter,
        objectmark_end_iter,
        first_iter,
    ):
        gaussians.enable_objectmark(opt)

    if enable_objectmark_filtering:
        if objectmark_end_iter <= objectmark_start_iter:
            print(
                "[INFO] ObjectMark Filtering active window is empty: "
                "start {}, end {}.".format(objectmark_start_iter, objectmark_end_iter)
            )
        elif gaussians.has_objectmark_score:
            print(
                "[INFO] ObjectMark Filtering active from iteration {} until "
                "before iteration {}.".format(
                    objectmark_start_iter, objectmark_end_iter
                )
            )
        else:
            print(
                "[INFO] ObjectMark Filtering delayed until iteration {} and "
                "ends before iteration {}.".format(
                    objectmark_start_iter, objectmark_end_iter
                )
            )
    else:
        print("[INFO] ObjectMark Filtering disabled for the full run.")

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
    lambda_pa = (
        getattr(opt, "lambda_pa", getattr(opt, "lambda_polarization", 0.0))
        if enable_polarization_alpha_loss
        else 0.0
    )
    lambda_po = getattr(opt, "lambda_po", 0.0) if enable_objectmark_filtering else 0.0
    train_log_interval = max(1, int(getattr(opt, "train_log_interval", 10)))
    objectmark_pruning_iterations = (
        {int(iter_i) for iter_i in getattr(opt, "objectmark_pruning_iterations", [])}
        if enable_objectmark_filtering
        else set()
    )
    vram_log_path = os.path.join(dataset.model_path, "train_vram.txt")
    gs_log_path = os.path.join(dataset.model_path, "gs.txt")
    objectmark_log_path = os.path.join(dataset.model_path, "ObjectMark.txt")
    max_num_gaussians = 0
    vram_log_file = open(vram_log_path, "w", buffering=1)
    vram_log_file.write("iter,vram_mb\n")
    gs_log_file = open(gs_log_path, "w", buffering=1)
    gs_log_file.write("iter,num_gaussians\n")
    objectmark_log_file = open(objectmark_log_path, "w", buffering=1)
    objectmark_log_file.write("iter,num_gaussians,{}\n".format(",".join(OBJECT_MARK_SCORE_BIN_LABELS)))

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    last_progress_iter = first_iter - 1
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(first_iter, opt.iterations + 1):

        should_update_logs = (
            iteration % train_log_interval == 0
            or iteration == opt.iterations
            or iteration in testing_iterations
        )
        should_time_iteration = tb_writer is not None and should_update_logs
        if should_time_iteration:
            iter_start.record()

        if (
            enable_objectmark_filtering
            and not gaussians.has_objectmark_score
            and object_mark_active_at_iteration(
                enable_objectmark_filtering,
                objectmark_start_iter,
                objectmark_end_iter,
                iteration,
            )
        ):
            gaussians.enable_objectmark(opt)
            print(
                "\n[ITER {}] ObjectMark Filtering activated: initialized O_j for {} gaussians and registered optimizer group.".format(
                    iteration,
                    gaussians.get_xyz.shape[0],
                )
            )
            if tb_writer is not None:
                tb_writer.add_scalar('objectmark/active', 1, iteration)

        if (
            enable_objectmark_filtering
            and gaussians.has_objectmark_score
            and not object_mark_active_at_iteration(
                enable_objectmark_filtering,
                objectmark_start_iter,
                objectmark_end_iter,
                iteration,
            )
        ):
            gaussians.disable_objectmark()
            print(
                "\n[ITER {}] ObjectMark Filtering ended: removed O_j, optimizer group, Lpo rendering/loss, and ObjectMark pruning.".format(
                    iteration
                )
            )
            if tb_writer is not None:
                tb_writer.add_scalar('objectmark/active', 0, iteration)

        objectmark_active = (
            enable_objectmark_filtering
            and gaussians.has_objectmark_score
            and object_mark_active_at_iteration(
                enable_objectmark_filtering,
                objectmark_start_iter,
                objectmark_end_iter,
                iteration,
            )
        )

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        ground_truth_mask, mask_denominator = camera_mask_on_device(
            viewpoint_cam,
            background.device,
            background.dtype,
            viewpoint_cam.original_image.shape[0],
        )

        needs_viewspace_grad = iteration < opt.densify_until_iter
        needs_pa = (
            enable_polarization_alpha_loss
            and ground_truth_mask is not None
            and lambda_pa > 0.0
        )
        needs_po = (
            objectmark_active
            and ground_truth_mask is not None
            and iteration >= opt.objectmark_guidance_from_iter
            and lambda_po > 0.0
        )
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        needs_normal = lambda_normal > 0.0
        needs_dist = lambda_dist > 0.0

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            need_viewspace_grad=needs_viewspace_grad,
            need_alpha=needs_pa or needs_normal,
            need_depth=needs_normal,
            need_normal=needs_normal,
            need_dist=needs_dist,
            need_objectmark=needs_po,
        )
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.to(background.device, non_blocking=True)

        if ground_truth_mask is not None and not opt.disable_mask_l1:
            rgb_reconstruction_loss = masked_l1_loss(
                image,
                gt_image,
                ground_truth_mask,
                mask_denominator,
            )
        else:
            rgb_reconstruction_loss = l1_loss(image, gt_image)
        loss = rgb_reconstruction_loss

        pa_loss = None
        if needs_pa:
            rendered_alpha = render_pkg["polarization_alpha"]
            pa_loss = alpha_polarization_loss(rendered_alpha, ground_truth_mask)
            loss = loss + lambda_pa * pa_loss

        po_loss = None
        if needs_po:
            po_loss = object_mark_polarization_loss(
                render_pkg["rend_object_mark"], ground_truth_mask
            )
            loss = loss + lambda_po * po_loss

        # regularization
        normal_loss = image.new_zeros(())
        if needs_normal:
            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]
            normal_loss = lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0)).mean()

        dist_loss = image.new_zeros(())
        if needs_dist:
            dist_loss = lambda_dist * render_pkg["rend_dist"].mean()

        # loss
        total_loss = loss + dist_loss + normal_loss

        total_loss.backward()

        if should_time_iteration:
            iter_end.record()

        with torch.no_grad():
            scalar_values = None
            if should_update_logs:
                scalar_names = [
                    "loss",
                    "rgb_reconstruction_loss",
                    "total_loss",
                    "dist_loss",
                    "normal_loss",
                ]
                scalar_tensors = [
                    loss.detach(),
                    rgb_reconstruction_loss.detach(),
                    total_loss.detach(),
                    dist_loss.detach(),
                    normal_loss.detach(),
                ]
                if pa_loss is not None:
                    scalar_names.append("pa_loss")
                    scalar_tensors.append(pa_loss.detach())
                if po_loss is not None:
                    scalar_names.append("po_loss")
                    scalar_tensors.append(po_loss.detach())
                    scalar_names.append("objectmark_mean")
                    scalar_tensors.append(gaussians.get_objectmark_score_prob.mean().detach())
                scalar_values = dict(zip(
                    scalar_names,
                    torch.stack([scalar.reshape(()) for scalar in scalar_tensors]).cpu().tolist(),
                ))

                ema_loss_for_log = 0.4 * scalar_values["loss"] + 0.6 * ema_loss_for_log
                if pa_loss is not None:
                    ema_pa_loss_for_log = 0.4 * scalar_values["pa_loss"] + 0.6 * ema_pa_loss_for_log
                if po_loss is not None:
                    ema_po_loss_for_log = 0.4 * scalar_values["po_loss"] + 0.6 * ema_po_loss_for_log
                ema_dist_for_log = 0.4 * scalar_values["dist_loss"] + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * scalar_values["normal_loss"] + 0.6 * ema_normal_for_log

                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                if pa_loss is not None:
                    loss_dict["Lpa"] = f"{ema_pa_loss_for_log:.{5}f}"
                if po_loss is not None:
                    loss_dict["Lpo"] = f"{ema_po_loss_for_log:.{5}f}"
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(iteration - last_progress_iter)
                last_progress_iter = iteration
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None and scalar_values is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', scalar_values["dist_loss"], iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', scalar_values["normal_loss"], iteration)
                if pa_loss is not None:
                    tb_writer.add_scalar('train_loss_patches/Lpa', scalar_values["pa_loss"], iteration)
                if po_loss is not None:
                    tb_writer.add_scalar('train_loss_patches/Lpo', scalar_values["po_loss"], iteration)
                    tb_writer.add_scalar('objectmark/mean_score', scalar_values["objectmark_mean"], iteration)

            if (tb_writer is not None and scalar_values is not None) or iteration in testing_iterations:
                elapsed = iter_start.elapsed_time(iter_end) if should_time_iteration else None
                training_report(
                    tb_writer,
                    iteration,
                    rgb_reconstruction_loss,
                    total_loss,
                    l1_loss,
                    elapsed,
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                    scalar_values,
                    enable_polarization_alpha_loss,
                    objectmark_active,
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

            if objectmark_active and iteration in objectmark_pruning_iterations:
                n_pruned = prune_gaussians_by_object_mark(
                    gaussians, opt.objectmark_pruning_threshold
                )
                if tb_writer is not None:
                    tb_writer.add_scalar('objectmark/pruned_gaussians', n_pruned, iteration)
                if n_pruned > 0:
                    print("\n[ITER {}] ObjectMark Pruning removed {} gaussians".format(iteration, n_pruned))

            current_num_gaussians = int(gaussians.get_xyz.shape[0])
            max_num_gaussians = max(max_num_gaussians, current_num_gaussians)
            if iteration % 500 == 0:
                current_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                objectmark_bin_counts = object_mark_score_bin_counts(gaussians)
                vram_log_file.write(f"{iteration},{current_vram_mb:.2f}\n")
                gs_log_file.write(f"{iteration},{current_num_gaussians}\n")
                objectmark_log_file.write("{},{},{}\n".format(
                    iteration,
                    current_num_gaussians,
                    ",".join(str(count) for count in objectmark_bin_counts),
                ))

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
                            custom_cam,
                            gaussians,
                            pipe,
                            background,
                            scaling_modifier,
                            need_viewspace_grad=False,
                            **get_render_output_requirements(dataset.render_items, render_mode),
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
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                # Network input is an external process boundary; a malformed
                # or closed connection should not terminate optimization.
                except Exception:
                    network_gui.conn = None

    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    vram_log_file.write(f"max_vram_mb,{peak_allocated_mb:.2f}\n")
    vram_log_file.close()
    gs_log_file.write(f"max_num_gaussians,{max_num_gaussians}\n")
    gs_log_file.close()
    objectmark_log_file.close()

    total_training_time = time.time() - training_start_time
    with open(os.path.join(dataset.model_path, "time.txt"), "w") as time_file:
        time_file.write(f"{total_training_time:.6f}\n")

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

def cli_arg_was_provided(flag):
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])

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
    scalar_values=None,
    enable_polarization_alpha_loss=True,
    enable_objectmark_filtering=True,
):
    if tb_writer and scalar_values is not None:
        tb_writer.add_scalar(
            'train_loss_patches/rgb_reconstruction_loss',
            scalar_values["rgb_reconstruction_loss"],
            iteration,
        )
        tb_writer.add_scalar('train_loss_patches/total_loss', scalar_values["total_loss"], iteration)
        if elapsed is not None:
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
                    render_pkg = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        *renderArgs,
                        need_viewspace_grad=False,
                        need_alpha=should_log_aux,
                        need_depth=should_log_aux,
                        need_normal=should_log_aux,
                        need_dist=should_log_aux,
                        need_objectmark=should_log_aux and enable_objectmark_filtering,
                    )
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
                            if "polarization_alpha" in render_pkg:
                                polarization_alpha = render_pkg['polarization_alpha']
                                tb_writer.add_images(config['name'] + "_view_{}/polarization_alpha".format(viewpoint.image_name), polarization_alpha[None], global_step=iteration)
                            if enable_objectmark_filtering and "objectmark_response" in render_pkg:
                                tb_writer.add_images(config['name'] + "_view_{}/objectmark_response".format(viewpoint.image_name), render_pkg["objectmark_response"][None], global_step=iteration)

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
    parser.add_argument(
        "--objectmark_pruning_iterations",
        "--objectmark-pruning-iterations",
        nargs="+",
        type=int,
        default=[],
        help="Explicit ObjectMark pruning iterations; defaults to 2500:500:29500.",
    )
    parser.add_argument(
        "--objectmark-start-iter",
        dest="objectmark_start_iter",
        type=int,
        default=SUPPRESS,
        help="Hyphenated alias for --objectmark_start_iter.",
    )
    parser.add_argument(
        "--objectmark-end-iter",
        dest="objectmark_end_iter",
        type=int,
        default=SUPPRESS,
        help="Hyphenated alias for --objectmark_end_iter.",
    )
    parser.add_argument(
        "--objectmark-pruning-threshold",
        dest="objectmark_pruning_threshold",
        type=float,
        default=SUPPRESS,
        help="Hyphenated alias for --objectmark_pruning_threshold.",
    )
    parser.add_argument(
        "--disable-polarization-alpha-loss",
        dest="disable_polarization_alpha_loss",
        action="store_true",
        help="Hyphenated alias for --disable_polarization_alpha_loss.",
    )
    parser.add_argument(
        "--disable-objectmark-filtering",
        dest="disable_objectmark_filtering",
        action="store_true",
        help="Hyphenated alias for --disable_objectmark_filtering.",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    import torch
    from tqdm import tqdm

    from gaussian_renderer import network_gui
    from object_aware.losses import (
        alpha_polarization_loss,
        masked_l1_loss,
        object_mark_polarization_loss,
    )
    from object_aware.masking import camera_mask_on_device, prepare_dataset_masks
    from object_aware.pruning import (
        OBJECT_MARK_SCORE_BIN_LABELS,
        object_mark_score_bin_counts,
        prune_gaussians_by_object_mark,
    )
    from object_aware.schedules import (
        default_object_mark_pruning_iterations,
        object_mark_active_at_iteration,
    )
    from utils.general_utils import safe_state
    from utils.image_utils import get_render_output_requirements, psnr, render_net_image
    from utils.loss_utils import l1_loss

    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_FOUND = True
    except ImportError:
        TENSORBOARD_FOUND = False

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    opt.objectmark_start_iter = max(0, int(getattr(opt, "objectmark_start_iter", 0)))
    opt.objectmark_end_iter = max(0, int(getattr(opt, "objectmark_end_iter", 30_000)))
    if cli_arg_was_provided("--objectmark_pruning_iterations") or cli_arg_was_provided("--objectmark-pruning-iterations"):
        opt.objectmark_pruning_iterations = args.objectmark_pruning_iterations
    else:
        opt.objectmark_pruning_iterations = default_object_mark_pruning_iterations(
            opt.objectmark_end_iter,
            opt.iterations,
        )
    if not opt.disable_objectmark_filtering and not 0.0 <= opt.objectmark_pruning_threshold <= 1.0:
        raise ValueError("--objectmark-pruning-threshold must be in [0, 1]")
    if cli_arg_was_provided("--lambda_polarization") and not cli_arg_was_provided("--lambda_pa"):
        opt.lambda_pa = opt.lambda_polarization
    if (
        not cli_arg_was_provided("--lambda_po")
        and (
            cli_arg_was_provided("--lambda_objectmark_foreground")
            or cli_arg_was_provided("--lambda_objectmark_background")
        )
    ):
        opt.lambda_po = opt.lambda_objectmark_foreground + opt.lambda_objectmark_background
    if opt.disable_polarization_alpha_loss:
        opt.lambda_pa = 0.0
    if opt.disable_objectmark_filtering:
        opt.lambda_po = 0.0
        opt.objectmark_pruning_iterations = []
    print("[INFO] Ablation settings:")
    print("       Polarization Alpha Loss: {}".format("disabled" if opt.disable_polarization_alpha_loss else "enabled"))
    print("       ObjectMark Filtering: {}".format("disabled" if opt.disable_objectmark_filtering else "enabled"))
    print("       ObjectMark Start Iter: {}".format("ignored (disabled)" if opt.disable_objectmark_filtering else opt.objectmark_start_iter))
    print("       ObjectMark End Iter: {}".format("ignored (disabled)" if opt.disable_objectmark_filtering else opt.objectmark_end_iter))
    print("       ObjectMark Pruning Iterations: {}".format("ignored (disabled)" if opt.disable_objectmark_filtering else opt.objectmark_pruning_iterations))
    print("       ObjectMark Pruning Threshold: {}".format("ignored (disabled)" if opt.disable_objectmark_filtering else opt.objectmark_pruning_threshold))
    if opt.disable_objectmark_filtering:
        print("       ObjectMark branch removed: no O_j parameter, optimizer group, O_i render, Lpo, pruning, or ObjectMark propagation.")
    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
