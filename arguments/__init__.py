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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        help_texts = vars(self).get("_help", {})
        for key, value in vars(self).items():
            if key == "_help":
                continue
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true", help=help_texts.get(key))
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t, help=help_texts.get(key))
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true", help=help_texts.get(key))
                else:
                    group.add_argument("--" + key, default=value, type=t, help=help_texts.get(key))

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        self._help = {
            "sh_degree": "Maximum spherical-harmonics degree.",
            "source_path": "COLMAP or Blender scene directory.",
            "model_path": "Directory for checkpoints, renders, and meshes.",
            "images": "RGB image directory relative to the scene root.",
            "resolution": "Input downsampling factor; -1 auto-resizes widths above 1600 px.",
            "white_background": "Use a white rendering background.",
            "data_device": "Device used for camera image data.",
            "eval": "Hold out evaluation views using the inherited dataset split.",
            "render_items": "Viewer output modes retained for network GUI compatibility.",
        }
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        self._help = {
            "convert_SHs_python": "Evaluate spherical harmonics in Python.",
            "compute_cov3D_python": "Compute Gaussian covariance in Python.",
            "depth_ratio": "Blend expected depth (0) and median depth (1) for meshing.",
            "debug": "Enable rasterizer debug behavior.",
        }
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self._help = {
            "iterations": "Total optimization iterations.",
            "position_lr_init": "Initial Gaussian-position learning rate.",
            "position_lr_final": "Final Gaussian-position learning rate.",
            "position_lr_delay_mult": "Position learning-rate delay multiplier.",
            "position_lr_max_steps": "Steps used by the position learning-rate schedule.",
            "feature_lr": "Spherical-harmonics feature learning rate.",
            "opacity_lr": "Gaussian opacity learning rate.",
            "scaling_lr": "Gaussian scale learning rate.",
            "rotation_lr": "Gaussian rotation learning rate.",
            "percent_dense": "Scene-extent fraction used by densification.",
            "lambda_dssim": "Retained for 2DGS config compatibility; masked training does not use SSIM.",
            "lambda_dist": "Weight of inherited 2DGS depth-distortion regularization.",
            "lambda_normal": "Weight of inherited 2DGS normal-consistency regularization.",
            "opacity_cull": "Opacity threshold used by inherited 2DGS densification pruning.",
            "densification_interval": "Iterations between densification passes.",
            "opacity_reset_interval": "Iterations between opacity resets.",
            "densify_from_iter": "First iteration after which densification may run.",
            "densify_until_iter": "Exclusive iteration that stops densification.",
            "densify_grad_threshold": "View-space gradient threshold for densification.",
        }
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath, encoding="utf-8") as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (FileNotFoundError, TypeError):
        print("Config file not found; using command-line values.")
    args_cfgfile = eval(
        cfgfile_string,
        {"Namespace": Namespace, "__builtins__": {}},
        {},
    )

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
