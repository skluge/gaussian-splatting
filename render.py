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
from utils.general_utils import safe_state
from utils.depth_utils import SurfaceNet
from utils.depth_utils import get_surface_normal_by_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from relighting.blinn_phong import BlinnPhongModel

def calculateNormalMap(depth, view):
    surfaceNet = SurfaceNet().cuda()
    surface = surfaceNet(depth, view.FoVx, view.FoVy)

    return surface

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, lighting_model = None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    albedo_path = os.path.join(model_path, name, "ours_{}".format(iteration), "albedo")

    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    normals_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
    normals_from_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals_from_depth")

    makedirs(render_path, exist_ok=True)
    makedirs(albedo_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normals_path, exist_ok=True)
    makedirs(normals_from_depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background, lighting_model = lighting_model)
        gt = view.original_image[0:3, :, :]

        rendering = results["render"]
        albedo = results["albedo"]
        depths = results["depths"]
        depths_vis = (depths - depths.min()) / (depths.max() - depths.min())
        #normals = results["normals"]
        #normals = torch.linalg.vector_norm(normals, dim=0)

        normals = get_surface_normal_by_depth(depths, view.FoVx, view.FoVy)
        surface_norm_viz = torch.mul(torch.add(normals, 1.0), 0.5)
        #surface_norm_viz = normals       

        print(normals.mean())

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(albedo, os.path.join(albedo_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depths_vis, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(surface_norm_viz, os.path.join(normals_from_depth_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(results["normals"], os.path.join(normals_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():       

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")        

        lighting_model = BlinnPhongModel()
        lighting_model.load_state_dict(torch.load(scene.model_path + "/lighting_model_{}.pth".format(iteration)))
        #print(lighting_model.light_position)
        #lighting_model.light_position.set_(torch.tensor([0.5, 0.5, 0.2], dtype=torch.float32, device="cuda"))

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, lighting_model)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, lighting_model)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)