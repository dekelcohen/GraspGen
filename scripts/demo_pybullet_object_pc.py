# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Modifications for PyBullet RGB-D Unprojection

import argparse
import os
import numpy as np
import torch
import trimesh.transformations as tra
from PIL import Image

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal_with_color


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a single object point cloud extracted from PyBullet RGB-D"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"./data/open_door_pybullet",
        help="Directory containing PyBullet RGB, Depth, Seg and Camera matrices",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        required=True,
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.8,
        help="Threshold for valid grasps. If -1.0, then top N will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return when return_topk is True",
    )
    parser.add_argument(
        "--target_seg_id",
        type=int,
        default=1,
        help="The integer ID of the target object in the segmentation mask",
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=0.05,
        help="Distance threshold for KNN outlier removal. Increase if the object gets falsely removed.",
    )

    return parser.parse_args()


def load_and_unproject_pybullet_data(data_dir, target_seg_id):
    """
    Loads PyBullet data and unprojects masked pixels to 3D World Coordinates
    using the provided View and Projection matrices.
    """
    # 1. Load Files
    rgb_img = np.array(Image.open(os.path.join(data_dir, "rgb_image_head.png")))
    depth_array = np.load(os.path.join(data_dir, "depth_image_head.npy"))
    seg_mask = np.load(os.path.join(data_dir, "door handle_seg_mask.npy"))
    
    view_matrix = np.load(os.path.join(data_dir, "head_view_matrix.npy"))
    projection_matrix = np.load(os.path.join(data_dir, "head_projection_matrix.npy"))

    # 2. Extract valid pixels based on segmentation mask AND valid depth
    valid_mask = (seg_mask == target_seg_id) & (depth_array < 0.999) & (depth_array > 0.0)
    v, u = np.where(valid_mask)
    
    if len(v) == 0:
        raise ValueError(f"No valid foreground pixels found for target object ID {target_seg_id}")
    
    image_height, image_width = depth_array.shape
    
    # Extract depth and colors for masked pixels
    z_raw = depth_array[v, u]
    pc_colors = rgb_img[v, u, :3] # N x 3 (drop alpha channel if RGBA)
    
    # 3. Vectorized NDC Unprojection (matching new_3d_proj from robot.py)
    ndc_x = (2.0 * u / image_width) - 1.0
    ndc_y = 1.0 - (2.0 * v / image_height)
    z_ndc = 2.0 * z_raw - 1.0
    
    clip_pos = np.stack([ndc_x, ndc_y, z_ndc, np.ones_like(ndc_x)], axis=0) # Shape: (4, N)
    
    VP = projection_matrix @ view_matrix
    inv_VP = np.linalg.inv(VP)
    
    world_hom = inv_VP @ clip_pos # Transform: Clip Space -> World Space
    pc_world_uncentered = (world_hom[:3, :] / world_hom[3, :]).T # Perspective Divide -> N x 3
    
    return pc_world_uncentered, pc_colors


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.gripper_config):
        raise ValueError(f"Gripper config {args.gripper_config} does not exist")

    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    # 1. Unproject point cloud in World coordinates (Uncentered)
    print(f"Loading and processing data from: {args.data_dir}")
    pc_uncentered, pc_color = load_and_unproject_pybullet_data(args.data_dir, args.target_seg_id)

    if len(pc_uncentered) < 20:
        print(f"ERROR: Found only {len(pc_uncentered)} valid points. GraspGen requires at least 20 points. Exiting.")
        exit(1)

    # 2. Filter the point cloud to remove floaters/noise
    print(f"Filtering outlier points with threshold {args.outlier_threshold}...")
    try:
        pc_filtered_torch, pc_removed_torch, pc_color_filtered_torch, pc_color_removed_torch = point_cloud_outlier_removal_with_color(
            torch.from_numpy(pc_uncentered), torch.from_numpy(pc_color), threshold=args.outlier_threshold
        )
        pc_filtered = pc_filtered_torch.numpy()
        pc_color_filtered = pc_color_filtered_torch.numpy()
        pc_removed = pc_removed_torch.numpy()
    except Exception as e:
        print(f"Outlier removal failed: {e}")
        pc_filtered = np.empty((0, 3)) # Force fallback below

    # Safety Fallback: If outlier removal destroyed the point cloud
    if len(pc_filtered) < 50 or len(pc_filtered) < 0.25 * len(pc_uncentered):
        print(f"WARNING: Outlier filter was too aggressive (kept {len(pc_filtered)} out of {len(pc_uncentered)} points).")
        print("Falling back to the raw, unprojected point cloud!")
        pc_filtered = pc_uncentered
        pc_color_filtered = pc_color
        pc_removed = np.empty((0, 3))

    # 3. Center the filtered Point Cloud for GraspGen Model
    pc_mean = pc_filtered.mean(axis=0)
    T_subtract_pc_mean = tra.translation_matrix(-pc_mean)
    
    pc_centered = tra.transform_points(pc_filtered, T_subtract_pc_mean)
    
    if len(pc_removed) > 0:
        pc_removed_centered = tra.transform_points(pc_removed, T_subtract_pc_mean)
    else:
        pc_removed_centered = np.empty((0, 3))

    # 4. Run GraspGen Inference
    print("Initializing GraspGen Sampler...")
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)
    
    print(f"Running Grasp Inference on {len(pc_centered)} points...")
    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_centered,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
    )

    if len(grasps_inferred) > 0:
        grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
        grasps_inferred = grasps_inferred.cpu().numpy()
        grasps_inferred[:, 3, 3] = 1
        
        # 5. Convert Grasp Poses back to Uncentered World Coordinates
        T_add_pc_mean = tra.translation_matrix(pc_mean)
        grasps_uncentered = np.array([T_add_pc_mean @ g for g in grasps_inferred])

        print(f"Inferred {len(grasps_inferred)} grasps, scores {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f}")

        # 6. Save outputs to destination directory
        out_dir = os.path.join(args.data_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        
        grasps_path = os.path.join(out_dir, "uncentered_grasps.npy")
        scores_path = os.path.join(out_dir, "grasp_scores.npy")
        np.save(grasps_path, grasps_uncentered)
        np.save(scores_path, grasp_conf_inferred)
        print(f"Successfully saved uncentered grasps to: {grasps_path}")

        # 7. Visualization
        print("Visualizing Results...")
        vis = create_visualizer()
        vis.scene.reset()
        
        # Visualize Centered Point Cloud (GraspGen requires centering visually too)
        visualize_pointcloud(vis, "pc_filtered", pc_centered, pc_color_filtered, size=0.0025)
        
        if len(pc_removed_centered) > 0:
            visualize_pointcloud(vis, "pc_removed", pc_removed_centered, [255, 0, 0], size=0.003)

        scores_inferred_colors = get_color_from_score(grasp_conf_inferred, use_255_scale=True)
        
        # Visualize Centered Grasps
        for j, grasp in enumerate(grasps_inferred):
            visualize_grasp(
                vis,
                f"grasps_objectpc_filtered/{j:03d}/grasp",
                grasp,
                color=scores_inferred_colors[j],
                gripper_name=gripper_name,
                linewidth=0.6,
            )
            
        print("Done! Check the browser visualization.")
        input("Press Enter to exit...")

    else:
        print("No valid grasps found from inference!")