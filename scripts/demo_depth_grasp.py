# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Coding Agent doc : update here changes:
Usage:
python scripts/demo_depth_grasp.py --data_dir D:\NLP\Robotics\GraspModels\GraspGen\data\open_door_pybullet --gripper_config /data/repos/GraspGenModels/checkpoints/graspgen_franka_panda.yml


Coding Agent change log : update here changes
Key design decisions:

   - Uses depth_and_segmentation_to_point_clouds (which wraps depth2points) — requires linearizing the OpenGL depth buffer first
   - Near/far are derived from the projection matrix (0.01 / 100)
   - Intrinsics extracted from projection matrix: fx=fy≈221.7, cx=cy=128
   - Grasps are visualized in centered camera frame (same as demo_object_pc.py), plus saved in world frame to grasp_poses_world_frame.npy
"""
import argparse
import glob
import os

import numpy as np
import torch
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.viser_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import (
    depth_and_segmentation_to_point_clouds,
    point_cloud_outlier_removal,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps from depth image + segmentation mask after GraspGen inference"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing depth .npy, segmentation mask .npy, RGB .png, and camera matrices",
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
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
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
    return parser.parse_args()


def find_file(data_dir, patterns):
    """Find a single file matching one of the given glob patterns."""
    for pattern in patterns:
        matches = glob.glob(os.path.join(data_dir, pattern))
        if matches:
            return matches[0]
    return None


def load_data(data_dir):
    """Load depth, segmentation, RGB, view matrix, and projection matrix from data_dir."""
    depth_path = find_file(data_dir, ["depth_image_*.npy", "depth*.npy"])
    seg_path = find_file(data_dir, ["*_seg_mask.npy", "*seg*.npy"])
    rgb_path = find_file(data_dir, ["rgb_image_*.png", "rgb*.png"])
    view_matrix_path = find_file(data_dir, ["*_view_matrix.npy", "*view_matrix*.npy"])
    proj_matrix_path = find_file(data_dir, ["*_projection_matrix.npy", "*projection_matrix*.npy"])

    if depth_path is None:
        raise FileNotFoundError(f"No depth .npy file found in {data_dir}")
    if seg_path is None:
        raise FileNotFoundError(f"No segmentation mask .npy file found in {data_dir}")
    if proj_matrix_path is None:
        raise FileNotFoundError(f"No projection matrix .npy file found in {data_dir}")

    print(f"Loading depth: {depth_path}")
    print(f"Loading segmentation: {seg_path}")

    depth_raw = np.load(depth_path)
    seg_mask = np.load(seg_path)
    projection_matrix = np.load(proj_matrix_path)

    view_matrix = None
    if view_matrix_path is not None:
        print(f"Loading view matrix: {view_matrix_path}")
        view_matrix = np.load(view_matrix_path)

    rgb_image = None
    if rgb_path is not None:
        from PIL import Image
        print(f"Loading RGB: {rgb_path}")
        rgb_image = np.array(Image.open(rgb_path).convert("RGB"))

    return depth_raw, seg_mask, rgb_image, view_matrix, projection_matrix


def derive_near_far_from_projection(projection_matrix):
    """Derive near and far clip planes from OpenGL projection matrix.

    For a symmetric perspective projection:
        P[2][2] = -(f+n)/(f-n)
        P[2][3] = -2*f*n/(f-n)
    """
    A = projection_matrix[2, 2]  # -(f+n)/(f-n)
    B = projection_matrix[2, 3]  # -2*f*n/(f-n)

    # near = B / (A - 1), far = B / (A + 1)
    near = B / (A - 1.0)
    far = B / (A + 1.0)
    return abs(near), abs(far)


def opengl_depth_to_linear_meters(depth_buffer, near, far):
    """Convert raw OpenGL depth buffer [0,1] to linear depth in meters."""
    z = depth_buffer
    linear_depth = (2.0 * near * far) / (far + near - (2.0 * z - 1.0) * (far - near))
    return linear_depth


def extract_intrinsics_from_projection(projection_matrix, image_width, image_height):
    """Extract fx, fy, cx, cy from OpenGL projection matrix.

    For symmetric frustum:
        P[0][0] = 2*n / (right - left) = 2*fx / width
        P[1][1] = 2*n / (top - bottom) = 2*fy / height
        cx = width / 2  (symmetric)
        cy = height / 2  (symmetric)
    """
    fx = projection_matrix[0, 0] * image_width / 2.0
    fy = projection_matrix[1, 1] * image_height / 2.0
    cx = image_width / 2.0
    cy = image_height / 2.0
    return fx, fy, cx, cy


def grasps_to_world_frame(grasps_centered, pc_mean, view_matrix):
    """Transform grasp poses from centered-camera-frame to world frame.

    grasps_centered are in camera frame with pc_mean subtracted.
    To get world frame: inv(view_matrix) @ T_add_mean @ grasp
    """
    # Undo centering
    T_add_mean = tra.translation_matrix(pc_mean)
    # Camera-to-world transform
    cam_to_world = np.linalg.inv(view_matrix)

    world_grasps = []
    for g in grasps_centered:
        grasp_cam = T_add_mean @ g
        grasp_world = cam_to_world @ grasp_cam
        world_grasps.append(grasp_world)
    return np.array(world_grasps)


if __name__ == "__main__":
    args = parse_args()
    
    CONVERT_TO_METER_DEPTH = False

    if not os.path.exists(args.gripper_config):
        raise ValueError(f"Gripper config {args.gripper_config} does not exist")

    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    # Load data
    depth_raw, seg_mask, rgb_image, view_matrix, projection_matrix = load_data(args.data_dir)
    image_height, image_width = depth_raw.shape[:2]
    
    if CONVERT_TO_METER_DEPTH:
        # Derive near/far from projection matrix and convert depth to meters
        near, far = derive_near_far_from_projection(projection_matrix)
        print(f"Derived near={near:.4f}, far={far:.4f} from projection matrix")

        depth_meters = opengl_depth_to_linear_meters(depth_raw, near, far)
        print(f"Linear depth range: [{depth_meters.min():.4f}, {depth_meters.max():.4f}] meters")
    else:
        depth_meters = depth_raw
        
    # Extract intrinsics from projection matrix
    fx, fy, cx, cy = extract_intrinsics_from_projection(projection_matrix, image_width, image_height)
    print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Generate point cloud from depth + segmentation
    scene_pc, object_pc, scene_colors, object_colors = depth_and_segmentation_to_point_clouds(
        depth_image=depth_meters,
        segmentation_mask=seg_mask,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rgb_image=rgb_image,
        target_object_id=1,
    )
    print(f"Object point cloud: {len(object_pc)} points")

    # Load grasp config and initialize sampler
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    # Center object point cloud (same as demo_object_pc.py)
    pc_mean = object_pc.mean(axis=0)
    T_subtract_pc_mean = tra.translation_matrix(-pc_mean)
    pc_centered = tra.transform_points(object_pc, T_subtract_pc_mean)

    # Create visualizer
    vis = create_visualizer()

    # Visualize object point cloud
    pc_color = object_colors if object_colors is not None else [0, 200, 0]
    visualize_pointcloud(vis, "object_pc", pc_centered, pc_color, size=0.0025)

    # Outlier removal
    pc_filtered, pc_removed = point_cloud_outlier_removal(torch.from_numpy(pc_centered))
    pc_filtered = pc_filtered.numpy()
    pc_removed = pc_removed.numpy()
    visualize_pointcloud(vis, "pc_removed", pc_removed, [255, 0, 0], size=0.003)

    # Run inference
    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc_filtered,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
    )

    if len(grasps_inferred) > 0:
        grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
        grasps_inferred = grasps_inferred.cpu().numpy()
        grasps_inferred[:, 3, 3] = 1
        scores_inferred = get_color_from_score(grasp_conf_inferred, use_255_scale=True)
        print(
            f"Inferred {len(grasps_inferred)} grasps, scores: "
            f"[{grasp_conf_inferred.min():.3f}, {grasp_conf_inferred.max():.3f}]"
        )

        # Visualize inferred grasps
        for j, grasp in enumerate(grasps_inferred):
            visualize_grasp(
                vis,
                f"grasps_inferred/{j:03d}/grasp",
                grasp,
                color=scores_inferred[j],
                gripper_name=gripper_name,
                linewidth=0.6,
            )

        # Transform grasps to world frame
        if view_matrix is not None:
            grasps_world = grasps_to_world_frame(grasps_inferred, pc_mean, view_matrix)
            print(f"\nWorld-frame grasp poses ({len(grasps_world)} grasps):")
            for i, g in enumerate(grasps_world[:5]):
                pos = g[:3, 3]
                print(f"  Grasp {i}: position=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            if len(grasps_world) > 5:
                print(f"  ... and {len(grasps_world) - 5} more")

            # Save world-frame grasps
            output_path = os.path.join(args.data_dir, "grasp_poses_world_frame.npy")
            np.save(output_path, grasps_world)
            print(f"Saved world-frame grasps to: {output_path}")
    else:
        print("No grasps found from inference!")

    input("Press Enter to exit...")
