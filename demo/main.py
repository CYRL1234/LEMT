import numpy as np
from LEMT.model.model_api import create_model
import torch
import torch.nn.functional as F
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
# import cv2
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from miniball import get_bounding_ball
from scipy import linalg
import time

from LEMT.model.metrics import calulate_error
from LEMT.misc.lr_scheduler import LinearWarmupCosineAnnealingLR
from LEMT.misc.utils import load_cfg, merge_args_cfg, torch2numpy, import_with_str, delete_prefix_from_state_dict, exists_and_is_true
from LEMT.misc.skeleton import ITOPSkeleton, JOINT_COLOR_MAP, MMFiSkeleton
from LEMT.misc.vis import visualize_sample
from LEMT.loss.unsup import UnsupLoss
from argparse import ArgumentParser
from glob import glob

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except Exception:
    rr = None
    RERUN_AVAILABLE = False

def _rerun_swap_y_up(points):
    if points is None or points.shape[0] == 0:
        return points
    return points[:, [0, 2, 1]]

def _rerun_log_skeleton(path, kps, color=None):
    if kps is None or kps.shape[0] == 0:
        return
    if kps.shape[0] != MMFiSkeleton.num_joints:
        return
    segments = []
    for a, b in MMFiSkeleton.bones:
        seg = np.stack([kps[a], kps[b]], axis=0)
        segments.append(seg)
    if segments:
        if color is None:
            rr.log(path, rr.LineStrips3D(segments))
        else:
            rr.log(path, rr.LineStrips3D(segments, colors=[color] * len(segments)))

def _rerun_offset_x(points, offset):
    if points is None or points.shape[0] == 0:
        return points
    shifted = points.copy()
    shifted[:, 0] += offset
    return shifted

def _rerun_offset_z(points, offset):
    if points is None or points.shape[0] == 0:
        return points
    shifted = points.copy()
    shifted[:, 2] += offset
    return shifted
def calculate_and_print_errors(predictions, ground_truths):
    """
    Calculates and prints MPJPE and PA-MPJPE from lists of predicted and ground truth poses.

    Args:
        predictions (list): A list of predicted poses (np.ndarray, shape (17, 3)).
        ground_truths (list): A list of ground truth poses (np.ndarray, shape (17, 3)).
    """
    if not predictions:
        print("No predictions were made, cannot calculate errors.")
        return

    # Convert lists to numpy arrays for vectorized operations
    preds = np.array(predictions)  # Shape: (num_frames, 17, 3)
    gts = np.array(ground_truths)    # Shape: (num_frames, 17, 3)

    # --- 1. Calculate MPJPE (Mean Per Joint Position Error) ---
    # This is the direct Euclidean distance between predicted and ground truth joints, averaged over all joints and all frames.
    mpjpe = np.mean(np.linalg.norm(preds - gts, axis=2)) * 1000  # Convert to mm

    # --- 2. Calculate PA-MPJPE (Procrustes-Aligned MPJPE) ---
    # This aligns the prediction to the ground truth before calculating the error.
    pa_mpjpe_errors = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        gt = gts[i]

        # Center both poses to the origin
        pred_centered = pred - np.mean(pred, axis=0)
        gt_centered = gt - np.mean(gt, axis=0)

        # Procrustes analysis to find the optimal rotation
        # Covariance matrix
        H = gt_centered.T @ pred_centered
        # Singular Value Decomposition
        U, S, Vt = linalg.svd(H)
        # Optimal rotation matrix
        R = Vt.T @ U.T

        # Handle reflection case
        if linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # Align the predicted pose
        pred_aligned = pred_centered @ R

        # Calculate the error for the aligned pose
        error = np.mean(np.linalg.norm(pred_aligned - gt_centered, axis=1)) * 1000 # Convert to mm
        pa_mpjpe_errors.append(error)

    pa_mpjpe = np.mean(pa_mpjpe_errors)

    print("\n==================== FINAL RESULTS ====================")
    print(f"MPJPE (Mean Per Joint Position Error): {mpjpe:.2f} mm")
    print(f"PA-MPJPE (Procrustes-Aligned MPJPE): {pa_mpjpe:.2f} mm")
    print("=====================================================\n")

def import_data(root_dir):
    """
    Imports and processes lidar, mmwave, and keypoints data from a given root directory.

    Args:
        root_dir (str): The path to the sequence directory, e.g., '/path/to/MMFi_Dataset/E01/S01/A01'.

    Returns:
        tuple: A tuple containing:
            - lidar_data (np.ndarray): LiDAR point clouds, shape [sequence_len, num_points, 3].
            - mmwave_data (np.ndarray): mmWave point clouds, shape [sequence_len, num_points, 3].
            - keypoints_data (np.ndarray): 3D keypoints, shape [sequence_len, 17, 3].
    """
    # --- 1. Load Keypoints Data ---
    # The ground truth file contains keypoints for all frames.
    keypoints_path = os.path.join(root_dir, 'ground_truth.npy')
    if not os.path.exists(keypoints_path):
        raise FileNotFoundError(f"ground_truth.npy not found in {root_dir}")
    
    keypoints_data = np.load(keypoints_path)
    
    # Apply coordinate transformations as seen in the preprocessor
    keypoints_data[..., 1] = -keypoints_data[..., 1] - 0.2
    keypoints_data[..., 2] = keypoints_data[..., 2] - 0.1

    # --- 2. Load LiDAR Data ---
    lidar_data_list = []
    lidar_dir = os.path.join(root_dir, "lidar")
    lidar_files = sorted(glob(os.path.join(lidar_dir, "frame*.bin")))

    if not lidar_files:
        print(f"Warning: No LiDAR files found in {lidar_dir}")
    
    for bin_fn in lidar_files:
        # LiDAR data has 3 features (x, y, z).
        data_tmp = np.fromfile(bin_fn, dtype=np.float64).reshape(-1, 3)
        # Keep only XYZ coordinates and apply transformations
        data_tmp = data_tmp[:, [1, 2, 0]]  # Switch from (x,y,z) to (y,z,x)
        data_tmp[..., 0] = -data_tmp[..., 0] # Negate the new x (original y)
        lidar_data_list.append(data_tmp)

    # --- 3. Load mmWave Data ---
    mmwave_data_list = []
    mmwave_dir = os.path.join(root_dir, "mmwave")
    mmwave_files = sorted(glob(os.path.join(mmwave_dir, "frame*.bin")))

    if not mmwave_files:
        print(f"Warning: No mmWave files found in {mmwave_dir}")

    for bin_fn in mmwave_files:
        # mmWave data has 5 features (x, y, z, intensity, doppler).
        data_tmp = np.fromfile(bin_fn, dtype=np.float64).reshape(-1, 5)
        # Match preprocessor: swap intensity/doppler before normalization
        data_tmp[:, [3, 4]] = data_tmp[:, [4, 3]]
        intensity = data_tmp[:, -1]
        intensity = np.clip(intensity, 0, 40.0) / 40.0
        data_tmp[:, -1] = intensity

        data_tmp = data_tmp[:, [1, 2, 0, 3, 4]]
        data_tmp[:, 1] *= -1  # negate y
        data_tmp[:, 2] += 0.1  # z + 0.1
        mmwave_data_list.append(data_tmp)

    # --- 4. Align Data Lengths ---
    # The number of frames should be consistent across LiDAR and mmWave.
    # Keypoints are loaded first and assumed to be the ground truth for sequence length.
    num_frames = len(keypoints_data)

    # Find which frames have corresponding data in all sources
    lidar_indices = {int(os.path.basename(f).split('.')[0][5:]) - 1 for f in lidar_files}
    mmwave_indices = {int(os.path.basename(f).split('.')[0][5:]) - 1 for f in mmwave_files}
    
    valid_indices = sorted(list(lidar_indices.intersection(mmwave_indices)))
    
    if not valid_indices:
        print("Warning: No matching frame indices found between LiDAR and mmWave data.")
        # Return empty arrays with correct dimensions if no frames align
        return np.array([]), np.array([]), np.array([])

    # Filter all data sources to keep only the aligned frames
    keypoints_data = keypoints_data[valid_indices]
    
    # Re-create lists based on valid indices
    final_lidar_list = [lidar_data_list[i] for i in range(len(lidar_files)) if (int(os.path.basename(lidar_files[i]).split('.')[0][5:]) - 1) in valid_indices]
    final_mmwave_list = [mmwave_data_list[i] for i in range(len(mmwave_files)) if (int(os.path.basename(mmwave_files[i]).split('.')[0][5:]) - 1) in valid_indices]

    # The lists of arrays cannot be directly converted to a single numpy array
    # because the number of points varies per frame. They remain as lists of arrays.
    return final_lidar_list, final_mmwave_list, keypoints_data

def extract_roi(point_cloud, predicted_location, edge_length):
    """
    Extracts a Region of Interest (ROI) from a point cloud based on a cubic volume.

    Args:
        point_cloud (np.ndarray): The input point cloud. Shape can be (num_points, 3) for LiDAR
                                  or (num_points, 5) for mmWave, where the first 3 columns are XYZ.
        predicted_location (np.ndarray or list/tuple): The center of the cubic ROI. Shape (3,).
        edge_length (float): The side length of the cubic ROI.

    Returns:
        np.ndarray: A new point cloud containing only the points within the ROI.
                    The original point features are preserved.
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        return np.array([])

    # Ensure predicted_location is a numpy array for calculations
    center = np.array(predicted_location)
    
    # Calculate the min and max corners of the bounding box (the cube)
    half_edge = edge_length / 2.0
    min_bound = center - half_edge
    max_bound = center + half_edge
    
    # Extract the XYZ coordinates for filtering. This handles both (N, 3) and (N, 5) cases.
    xyz = point_cloud[:, :3]
    
    # Create a boolean mask for points inside the cube
    in_roi_mask = np.all((xyz >= min_bound) & (xyz <= max_bound), axis=1)
    
    # Apply the mask to the original point cloud to get the ROI
    roi_points = point_cloud[in_roi_mask]
    
    return roi_points

def filter_pcl(bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, offset: float = 0.0):
    """
    Filter target_pcl to points inside bounding box of bounding_pcl, with buffer.
    Mirrors MMFiPreprocessor._filter_pcl.
    """
    if target_pcl is None or target_pcl.shape[0] == 0:
        return np.array([])
    upper_bound = bounding_pcl[:, :3].max(axis=0) + bound
    lower_bound = bounding_pcl[:, :3].min(axis=0) - bound
    lower_bound[2] += offset

    mask_x = (target_pcl[:, 0] >= lower_bound[0]) & (target_pcl[:, 0] <= upper_bound[0])
    mask_y = (target_pcl[:, 1] >= lower_bound[1]) & (target_pcl[:, 1] <= upper_bound[1])
    mask_z = (target_pcl[:, 2] >= lower_bound[2]) & (target_pcl[:, 2] <= upper_bound[2])
    index = mask_x & mask_y & mask_z
    return target_pcl[index]

def summarize_pc(name, pc):
    if pc is None or pc.shape[0] == 0:
        print(f"{name}: EMPTY")
        return
    xyz = pc[:, :3]
    mean = np.mean(xyz, axis=0)
    std = np.std(xyz, axis=0)
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    print(
        f"{name}: n={xyz.shape[0]} mean={mean.round(3)} std={std.round(3)} "
        f"min={min_xyz.round(3)} max={max_xyz.round(3)}"
    )

def remove_outliers_radius(point_cloud, radius=0.15, min_neighbors=3):
    """
    Removes points that have fewer than min_neighbors within radius.

    Args:
        point_cloud (np.ndarray): Input point cloud, shape (N, D) where D >= 3.
        radius (float): Neighbor search radius.
        min_neighbors (int): Minimum neighbors required to keep a point.

    Returns:
        np.ndarray: Filtered point cloud.
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        return point_cloud

    xyz = point_cloud[:, :3]
    neighbors = NearestNeighbors(radius=radius).fit(xyz)
    distances, _ = neighbors.radius_neighbors(xyz, return_distance=True)
    inliers = np.array([len(d) >= min_neighbors for d in distances])
    if np.sum(inliers) == 0:
        return point_cloud[:1]
    return point_cloud[inliers]
def remove_background(lidar_point_cloud, background_points, buffer):
    """
    Removes points from the LiDAR point cloud that are close to background points.

    Args:
        lidar_point_cloud (np.ndarray): The input LiDAR point cloud, shape (N, 3).
        background_points (np.ndarray): The point cloud representing the static background, shape (M, 3).
        buffer (float): The distance threshold. Any LiDAR point within this distance of any
                        background point will be removed.

    Returns:
        np.ndarray: The filtered LiDAR point cloud containing only foreground points.
    """
    if background_points is None or background_points.shape[0] == 0:
        # If there's no background, return the original point cloud
        return lidar_point_cloud
        
    if lidar_point_cloud is None or lidar_point_cloud.shape[0] == 0:
        return np.array([])

    # Build a k-d tree from the background points for efficient distance calculation
    background_tree = cKDTree(background_points)
    
    # For each point in the lidar cloud, find the distance to the nearest background point
    distances, _ = background_tree.query(lidar_point_cloud, k=1)
    
    # Create a boolean mask to keep only the points where the distance is greater than the buffer
    foreground_mask = distances > buffer
    
    # Apply the mask to get the foreground points
    foreground_points = lidar_point_cloud[foreground_mask]
    
    return foreground_points
def feature_transfer(lidar_points, mmwave_points, mode='empty', knn_k=3):
    """
    Transfers features from mmWave points to LiDAR points or appends zero features.

    Args:
        lidar_points (np.ndarray): The LiDAR point cloud, shape (N, 3).
        mmwave_points (np.ndarray): The mmWave point cloud, shape (M, 5).
        mode (str): The transfer mode. 'mmwave' to transfer features, 'empty' to add zeros.
        knn_k (int): The number of nearest neighbors to use for feature interpolation.

    Returns:
        np.ndarray: The LiDAR point cloud with added features, shape (N, 3 + num_features).
    """
    if lidar_points is None or lidar_points.shape[0] == 0:
        # Return an empty array with the correct number of feature columns
        num_features = (mmwave_points.shape[1] - 3) if mode == 'mmwave' and mmwave_points is not None else 0
        return np.empty((0, 3 + num_features))

    if mode == 'empty':
        # Append a column of zeros for each feature we would otherwise transfer.
        # Assuming mmWave has 2 extra features (intensity, doppler) beyond XYZ.
        num_features = 2 
        empty_features = np.zeros((lidar_points.shape[0], num_features))
        return np.concatenate([lidar_points, empty_features], axis=1)

    elif mode == 'mmwave':
        if mmwave_points is None or mmwave_points.shape[0] == 0:
            # If no mmwave data, fall back to empty mode
            return feature_transfer(lidar_points, None, mode='empty')

        lidar_xyz = lidar_points[:, :3]
        mmwave_xyz = mmwave_points[:, :3]
        mmwave_feat = mmwave_points[:, 3:]  # Get features like intensity, doppler, etc.

        # For stability, add vertices of the combined bounding box to the mmWave data.
        # This prevents issues where LiDAR points are far from any mmWave points.
        all_xyz = np.concatenate([lidar_xyz, mmwave_xyz], axis=0)
        min_xyz, max_xyz = np.min(all_xyz, axis=0), np.max(all_xyz, axis=0)
        
        # Add a small buffer to the bounding box
        expand_ratio = 0.1
        min_xyz -= (max_xyz - min_xyz) * expand_ratio
        max_xyz += (max_xyz - min_xyz) * expand_ratio

        # Create vertices of the expanded cube
        cube_vertices = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]], [min_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]], [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]], [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]], [max_xyz[0], max_xyz[1], max_xyz[2]]
        ])
        # Assign zero features to these virtual points
        cube_vertex_feat = np.zeros((8, mmwave_feat.shape[1]))

        # Add virtual points to the mmWave data
        mmwave_xyz_aug = np.concatenate([mmwave_xyz, cube_vertices], axis=0)
        mmwave_feat_aug = np.concatenate([mmwave_feat, cube_vertex_feat], axis=0)

        # Find k-nearest mmWave neighbors for each LiDAR point
        neighbors = NearestNeighbors(n_neighbors=knn_k, algorithm='auto').fit(mmwave_xyz_aug)
        distances, indices = neighbors.kneighbors(lidar_xyz)

        # Calculate weights for interpolation (inverse distance weighting)
        weights = 1.0 / (distances + 1e-8)  # Add epsilon to avoid division by zero
        weights = weights / np.sum(weights, axis=1, keepdims=True)

        # Interpolate the features
        transferred_features = np.sum(mmwave_feat_aug[indices] * weights[..., np.newaxis], axis=1)
        
        # Concatenate the original LiDAR points with the new features
        return np.concatenate([lidar_points, transferred_features], axis=1)

    else:
        raise ValueError("Invalid mode. Choose 'mmwave' or 'empty'.")
    
def pad_point_cloud(point_cloud, max_points):
    """
    Pads or downsamples a point cloud to a specific number of points.

    Args:
        point_cloud (np.ndarray): The input point cloud, shape (N, D), where D can be 3, 5, or more.
        max_points (int): The target number of points for the output cloud.

    Returns:
        np.ndarray: The processed point cloud with exactly max_points, shape (max_points, D).
    """
    num_points = point_cloud.shape[0]
    
    if num_points == max_points:
        return point_cloud
    
    # Handle empty input: create a cloud of zeros.
    if num_points == 0:
        num_features = point_cloud.shape[1] if point_cloud.ndim > 1 else 3 # Default to 3 features if shape is (0,)
        return np.zeros((max_points, num_features), dtype=point_cloud.dtype)

    if num_points < max_points:
        # --- Padding: Randomly duplicate existing points ---
        num_to_add = max_points - num_points
        # Choose indices to duplicate, with replacement
        random_indices = np.random.choice(num_points, size=num_to_add, replace=True)
        points_to_add = point_cloud[random_indices]
        
        # Concatenate original points with the new duplicated points
        padded_cloud = np.concatenate([point_cloud, points_to_add], axis=0)
        return padded_cloud

    else: # num_points > max_points
        # --- Downsampling: Randomly select a subset of points ---
        # Choose indices to keep, without replacement
        random_indices = np.random.choice(num_points, size=max_points, replace=False)
        
        downsampled_cloud = point_cloud[random_indices]
        return downsampled_cloud
    
def prepare_hpe_input(previous_frames):
    """
    Concatenates a list of point cloud frames into a single batch for HPE model input.

    Args:
        previous_frames (list): A list of NumPy arrays, where each array is a point cloud
                                for a single frame, expected to be of shape (max_points, 5).

    Returns:
        np.ndarray: A single NumPy array of shape (1, clip_len, max_points, 5).
    """
    # Stack the list of frames along a new axis (axis 0) to create a single array
    # of shape (clip_len, max_points, 5).
    hpe_input = np.stack(previous_frames, axis=0)
    
    # Add a new batch dimension at the beginning to get the final shape (1, clip_len, max_points, 5).
    hpe_input = np.expand_dims(hpe_input, axis=0)
    
    return hpe_input

def create_background(raw_lidar_pc, hpe_output, buffer, num_points):
    """
    Creates a background point cloud by removing points around the detected human pose.

    Args:
        raw_lidar_pc (np.ndarray): The raw LiDAR point cloud for the frame, shape (N, 3).
        hpe_output (np.ndarray): The estimated keypoints of the human pose, shape (17, 3).
        buffer (float): The buffer distance to add around the pose's bounding box.
        num_points (int): The target number of points for the final background cloud.

    Returns:
        np.ndarray: The processed background point cloud with shape (num_points, 3).
    """
    if raw_lidar_pc is None or raw_lidar_pc.shape[0] == 0:
        return np.zeros((num_points, 3))

    # 1. Find the bounding box of the human pose
    min_coords = np.min(hpe_output, axis=0)
    max_coords = np.max(hpe_output, axis=0)

    # 2. Expand the bounding box with the buffer
    min_bound = min_coords - buffer
    max_bound = max_coords + buffer

    # 3. Filter out points that are inside the expanded bounding box
    lidar_xyz = raw_lidar_pc[:, :3]
    
    # Create a mask for points that are *outside* the bounding box
    outside_bounds_mask = np.any((lidar_xyz < min_bound) | (lidar_xyz > max_bound), axis=1)
    
    background_candidates = raw_lidar_pc[outside_bounds_mask]

    # 4. Pad or downsample the background to the desired number of points
    final_background = pad_point_cloud(background_candidates, num_points)
    
    return final_background

def update_background(old_background, new_background, num_background_points):
    """
    Updates the main background point cloud with new background points from the current frame.

    Args:
        old_background (np.ndarray or None): The existing aggregated background point cloud.
        new_background (np.ndarray): The new background points identified in the current frame.
        num_background_points (int): The target number of points for the updated background cloud.

    Returns:
        np.ndarray: The updated and resized background point cloud.
    """
    if old_background is None:
        # If there is no old background, the new one becomes the current one.
        combined_background = new_background
    else:
        # Concatenate the old and new points.
        if new_background.shape[0] > 0:
            combined_background = np.concatenate([old_background, new_background], axis=0)
        else:
            combined_background = old_background
    
    # Use the pad_point_cloud function to downsample the combined cloud to the desired size.
    # This keeps a mix of old and new points while ensuring the background doesn't grow indefinitely.
    updated_background = pad_point_cloud(combined_background, num_background_points)
    
    return updated_background

def get_centroid(point_cloud, centroid_type='median'):
    """
    Calculates the centroid of a point cloud based on the specified type.

    Args:
        point_cloud (np.ndarray): The input point cloud, shape (N, 3) or (N, 5).
        centroid_type (str): The method to use for centroid calculation.
                             Options: 'minball', 'mean', 'median', 'zonly', 'xz', 'none'.

    Returns:
        np.ndarray: The calculated centroid, shape (3,).
    """
    # Ensure there are points to process
    if point_cloud is None or point_cloud.shape[0] == 0:
        return np.zeros(3)

    # Use only unique XYZ coordinates for centroid calculation
    pc_dedupe = np.unique(point_cloud[:, :3], axis=0)
    
    if pc_dedupe.shape[0] == 0:
        return np.zeros(3)

    if centroid_type == 'minball':
        try:
            # get_bounding_ball returns center and squared radius
            centroid, _ = get_bounding_ball(pc_dedupe)
        except Exception as e:
            print(f"Warning: Miniball failed ({e}), falling back to mean centroid.")
            centroid = np.mean(pc_dedupe, axis=0)
    elif centroid_type == 'mean':
        centroid = np.mean(pc_dedupe, axis=0)
    elif centroid_type == 'median':
        centroid = np.median(pc_dedupe, axis=0)
    elif centroid_type == 'zonly':
        centroid = np.zeros(3)
        centroid[2] = np.median(pc_dedupe[:, 2])
    elif centroid_type == 'xz':
        centroid = np.array([np.median(pc_dedupe[:, 0]), 0, np.median(pc_dedupe[:, 2])])
    elif centroid_type == 'none':
        centroid = np.zeros(3)
    else:
        raise ValueError(f"Invalid centroid_type: {centroid_type}")
    
    return np.array(centroid)

def normalize(point_cloud, centroid):
    """
    Normalizes a point cloud by subtracting the centroid.

    Args:
        point_cloud (np.ndarray): The input point cloud, shape (N, D) where D >= 3.
        centroid (np.ndarray or list/tuple): The centroid to subtract, shape (3,).

    Returns:
        np.ndarray: The normalized point cloud.
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        return point_cloud

    centroid = np.array(centroid).reshape(1, 3)
    normalized_pc = point_cloud.copy()
    normalized_pc[:, :3] -= centroid  # Subtract only from XYZ coordinates

    return normalized_pc

def remove_outliers(point_cloud, z_thresh=3.0):
    """
    Removes outlier points from a point cloud based on Z-score thresholding.

    Args:
        point_cloud (np.ndarray): The input point cloud, shape (N, D) where D >= 3.
        z_thresh (float): The Z-score threshold for identifying outliers.

    Returns:
        np.ndarray: The filtered point cloud with outliers removed.
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        return point_cloud

    xyz = point_cloud[:, :3]
    mean = np.mean(xyz, axis=0)
    std = np.std(xyz, axis=0)

    # Calculate Z-scores
    z_scores = np.abs((xyz - mean) / (std + 1e-8))  # Add epsilon to avoid division by zero

    # Create a mask for points within the Z-score threshold for all dimensions
    inlier_mask = np.all(z_scores < z_thresh, axis=1)

    # Filter the point cloud
    filtered_pc = point_cloud[inlier_mask]

    return filtered_pc

def remove_outliers_box(point_cloud, radius=3.0, center=(0.0, 1.0)):
    """
    Removes outliers using a fixed box in the XY plane, matching the dataset pipeline.

    Args:
        point_cloud (np.ndarray): The input point cloud, shape (N, D) where D >= 3.
        radius (float): Half-width of the box in XY.
        center (tuple): Center of the box in XY.

    Returns:
        np.ndarray: The filtered point cloud.
    """
    if point_cloud is None or point_cloud.shape[0] == 0:
        return point_cloud

    center_xy = np.array([[center[0], center[1]]])
    inliers = np.all(np.abs(point_cloud[:, :2] - center_xy) < radius, axis=1)
    if np.sum(inliers) == 0:
        return point_cloud[:1]
    return point_cloud[inliers]

def visualize_and_save(frame_idx, step_name, point_clouds, colors, sizes, output_dir="/home/ryan/MM-Fi/LEMT/demo/display_images", mode = 'after_normalization', draw_bones=False):
    """
    Visualizes point clouds and saves the plot to a file.

    Args:
        frame_idx (int): The current frame index.
        step_name (str): A name for the visualization step (e.g., 'raw_data').
        point_clouds (list): A list of point cloud numpy arrays to plot. Each array should be (N, 3).
        colors (list): A list of colors corresponding to each point cloud.
        sizes (list): A list of marker sizes corresponding to each point cloud.
        output_dir (str): The directory to save the images.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot with Y as vertical axis (map Y -> Z)
    for pc, color, size in zip(point_clouds, colors, sizes):
        if pc is not None and pc.shape[0] > 0:
            ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=color, s=size)
            if draw_bones and pc.shape[0] == MMFiSkeleton.num_joints:
                for a, b in MMFiSkeleton.bones:
                    xa, za, ya = pc[a, 0], pc[a, 2], pc[a, 1]
                    xb, zb, yb = pc[b, 0], pc[b, 2], pc[b, 1]
                    ax.plot([xa, xb], [za, zb], [ya, yb], c=color, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(f'Frame {frame_idx} - {step_name}')
    ax.view_init(elev=30, azim=45)
    
    # Set axis limits for consistent viewing
    # ax.set_xlim([-2, 2])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([4, 2])
    if mode == 'before_normalization':
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([1, 4])
        ax.set_zlim([-1.5, 1.5])
    elif mode == 'after_normalization':
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-2.5, 0.5])
        ax.set_zlim([-1, 2])
    else:
        raise ValueError("Invalid mode for visualization limits.")

    filename = os.path.join(output_dir, f"frame_0_{step_name}.png")
    plt.savefig(filename)
    plt.close(fig)

def main(args):
    background_points = None
    previous_frames = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    enable_rerun = getattr(args, 'enable_rerun', False) and RERUN_AVAILABLE
    if enable_rerun:
        rr.init("mmfi_demo", spawn=False)
        grpc_port = getattr(args, 'rerun_grpc_port', 9877)
        web_port = getattr(args, 'rerun_web_port', 9090)
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=server_uri)

    # while True:
    background_points = None
    previous_frames = []
    previous_mmwave_frames = []
    previous_lidar_frames = []
    previous_kps_frames = []
    previous_filtered_lidar_frames = []
    previous_extracted_mmwave_frames = []
    all_predictions = []
    all_ground_truths = []

    clip_len = 5
    max_points = 128
    num_background_points = 2048
    num_new_background_points = 1024

    root_dir = args.dataset_test_path
    # root_dir = '/home/ryan/MM-Fi/MMFi_Dataset/E01/S01/A01'
    # print(f"Received input path: {root_dir}")
    # if os.path.isdir(root_dir):
    #     break
    # else:
    #     print("The provided path is not a valid directory. Please try again.")
    # print(f"Processing data from: {root_dir}")
    localization_model = create_model(args.localization_model_name, args.localization_model_params)
    checkpoint = torch.load(args.localization_checkpoint, map_location=device)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    localization_model.load_state_dict(state_dict, strict=False)
    localization_model.to(device)
    localization_model.eval()
    for param in localization_model.parameters():
        param.requires_grad = False

    HPE_model = create_model(args.model_name, args.model_params)
    checkpoint_HPE = torch.load(args.model_checkpoint, map_location=device)
    state_dict_HPE = {k.replace('model.', ''): v for k, v in checkpoint_HPE['state_dict'].items() if k.startswith('model.')}
    HPE_model.load_state_dict(state_dict_HPE, strict=False)
    HPE_model.to(device)
    HPE_model.eval()
    for param in HPE_model.parameters():
        param.requires_grad = False

    lidar_data, mmwave_data, keypoints_data = import_data(root_dir)
    # print(f"Data imported. Number of frames: {len(lidar_data)}")
    
    for frame_idx in range(len(lidar_data)):
        if enable_rerun:
            if hasattr(rr, "set_time_sequence"):
                rr.set_time_sequence("frame", frame_idx)
            elif hasattr(rr, "set_time_seconds"):
                rr.set_time_seconds("frame", float(frame_idx))
        # Store original data for visualization before normalization
        original_lidar_for_vis = lidar_data[frame_idx].copy()
        original_mmwave_for_vis = mmwave_data[frame_idx].copy()
        original_gt_for_vis = keypoints_data[frame_idx].copy()

        # 1. Visualize raw lidar data + mmwave data
        # visualize_and_save(frame_idx, "1_raw_data",
        #                     [original_lidar_for_vis, original_mmwave_for_vis[:, :3]],
        #                     ['black', 'orange'], [1, 10], mode = 'before_normalization')

        if enable_rerun:
            rr.log("raw/lidar", rr.Points3D(_rerun_swap_y_up(original_lidar_for_vis)))
            rr.log("raw/mmwave", rr.Points3D(_rerun_swap_y_up(original_mmwave_for_vis[:, :3])))
            rr.log("raw/keypoints", rr.Points3D(_rerun_swap_y_up(original_gt_for_vis)))
            _rerun_log_skeleton("raw/skeleton", _rerun_swap_y_up(original_gt_for_vis))
        
        print(f"--- Processing Frame {frame_idx} ---")
        
        # if background_points is None:
        #     print("Current background is None.")
        # else:
        #     print(f"Current background has {background_points.shape[0]} points.")

        previous_lidar_frames.append(lidar_data[frame_idx])
        if len(previous_lidar_frames) > clip_len:
            previous_lidar_frames.pop(0)

        if len(previous_lidar_frames) < clip_len:
            fill_lidar_frames = [previous_lidar_frames[0]] * (clip_len - len(previous_lidar_frames))
            current_lidar_frames = fill_lidar_frames + previous_lidar_frames
        else:
            current_lidar_frames = previous_lidar_frames

        # Store raw mmWave frames; normalize the clip with the same keypoint later
        previous_mmwave_frames.append(mmwave_data[frame_idx])
        if len(previous_mmwave_frames) > clip_len:
            previous_mmwave_frames.pop(0)

        # Store keypoints for HPE centroid (training uses kps-based centroid)
        previous_kps_frames.append(keypoints_data[frame_idx])
        if len(previous_kps_frames) > clip_len:
            previous_kps_frames.pop(0)

        # Build kps clip for HPE centroid
        if len(previous_kps_frames) < clip_len:
            fill_kps = [previous_kps_frames[0]] * (clip_len - len(previous_kps_frames))
            current_kps_frames = fill_kps + previous_kps_frames
        else:
            current_kps_frames = previous_kps_frames

        # If the history is not full, pad it by duplicating the first frame
        if len(previous_mmwave_frames) < clip_len:
            fill_frames = [previous_mmwave_frames[0]] * (clip_len - len(previous_mmwave_frames))
            current_mmwave_frames = fill_frames + previous_mmwave_frames
        else:
            current_mmwave_frames = previous_mmwave_frames

        # Localization centroid (reuse get_centroid for mmWave median)
        keypoint = get_centroid(np.concatenate(current_mmwave_frames, axis=0), centroid_type='median')

        normalized_lidar = normalize(lidar_data[frame_idx], keypoint)
        normalized_gt_loc = normalize(keypoints_data[frame_idx], keypoint)

        # Normalize the entire mmWave clip with the same keypoint
        normalized_mmwave_frames = []
        for mm_frame in current_mmwave_frames:
            mm_norm = normalize(mm_frame, keypoint)
            mm_norm = remove_outliers_box(mm_norm, radius=3.0)
            mm_norm = pad_point_cloud(mm_norm, 128)
            normalized_mmwave_frames.append(mm_norm)
        
        # Prepare the batch from numpy arrays and then convert to a tensor
        mmwave_input_numpy = prepare_hpe_input(normalized_mmwave_frames)
        mmwave_input_tensor = torch.from_numpy(mmwave_input_numpy).float().to(device)

        # print(f"Localization model input shape: {mmwave_input_tensor.shape}")
        predicted_location_tensor = localization_model(mmwave_input_tensor)
        predicted_location = predicted_location_tensor.detach().cpu().numpy().squeeze()
        predicted_location_abs = predicted_location + keypoint

        if enable_rerun:
            x_raw = 0.0
            x_roi = 4.0
            x_hpe = 8.0

            gt_loc = keypoints_data[frame_idx][7]
            rr.log(
                "raw/lidar",
                rr.Points3D(
                    _rerun_swap_y_up(_rerun_offset_x(original_lidar_for_vis, x_raw)),
                    colors=[(150, 180, 255)]
                )
            )
            rr.log(
                "raw/mmwave",
                rr.Points3D(
                    _rerun_swap_y_up(_rerun_offset_x(original_mmwave_for_vis[:, :3], x_raw)),
                    colors=[(255, 210, 150)]
                )
            )
            rr.log(
                "raw/gt_location",
                rr.Points3D(
                    _rerun_swap_y_up(_rerun_offset_x(np.array([gt_loc]), x_raw)),
                    colors=[(0, 255, 0)],
                    radii=0.03
                )
            )
            rr.log(
                "raw/pred_location",
                rr.Points3D(
                    _rerun_swap_y_up(_rerun_offset_x(np.array([predicted_location_abs]), x_raw)),
                    colors=[(255, 0, 0)],
                    radii=0.03
                )
            )

        # HPE centroid (training uses kps: median x, min y, median z)
        kps_cat = np.concatenate(current_kps_frames, axis=0)
        # hpe_centroid = np.array([
        #     np.median(kps_cat[:, 0]),
        #     np.min(kps_cat[:, 1]),
        #     np.median(kps_cat[:, 2])
        # ])

        
        # 2. Visualize predicted human location + lidar data
        # visualize_and_save(frame_idx, "2_localization",
        #                     [normalized_lidar, np.array([predicted_location]), np.array([normalized_gt_loc[7]])],
        #                     ['black', 'red', 'blue'], [1, 50, 50])

        # print(f"Predicted Location: {predicted_location}")
        # print(f"Ground Truth Location: {normalized_gt_loc[7]}")

        extracted_mmwave = extract_roi(mmwave_data[frame_idx], predicted_location_abs, edge_length=2.5)
        extracted_lidar = extract_roi(lidar_data[frame_idx], predicted_location_abs, edge_length=2.5)
        # print(f"Extracted {extracted_lidar.shape[0]} LiDAR points and {extracted_mmwave.shape[0]} mmWave points in ROI.")

        # Obtain hpe_centroid using predicted
        

        #store extracted mmWave for future use
        previous_extracted_mmwave_frames.append(extracted_mmwave)
        if len(previous_extracted_mmwave_frames) > clip_len:
            previous_extracted_mmwave_frames.pop(0)
        if len(previous_extracted_mmwave_frames) < clip_len:
            fill_extracted_mmwave = [previous_extracted_mmwave_frames[0]] * (clip_len - len(previous_extracted_mmwave_frames))
            current_extracted_mmwave_frames = fill_extracted_mmwave + previous_extracted_mmwave_frames
        else:
            current_extracted_mmwave_frames = previous_extracted_mmwave_frames

        

        # 4. Remove background points from LiDAR
        if background_points is not None:
            filtered_lidar = remove_background(extracted_lidar, background_points, buffer=0.1)
            # extracted_mmwave = remove_background(extracted_mmwave[:, :3], background_points, buffer=0.1)
        else:
            filtered_lidar = extracted_lidar

        if enable_rerun:
            rr.log(
                "roi_bg/lidar",
                rr.Points3D(
                    _rerun_swap_y_up(_rerun_offset_x(filtered_lidar, x_roi)),
                    colors=[(60, 90, 220)]
                )
            )
            rr.log(
                "roi_bg/mmwave",
                rr.Points3D(
                    _rerun_swap_y_up(_rerun_offset_x(extracted_mmwave[:, :3], x_roi)),
                    colors=[(220, 150, 70)]
                )
            )

        previous_filtered_lidar_frames.append(filtered_lidar)

        if len(previous_filtered_lidar_frames) > clip_len:
            previous_filtered_lidar_frames.pop(0)
        
        if len(previous_filtered_lidar_frames) < clip_len:
            fill_filtered_lidar = [previous_filtered_lidar_frames[0]] * (clip_len - len(previous_filtered_lidar_frames))
            current_filtered_lidar_frames = fill_filtered_lidar + previous_filtered_lidar_frames
        else:
            current_filtered_lidar_frames = previous_filtered_lidar_frames
        
        # Obtain hpe_centroid using lidar_human over the clip
        lidar_cat = np.concatenate(current_filtered_lidar_frames, axis=0) if len(current_filtered_lidar_frames) > 0 else np.zeros((0, 3))
        if lidar_cat.shape[0] > 0:
            hpe_centroid = np.array([
                np.median(lidar_cat[:, 0]),
                np.min(lidar_cat[:, 1]),
                np.median(lidar_cat[:, 2])
            ])
        else:
            hpe_centroid = predicted_location_abs.copy()

        normalized_gt_hpe = normalize(keypoints_data[frame_idx], hpe_centroid)

        extracted_lidar_norm = normalize(extracted_lidar, hpe_centroid)
        extracted_mmwave_norm = normalize(extracted_mmwave, hpe_centroid)
        # visualize_and_save(frame_idx, "3_roi_extraction",
        #                     [extracted_lidar_norm, extracted_mmwave_norm[:, :3]],
        #                     ['green', 'orange'], [1, 10])
        
        filtered_lidar_norm = normalize(filtered_lidar, hpe_centroid)
        # print(f"Filtered LiDAR has {filtered_lidar.shape[0]} points after background removal.")
        # 4.5 Visualize filtered LiDAR
        # visualize_and_save(frame_idx, "4_background_filtered_lidar",
        #                     [filtered_lidar_norm, extracted_mmwave_norm[:, :3]],
        #                     ['blue', 'orange'], [1, 10])

        
        #visualize pcl_filtered lidar and mmwave point cloud for the current single frame
        temp_filt_lidar = filter_pcl(keypoints_data[frame_idx], lidar_data[frame_idx], bound=0.2)
        temp_filt_mmwave = filter_pcl(keypoints_data[frame_idx], mmwave_data[frame_idx], bound=0.2)
        temp_filt_lidar_norm = normalize(temp_filt_lidar, hpe_centroid)
        temp_filt_mmwave_norm = normalize(temp_filt_mmwave, hpe_centroid)
        # visualize_and_save(frame_idx, "4_filtered_lidar_single_frame",
        #                     [temp_filt_lidar_norm, temp_filt_mmwave_norm[:, :3]],
        #                     ['blue', 'orange'], [1, 10])
        # Compare filter_pcl vs ROI+background stats (current frame)
        # summarize_pc("filter_pcl_lidar", temp_filt_lidar)
        # summarize_pc("roi_bg_lidar", filtered_lidar)
        # summarize_pc("filter_pcl_mmwave", temp_filt_mmwave)
        # summarize_pc("roi_mmwave", extracted_mmwave)
        # filter_pcl approach (visualize last frame only)
        if len(current_lidar_frames) > 0:
            last_pc = current_lidar_frames[-1]
            last_mm = current_mmwave_frames[-1]
            last_kp = current_kps_frames[-1]
            pc_filt = filter_pcl(last_kp, last_pc, bound=0.2)
            mm_filt = filter_pcl(last_kp, last_mm, bound=0.2)
            pc_norm = normalize(pc_filt, hpe_centroid)
            mm_norm = normalize(mm_filt, hpe_centroid)
            pc_norm = remove_outliers_box(pc_norm, radius=1.5)
            pc_norm = remove_outliers_radius(pc_norm, radius=0.15, min_neighbors=3)
            pc_feat = feature_transfer(pc_norm, mm_norm, mode='mmwave', knn_k=3)
            pc_padded = pad_point_cloud(pc_feat, max_points)
            visualize_and_save(frame_idx, "4d_filter_pcl_last_frame",
                                [pc_norm, mm_norm[:, :3]],
                                ['blue', 'orange'], [1, 10])

        # ROI + background approach (used for HPE)
        hpe_frames = []
        for idx, (pc_frame, mm_frame) in enumerate(zip(current_filtered_lidar_frames, current_extracted_mmwave_frames)):
            pc_norm = normalize(pc_frame, hpe_centroid)
            mm_norm = normalize(mm_frame, hpe_centroid)
            pc_norm = remove_outliers_box(pc_norm, radius=1.5)
            mm_norm = remove_outliers_box(mm_norm, radius=1.5)
            pc_norm = remove_outliers_radius(pc_norm, radius=0.15, min_neighbors=3)
            pc_feat = feature_transfer(pc_norm, mm_norm, mode='mmwave', knn_k=3)
            pc_padded = pad_point_cloud(pc_feat, max_points)
            hpe_frames.append(pc_padded)
            if idx == len(current_filtered_lidar_frames) - 1:
                visualize_and_save(frame_idx, "4e_roi_bg_last_frame",
                                    [pc_norm, mm_norm[:, :3]],
                                    ['blue', 'orange'], [1, 10])

        previous_frames = hpe_frames
        if len(previous_frames) > clip_len:
            previous_frames.pop(0)
        elif len(previous_frames) < clip_len and previous_frames:
            while len(previous_frames) < clip_len:
                previous_frames.insert(0, previous_frames[0])
        
        if len(previous_frames) == clip_len:
            HPE_input_numpy = prepare_hpe_input(previous_frames)
            HPE_input_tensor = torch.from_numpy(HPE_input_numpy).float().to(device)
            # print(f"HPE input shape: {HPE_input_tensor.shape}")

            with torch.no_grad():
                hpe_output_tensor = HPE_model(HPE_input_tensor)
            
            hpe_output = hpe_output_tensor.detach().cpu().numpy().squeeze()
            # print(f"HPE output shape: {hpe_output.shape}")
            all_predictions.append(hpe_output)
            all_ground_truths.append(normalized_gt_hpe)

            hpe_output_vis = hpe_output + hpe_centroid
            
            # 5. Visualize predicted keypoints + original lidar data
            # visualize_and_save(frame_idx, "5_predicted_keypoints",
            #                 [extracted_lidar_norm, hpe_output],
            #                 ['black', 'blue'], [1, 20], draw_bones=True)

            # 6. Visualize ground truth keypoints + original lidar data
            # visualize_and_save(frame_idx, "6_ground_truth_keypoints",
            #                 [extracted_lidar_norm, normalized_gt_hpe],
            #                 ['black', 'purple'], [1, 20], draw_bones=True)

            if enable_rerun:
                rr.log(
                    "hpe/lidar",
                    rr.Points3D(
                        _rerun_swap_y_up(_rerun_offset_x(original_lidar_for_vis, x_hpe)),
                        colors=[(0, 0, 255)]
                    )
                )
                rr.log(
                    "hpe/mmwave",
                    rr.Points3D(
                        _rerun_swap_y_up(_rerun_offset_x(original_mmwave_for_vis[:, :3], x_hpe)),
                        colors=[(255, 165, 0)]
                    )
                )

                gt_kps_raw = _rerun_offset_x(original_gt_for_vis, x_hpe)
                pred_kps_raw = _rerun_offset_x(hpe_output_vis, x_hpe)
                rr.log(
                    "hpe/gt_keypoints",
                    rr.Points3D(_rerun_swap_y_up(gt_kps_raw), colors=[(0, 255, 0)])
                )
                _rerun_log_skeleton(
                    "hpe/gt_skeleton",
                    _rerun_swap_y_up(gt_kps_raw),
                    color=(0, 255, 0)
                )
                rr.log(
                    "hpe/pred_keypoints",
                    rr.Points3D(_rerun_swap_y_up(pred_kps_raw), colors=[(255, 0, 0)])
                )
                _rerun_log_skeleton(
                    "hpe/pred_skeleton",
                    _rerun_swap_y_up(pred_kps_raw),
                    color=(255, 0, 0)
                )

            new_background = create_background(lidar_data[frame_idx], hpe_output_vis, buffer=0.2, num_points=num_new_background_points)
            # print(f"Created new background with {new_background.shape[0]} points.")

            background_points = update_background(background_points, new_background, num_background_points)
            # print(f"Updated background has {background_points.shape[0]} points.")
        else:
            print("Not enough frames to run HPE model yet.")
        
        print("--------------------------------------------------")
        time.sleep(0.1) # Sleep for 0.1s in each iteration
        # === After the loop finishes, calculate and print the errors ===
        if all_predictions:
            calculate_and_print_errors(all_predictions, all_ground_truths)
        else:
            print("No frames were processed to calculate error.")

    if enable_rerun:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down visualization server.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/test.yaml')
    parser.add_argument('--dataset_test_path', type=str, default='/home/ryan/MM-Fi/MMFi_Dataset/E04/S31/A03')
    parser.add_argument('--enable_rerun', action='store_true')
    parser.add_argument('--rerun_grpc_port', type=int, default=9877)
    parser.add_argument('--rerun_web_port', type=int, default=9090)

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    main(args)



