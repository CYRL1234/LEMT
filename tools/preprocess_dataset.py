import numpy as np
import pickle
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import argparse
import h5py
from plyfile import PlyData
import json
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
# from smpl import SMPL
import torch
try:
    from LEMT.model.model_api import create_model
    from LEMT.misc.utils import load_cfg, merge_args_cfg
except Exception:
    import sys
    lemt_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if lemt_root not in sys.path:
        sys.path.insert(0, lemt_root)
    from model.model_api import create_model
    from misc.utils import load_cfg, merge_args_cfg
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#-----Debug Functions-----#
def plot_data(point_clouds, mmwave_data, name):
    # pkl_folder = '/home/ryan/MM-Fi/LEMT/data_dual/mmfi_dual.pkl'  # Change this to your folder path
    # case_name = 'case_1'  # Specify the case you want to check
    output_folder = '/home/ryan/MM-Fi/PcDisplay'  # Change this to your desired output folder

    os.makedirs(output_folder, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot point clouds (in blue)
    ax.scatter(point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2], c='blue', label='Point Clouds', s=1)

    # Plot mmwave data (in orange) if available
    if mmwave_data is not None and len(mmwave_data) > 0:
        ax.scatter(mmwave_data[:, 0], mmwave_data[:, 1], mmwave_data[:, 2], c='orange', label='MMWave Data', s=1)

    # Set requested axis limits: x: [-0.5, 1.0], y: [-1.5, 1.5], z: [2.0, 4.0]
    ax.set_xlim([-0.5, 1.0])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([2.0, 4.0])

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Point Cloud and MMWave Data Visualization')
    ax.legend()

    # Define output file name
    output_file = os.path.join(output_folder, f"{name}_frame.png")
    # Save the plot as an image file
    plt.savefig(output_file)
    plt.close()  # Close the figure to free memory

# ----- ROI + Background Helpers (match demo/main.py) -----
def extract_roi(point_cloud, predicted_location, edge_length):
    if point_cloud is None or point_cloud.shape[0] == 0:
        if point_cloud is None:
            print("Point cloud is None at extract_roi")
            return np.zeros((0, 3))
        # if point_cloud.shape[0] == 0:
        #     print("Point cloud is empty (shape[0] == 0) at extract_roi")
        return np.zeros((0, point_cloud.shape[1]))

    center = np.array(predicted_location)
    half_edge = edge_length / 2.0
    min_bound = center - half_edge
    max_bound = center + half_edge
    xyz = point_cloud[:, :3]
    in_roi_mask = np.all((xyz >= min_bound) & (xyz <= max_bound), axis=1)
    return point_cloud[in_roi_mask]

def normalize(point_cloud, centroid):
    if point_cloud is None or point_cloud.shape[0] == 0:
        return point_cloud
    normalized_pc = point_cloud.copy()
    normalized_pc[:, :3] -= np.array(centroid).reshape(1, 3)
    return normalized_pc

def remove_outliers_box(point_cloud, radius=3.0, center=(0.0, 1.0)):
    if point_cloud is None or point_cloud.shape[0] == 0:
        return point_cloud
    center_xy = np.array([[center[0], center[1]]])
    inliers = np.all(np.abs(point_cloud[:, :2] - center_xy) < radius, axis=1)
    if np.sum(inliers) == 0:
        return point_cloud[:1]
    return point_cloud[inliers]

def remove_outliers_radius(point_cloud, radius=0.15, min_neighbors=3):
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
    if background_points is None or background_points.shape[0] == 0:
        return lidar_point_cloud
    if lidar_point_cloud is None or lidar_point_cloud.shape[0] == 0:
        if lidar_point_cloud is None:
            print("LiDAR point cloud is None at remove_background")
            return np.zeros((0, 3))
        # if lidar_point_cloud.shape[0] == 0:
        #     print("LiDAR point cloud is empty (shape[0] == 0) at remove_background")
        return np.zeros((0, lidar_point_cloud.shape[1]))
    background_tree = cKDTree(background_points)
    distances, _ = background_tree.query(lidar_point_cloud, k=1)
    foreground_mask = distances > buffer
    return lidar_point_cloud[foreground_mask]

def feature_transfer(lidar_points, mmwave_points, mode='mmwave', knn_k=3):
    if lidar_points is None or lidar_points.shape[0] == 0:
        num_features = (mmwave_points.shape[1] - 3) if mode == 'mmwave' and mmwave_points is not None else 0
        return np.empty((0, 3 + num_features))

    if mode == 'empty':
        num_features = 2
        empty_features = np.zeros((lidar_points.shape[0], num_features))
        return np.concatenate([lidar_points, empty_features], axis=1)

    if mmwave_points is None or mmwave_points.shape[0] == 0:
        return feature_transfer(lidar_points, None, mode='empty')

    lidar_xyz = lidar_points[:, :3]
    mmwave_xyz = mmwave_points[:, :3]
    mmwave_feat = mmwave_points[:, 3:]

    all_xyz = np.concatenate([lidar_xyz, mmwave_xyz], axis=0)
    min_xyz, max_xyz = np.min(all_xyz, axis=0), np.max(all_xyz, axis=0)
    expand_ratio = 0.1
    min_xyz -= (max_xyz - min_xyz) * expand_ratio
    max_xyz += (max_xyz - min_xyz) * expand_ratio
    cube_vertices = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]], [min_xyz[0], min_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]], [min_xyz[0], max_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]], [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]], [max_xyz[0], max_xyz[1], max_xyz[2]]
    ])
    cube_vertex_feat = np.zeros((8, mmwave_feat.shape[1]))
    mmwave_xyz_aug = np.concatenate([mmwave_xyz, cube_vertices], axis=0)
    mmwave_feat_aug = np.concatenate([mmwave_feat, cube_vertex_feat], axis=0)

    neighbors = NearestNeighbors(n_neighbors=knn_k, algorithm='auto').fit(mmwave_xyz_aug)
    distances, indices = neighbors.kneighbors(lidar_xyz)
    weights = 1.0 / (distances + 1e-8)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    transferred_features = np.sum(mmwave_feat_aug[indices] * weights[..., np.newaxis], axis=1)
    return np.concatenate([lidar_points, transferred_features], axis=1)

def pad_point_cloud(point_cloud, max_points):
    num_points = point_cloud.shape[0]
    if num_points == max_points:
        return point_cloud
    if num_points == 0:
        num_features = point_cloud.shape[1] if point_cloud.ndim > 1 else 3
        return np.zeros((max_points, num_features), dtype=point_cloud.dtype)
    if num_points < max_points:
        num_to_add = max_points - num_points
        random_indices = np.random.choice(num_points, size=num_to_add, replace=True)
        points_to_add = point_cloud[random_indices]
        return np.concatenate([point_cloud, points_to_add], axis=0)

    # Downsample uniformly if too many points
    keep_indices = np.random.choice(num_points, size=max_points, replace=False)
    return point_cloud[keep_indices]

def create_background(raw_lidar_pc, hpe_output, buffer, num_points):
    if raw_lidar_pc is None or raw_lidar_pc.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float64)
    min_coords = np.min(hpe_output, axis=0)
    max_coords = np.max(hpe_output, axis=0)
    min_coords = min_coords - buffer
    max_coords = max_coords + buffer
    outside_bounds_mask = np.any((raw_lidar_pc < min_coords) | (raw_lidar_pc > max_coords), axis=1)
    background_candidates = raw_lidar_pc[outside_bounds_mask]
    return pad_point_cloud(background_candidates, num_points)

def update_background(old_background, new_background, num_background_points):
    if old_background is None:
        combined_background = new_background
    else:
        if new_background.shape[0] > 0:
            combined_background = np.concatenate([old_background, new_background], axis=0)
        else:
            combined_background = old_background
    return pad_point_cloud(combined_background, num_background_points)

def get_centroid(point_cloud, centroid_type='median'):
    if point_cloud is None or point_cloud.shape[0] == 0:
        return np.zeros(3)
    pc_dedupe = np.unique(point_cloud[:, :3], axis=0)
    if centroid_type == 'median':
        centroid = np.median(pc_dedupe, axis=0)
    elif centroid_type == 'mean':
        centroid = np.mean(pc_dedupe, axis=0)
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

def prepare_hpe_input(previous_frames):
    hpe_input = np.stack(previous_frames, axis=0)
    hpe_input = np.expand_dims(hpe_input, axis=0)
    return hpe_input

#------------------------#

class Preprocessor():
    def __init__(self, root_dir, out_dir):
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.results = {}
        self.results['splits'] = {}
        self.results['sequences'] = []

    def process(self):
        pass

    def save(self, name):
        print(os.path.join(self.out_dir, f'{name}.pkl'))
        with open(os.path.join(self.out_dir, f'{name}.pkl'), 'wb') as f:
            pickle.dump(self.results, f)

    def _add_to_split(self, split_name, idx):
        if split_name not in self.results['splits']:
            self.results['splits'][split_name] = []
        self.results['splits'][split_name].append(idx)

    def _normalize_intensity(self, feat, max_value):
        feat = np.clip(feat, 0, max_value)
        feat /= max_value
        return feat

class MiliPointPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir):
        super().__init__(root_dir, out_dir)

    def process(self):
        point_fns = [os.path.join(self.root_dir, f'{i}.pkl') for i in range(0, 19)]
        action_labels = np.load(os.path.join(self.root_dir, 'action_label.npy'))

        pcs = []
        kps = []
        for point_fn in point_fns:
            with open(point_fn, 'rb') as f:
                data = pickle.load(f)
            for d in data:
                pcs.append(d['x'])
                kps.append(d['y'])

        assert len(pcs) == len(action_labels), 'Number of point clouds and action labels do not match'
        start_idx = 0
        for i in tqdm(range(len(action_labels)+1)):
            if i == len(action_labels) or (i >= 1 and action_labels[i] != action_labels[i-1]):
                if action_labels[i-1] == -1:
                    continue
                self.results['sequences'].append({
                    'point_clouds': pcs[start_idx:i],
                    'keypoints': np.stack(kps[start_idx:i]),
                    'action': action_labels[i-1]
                })
                start_idx = i

        seq_idxs = np.arange(len(self.results['sequences']))
        np.random.shuffle(seq_idxs)
        num_train = int(len(seq_idxs) * 0.8)
        num_val = int(len(seq_idxs) * 0.1)
        self.results['splits']['train'] = seq_idxs[:num_train]
        self.results['splits']['val'] = seq_idxs[num_train:num_train+num_val]
        self.results['splits']['test'] = seq_idxs[num_train+num_val:]

    def save(self):
        super().save('milipoint')
# class MMFiPreprocessor(Preprocessor):
#     def __init__(self, root_dir, out_dir, modality='mmwave'):
#         super().__init__(root_dir, out_dir)
#         self.action_p1 = ['2', '3', '4', '5', '13', '14', '17', '18', '19', '20', '21', '22', '23', '27']
#         assert modality in ['mmwave', 'lidar', 'dual']  # Added 'dual' mode
#         self.modality = modality

#     def process(self):
#         dirs = sorted(glob(os.path.join(self.root_dir, 'E*/S*/A*')))

#         seq_idxs = np.arange(len(dirs))
#         np.random.shuffle(seq_idxs)
#         num_train = int(len(seq_idxs) * 0.8)
#         num_val = int(len(seq_idxs) * 0.1)
#         self.results['splits']['train_rdn_p3'] = sorted(seq_idxs[:num_train])
#         self.results['splits']['val_rdn_p3'] = sorted(seq_idxs[num_train:num_train+num_val])
#         self.results['splits']['test_rdn_p3'] = sorted(seq_idxs[num_train+num_val:])

#         for i, d in tqdm(enumerate(dirs)):
#             env = int(d.split('/')[-3][1:])
#             subject = int(d.split('/')[-2][1:])
#             action = int(d.split('/')[-1][1:])
#             pcs = []
#             mmwave_data = None  # Initialize mmWave data storage

#             if self.modality in ['mmwave', 'dual']:
#                 keep_idxs = []
#                 for bin_fn in sorted(glob(os.path.join(d.replace('MMFi_Dataset', 'filtered_mmwave'), "frame*.bin"))):
#                     data_tmp = self._read_bin(bin_fn)
#                     data_tmp[:, -1] = self._normalize_intensity(data_tmp[:, -1], 40.0)
#                     data_tmp = data_tmp[:, [1, 2, 0, 3, 4]]
#                     pcs.append(data_tmp)
#                     keep_idx = int(os.path.basename(bin_fn).split('.')[0][5:]) - 1
#                     keep_idxs.append(keep_idx)
#                 mmwave_data = np.load(os.path.join(d, 'ground_truth.npy'))[keep_idxs,...]

#             if self.modality in ['lidar', 'dual']:
#                 for bin_fn in sorted(glob(os.path.join(d, "lidar", "frame*.bin"))):
#                     data_tmp = self._read_bin(bin_fn)
#                     data_tmp = data_tmp[:, [1, 2, 0]]
#                     data_tmp[..., 0] = -data_tmp[..., 0]
#                     pcs.append(data_tmp)
#                 kps = np.load(os.path.join(d, 'ground_truth.npy'))
#                 kps[..., 0] = kps[..., 0]
#                 kps[..., 1] = -kps[..., 1] - 0.2
#                 kps[..., 2] = kps[..., 2] - 0.1

#             new_pcs = []
#             for pc, kp in zip(pcs, kps):
#                 print(f'Before filtering: {len(pc)} points')
#                 pc = self._filter_pcl(kp, pc, bound=0.2)
#                 print(f'After filtering: {len(pc)} points')
#                 new_pcs.append(pc)

#             # Store both LiDAR and mmWave data in the output
#             self.results['sequences'].append({
#                 'point_clouds': new_pcs,
#                 'keypoints': kps,
#                 'action': action,
#                 'mmwave_data': mmwave_data if mmwave_data is not None else []  # Add mmWave data if available
#             })
class MMFiPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir, modality='mmwave', localization_checkpoint=None,
                 localization_model_name=None, localization_model_params=None, device=None, split_mode='random', seed = 0,
                 localization_input='mmwave', background_downsample_rate=None, background_update_rate=None):
        super().__init__(root_dir, out_dir)
        self.action_p1 = ['2', '3', '4', '5', '13', '14', '17', '18', '19', '20', '21', '22', '23', '27']
        assert modality in ['mmwave', 'lidar', 'dual', 'raw_dual', 'roi_bg_dual', 'bg_only_dual']
        self.modality = modality
        self.localization_checkpoint = localization_checkpoint
        self.localization_model_name = localization_model_name
        self.localization_model_params = localization_model_params
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.localization_model = None
        self.localization_input = localization_input
        self.split_mode = split_mode
        self._fixed_rng = np.random.RandomState(seed) # For fixed separation splitting, 42 is arbitrary seed which will produce same splits every time
        self.seed = seed
        self.background_downsample_rate = background_downsample_rate
        self.background_update_rate = background_update_rate

    def _load_localization_model(self):
        if self.localization_model is not None:
            return
        if self.localization_checkpoint is None:
            raise ValueError('roi_bg_dual requires --localization_checkpoint')

        checkpoint = torch.load(self.localization_checkpoint, map_location=self.device)
        model_name = self.localization_model_name
        model_params = self.localization_model_params

        if (model_name is None or model_params is None) and isinstance(checkpoint, dict):
            hparams = checkpoint.get('hyper_parameters', {})
            if model_name is None:
                model_name = hparams.get('model_name', None)
            if model_params is None:
                model_params = hparams.get('model_params', None)

        if model_name is None:
            raise ValueError('Localization model name not found. Provide --localization_model_name or ensure checkpoint has hyper_parameters.model_name')

        model = create_model(model_name, model_params)
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.localization_model = model

    def process(self):
        # print("modality: ",self.modality)
        print(
            "[MMFiPreprocessor] modality={modality} localization_input={loc_input} split_mode={split_mode} "
            "background_downsample_rate={bg_down} background_update_rate={bg_update}".format(
                modality=self.modality,
                loc_input=self.localization_input,
                split_mode=self.split_mode,
                bg_down=self.background_downsample_rate,
                bg_update=self.background_update_rate,
            )
        )
        dirs = sorted(glob(os.path.join(self.root_dir, 'E*/S*/A*')))

        seq_idxs = np.arange(len(dirs))
        if self.split_mode == 'fixed_seperation':
            self._fixed_rng.shuffle(seq_idxs)
        else:
            np.random.shuffle(seq_idxs)
        num_train = int(len(seq_idxs) * 0.8)
        num_val = int(len(seq_idxs) * 0.1)
        self.results['splits']['train_rdn_p3'] = sorted(seq_idxs[:num_train])
        self.results['splits']['val_rdn_p3'] = sorted(seq_idxs[num_train:num_train+num_val])
        self.results['splits']['test_rdn_p3'] = sorted(seq_idxs[num_train+num_val:])

        for i, d in tqdm(enumerate(dirs)):
            env = int(d.split('/')[-3][1:])
            subject = int(d.split('/')[-2][1:])
            action = int(d.split('/')[-1][1:])
            if self.modality == 'roi_bg_dual':
                self._load_localization_model()
                # --- Load mmWave and align frames with LiDAR ---
                mmwave_map = {}
                mmwave_indices = []
                for bin_fn in sorted(glob(os.path.join(d, "mmwave", "frame*.bin"))):
                    data_tmp = self._read_bin(bin_fn, True)
                    data_tmp[:, -1] = self._normalize_intensity(data_tmp[:, -1], 40.0)
                    data_tmp = data_tmp[:, [1, 2, 0, 3, 4]]
                    data_tmp[:, 1] *= -1  # negate y
                    data_tmp[:, 2] += 0.1  # z + 0.1
                    keep_idx = int(os.path.basename(bin_fn).split('.')[0][5:]) - 1
                    mmwave_map[keep_idx] = data_tmp
                    mmwave_indices.append(keep_idx)

                lidar_map = {}
                lidar_indices = []
                for bin_fn in sorted(glob(os.path.join(d, "lidar", "frame*.bin"))):
                    data_tmp = self._read_bin(bin_fn)
                    data_tmp = data_tmp[:, [1, 2, 0]]
                    data_tmp[..., 0] = -data_tmp[..., 0]
                    lidar_idx = int(os.path.basename(bin_fn).split('.')[0][5:]) - 1
                    lidar_map[lidar_idx] = data_tmp
                    lidar_indices.append(lidar_idx)

                valid_indices = sorted(set(mmwave_indices).intersection(lidar_indices))
                if len(valid_indices) == 0:
                    continue

                kps = np.load(os.path.join(d, 'ground_truth.npy'))
                kps[..., 1] = -kps[..., 1] - 0.2
                kps[..., 2] = kps[..., 2] - 0.1
                kps = kps[valid_indices]

                pcs = [lidar_map[idx] for idx in valid_indices]
                pcs_mmwave = [mmwave_map[idx] for idx in valid_indices]

                # --- ROI + Background pipeline (match demo/main.py) ---
                clip_len = 5
                max_points = 128
                num_background_points = 512 #original 2048
                num_new_background_points = 128 #original 1024, 256
                use_dynamic_bg = (
                    self.background_downsample_rate is not None and
                    self.background_update_rate is not None
                )

                background_points = None
                previous_mmwave_frames = []
                previous_lidar_frames = []

                new_pcs = []
                new_kps = []
                new_mms = []
                for frame_idx, (lidar_frame, mm_frame, kp_frame) in enumerate(zip(pcs, pcs_mmwave, kps)):
                    previous_lidar_frames.append(lidar_frame)
                    if len(previous_lidar_frames) > clip_len:
                        previous_lidar_frames.pop(0)

                    if len(previous_lidar_frames) < clip_len:
                        fill_lidar_frames = [previous_lidar_frames[0]] * (clip_len - len(previous_lidar_frames))
                        current_lidar_frames = fill_lidar_frames + previous_lidar_frames
                    else:
                        current_lidar_frames = previous_lidar_frames

                    previous_mmwave_frames.append(mm_frame)
                    if len(previous_mmwave_frames) > clip_len:
                        previous_mmwave_frames.pop(0)

                    if len(previous_mmwave_frames) < clip_len:
                        fill_frames = [previous_mmwave_frames[0]] * (clip_len - len(previous_mmwave_frames))
                        current_mmwave_frames = fill_frames + previous_mmwave_frames
                    else:
                        current_mmwave_frames = previous_mmwave_frames

                    if self.localization_input in ('lidar', 'feature_transferred_lidar'):
                        keypoint = get_centroid(np.concatenate(current_lidar_frames, axis=0), centroid_type='median')
                    else:
                        keypoint = get_centroid(np.concatenate(current_mmwave_frames, axis=0), centroid_type='median')

                    if self.localization_input == 'lidar':
                        normalized_loc_frames = []
                        for pc_prev in current_lidar_frames:
                            pc_norm = normalize(pc_prev, keypoint)
                            pc_norm = remove_outliers_box(pc_norm, radius=3.0)
                            pc_norm = pad_point_cloud(pc_norm, 256)
                            normalized_loc_frames.append(pc_norm)
                    elif self.localization_input == 'feature_transferred_lidar':
                        normalized_loc_frames = []
                        for pc_prev, mm_prev in zip(current_lidar_frames, current_mmwave_frames):
                            pc_norm = normalize(pc_prev, keypoint)
                            mm_norm = normalize(mm_prev, keypoint)
                            pc_norm = remove_outliers_box(pc_norm, radius=3.0)
                            pc_feat = feature_transfer(pc_norm, mm_norm, mode='mmwave', knn_k=3)
                            pc_feat = pad_point_cloud(pc_feat, 256)
                            normalized_loc_frames.append(pc_feat)
                    else:
                        normalized_loc_frames = []
                        for mm_prev in current_mmwave_frames:
                            mm_norm = normalize(mm_prev, keypoint)
                            mm_norm = remove_outliers_box(mm_norm, radius=3.0)
                            mm_norm = pad_point_cloud(mm_norm, 128)
                            normalized_loc_frames.append(mm_norm)

                    loc_input_numpy = prepare_hpe_input(normalized_loc_frames)
                    loc_input_tensor = torch.from_numpy(loc_input_numpy).float().to(self.device)

                    with torch.no_grad():
                        predicted_location_tensor = self.localization_model(loc_input_tensor)

                    predicted_location = predicted_location_tensor.detach().cpu().numpy().squeeze()
                    predicted_location_abs = predicted_location + keypoint

                    extracted_mmwave = extract_roi(mm_frame, predicted_location_abs, edge_length=2.5)
                    extracted_lidar = extract_roi(lidar_frame, predicted_location_abs, edge_length=2.5)

                    if background_points is not None:
                        filtered_lidar = remove_background(extracted_lidar, background_points, buffer=0.1)
                    else:
                        filtered_lidar = extracted_lidar

                    if extracted_lidar.shape[0] == 0 or extracted_mmwave.shape[0] == 0:
                        print(
                            f"[roi_bg_dual] empty ROI at seq {i} frame {frame_idx}: "
                            f"lidar_raw={lidar_frame.shape[0]} mm_raw={mm_frame.shape[0]} "
                            f"lidar_roi={extracted_lidar.shape[0]} mm_roi={extracted_mmwave.shape[0]} "
                            f"pred_loc={predicted_location_abs.round(3)}"
                        )

                    new_pcs.append(filtered_lidar)
                    new_kps.append(kp_frame)
                    new_mms.append(extracted_mmwave)

                    if use_dynamic_bg:
                        min_coords = np.min(kp_frame, axis=0)
                        max_coords = np.max(kp_frame, axis=0)
                        min_coords = min_coords - 0.2
                        max_coords = max_coords + 0.2
                        lidar_xyz = lidar_frame[:, :3]
                        outside_bounds_mask = np.any((lidar_xyz < min_coords) | (lidar_xyz > max_coords), axis=1)
                        curr_bg_num = int(np.sum(outside_bounds_mask))
                        num_background_points = max(1, int(curr_bg_num * self.background_downsample_rate))
                        num_new_background_points = max(1, int(num_background_points * self.background_update_rate))

                    new_background = create_background(
                        lidar_frame,
                        kp_frame,
                        buffer=0.2,
                        num_points=num_new_background_points
                    )
                    background_points = update_background(background_points, new_background, num_background_points)

                self.results['sequences'].append({
                    'point_clouds': new_pcs,
                    'keypoints': np.stack(new_kps),
                    'action': action,
                    'mmwave_data': new_mms
                })

                # Split assignment happens below for all modalities
                if i in self.results['splits']['train_rdn_p3']:
                    if action in self.action_p1:
                        self._add_to_split('train_rdn_p1', i)
                    else:
                        self._add_to_split('train_rdn_p2', i)
                elif i in self.results['splits']['val_rdn_p3']:
                    if action in self.action_p1:
                        self._add_to_split('val_rdn_p1', i)
                    else:
                        self._add_to_split('val_rdn_p2', i)
                else:
                    if action in self.action_p1:
                        self._add_to_split('test_rdn_p1', i)
                    else:
                        self._add_to_split('test_rdn_p2', i)

                if subject % 5 == 0:
                    self._add_to_split('test_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('test_xsub_p1', i)
                    else:
                        self._add_to_split('test_xsub_p2', i)
                elif subject % 5 == 1:
                    self._add_to_split('val_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('val_xsub_p1', i)
                    else:
                        self._add_to_split('val_xsub_p2', i)
                else:
                    self._add_to_split('train_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('train_xsub_p1', i)
                    else:
                        self._add_to_split('train_xsub_p2', i)

                if env == 4:
                    self._add_to_split('test_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('test_xenv_p1', i)
                    else:
                        self._add_to_split('test_xenv_p2', i)
                elif env == 3:
                    self._add_to_split('val_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('val_xenv_p1', i)
                    else:
                        self._add_to_split('val_xenv_p2', i)
                else:
                    self._add_to_split('train_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('train_xenv_p1', i)
                    else:
                        self._add_to_split('train_xenv_p2', i)

                continue
            elif self.modality == 'bg_only_dual':
                # --- Load mmWave and align frames with LiDAR ---
                mmwave_map = {}
                mmwave_indices = []
                for bin_fn in sorted(glob(os.path.join(d, "mmwave", "frame*.bin"))):
                    data_tmp = self._read_bin(bin_fn, True)
                    data_tmp[:, -1] = self._normalize_intensity(data_tmp[:, -1], 40.0)
                    data_tmp = data_tmp[:, [1, 2, 0, 3, 4]]
                    data_tmp[:, 1] *= -1  # negate y
                    data_tmp[:, 2] += 0.1  # z + 0.1
                    keep_idx = int(os.path.basename(bin_fn).split('.')[0][5:]) - 1
                    mmwave_map[keep_idx] = data_tmp
                    mmwave_indices.append(keep_idx)

                lidar_map = {}
                lidar_indices = []
                for bin_fn in sorted(glob(os.path.join(d, "lidar", "frame*.bin"))):
                    data_tmp = self._read_bin(bin_fn)
                    data_tmp = data_tmp[:, [1, 2, 0]]
                    data_tmp[..., 0] = -data_tmp[..., 0]
                    lidar_idx = int(os.path.basename(bin_fn).split('.')[0][5:]) - 1
                    lidar_map[lidar_idx] = data_tmp
                    lidar_indices.append(lidar_idx)

                valid_indices = sorted(set(mmwave_indices).intersection(lidar_indices))
                if len(valid_indices) == 0:
                    continue

                kps = np.load(os.path.join(d, 'ground_truth.npy'))
                kps[..., 1] = -kps[..., 1] - 0.2
                kps[..., 2] = kps[..., 2] - 0.1
                kps = kps[valid_indices]

                pcs = [lidar_map[idx] for idx in valid_indices]
                pcs_mmwave = [mmwave_map[idx] for idx in valid_indices]

                num_background_points = 1024 #original 2048
                num_new_background_points = 128 #original 1024, 256
                use_dynamic_bg = (
                    self.background_downsample_rate is not None and
                    self.background_update_rate is not None
                )

                if use_dynamic_bg:
                    min_coords = np.min(kps[0], axis=0)
                    max_coords = np.max(kps[0], axis=0)
                    min_coords = min_coords - 0.2
                    max_coords = max_coords + 0.2
                    lidar_xyz = pcs[0][:, :3]
                    outside_bounds_mask = np.any((lidar_xyz < min_coords) | (lidar_xyz > max_coords), axis=1)
                    curr_bg_num = int(np.sum(outside_bounds_mask))
                    num_background_points = max(1, int(curr_bg_num * self.background_downsample_rate))
                    num_new_background_points = max(1, int(num_background_points * self.background_update_rate))

                background_points = create_background(
                    pcs[0],
                    kps[0],
                    buffer=0.2,
                    num_points=num_background_points
                )

                new_pcs = []
                new_kps = []
                new_mms = []
                for lidar_frame, mm_frame, kp_frame in zip(pcs, pcs_mmwave, kps):
                    if background_points is not None:
                        filtered_lidar = remove_background(lidar_frame, background_points, buffer=0.1)
                    else:
                        filtered_lidar = lidar_frame

                    new_pcs.append(filtered_lidar)
                    new_kps.append(kp_frame)
                    new_mms.append(mm_frame)

                    if use_dynamic_bg:
                        min_coords = np.min(kp_frame, axis=0)
                        max_coords = np.max(kp_frame, axis=0)
                        min_coords = min_coords - 0.2
                        max_coords = max_coords + 0.2
                        lidar_xyz = lidar_frame[:, :3]
                        outside_bounds_mask = np.any((lidar_xyz < min_coords) | (lidar_xyz > max_coords), axis=1)
                        curr_bg_num = int(np.sum(outside_bounds_mask))
                        num_background_points = max(1, int(curr_bg_num * self.background_downsample_rate))
                        num_new_background_points = max(1, int(num_background_points * self.background_update_rate))

                    new_background = create_background(
                        lidar_frame,
                        kp_frame,
                        buffer=0.2,
                        num_points=num_new_background_points
                    )
                    background_points = update_background(background_points, new_background, num_background_points)

                self.results['sequences'].append({
                    'point_clouds': new_pcs,
                    'keypoints': np.stack(new_kps),
                    'action': action,
                    'mmwave_data': new_mms
                })

                # Split assignment happens below for all modalities
                if i in self.results['splits']['train_rdn_p3']:
                    if action in self.action_p1:
                        self._add_to_split('train_rdn_p1', i)
                    else:
                        self._add_to_split('train_rdn_p2', i)
                elif i in self.results['splits']['val_rdn_p3']:
                    if action in self.action_p1:
                        self._add_to_split('val_rdn_p1', i)
                    else:
                        self._add_to_split('val_rdn_p2', i)
                else:
                    if action in self.action_p1:
                        self._add_to_split('test_rdn_p1', i)
                    else:
                        self._add_to_split('test_rdn_p2', i)

                if subject % 5 == 0:
                    self._add_to_split('test_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('test_xsub_p1', i)
                    else:
                        self._add_to_split('test_xsub_p2', i)
                elif subject % 5 == 1:
                    self._add_to_split('val_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('val_xsub_p1', i)
                    else:
                        self._add_to_split('val_xsub_p2', i)
                else:
                    self._add_to_split('train_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('train_xsub_p1', i)
                    else:
                        self._add_to_split('train_xsub_p2', i)

                if env == 4:
                    self._add_to_split('test_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('test_xenv_p1', i)
                    else:
                        self._add_to_split('test_xenv_p2', i)
                elif env == 3:
                    self._add_to_split('val_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('val_xenv_p1', i)
                    else:
                        self._add_to_split('val_xenv_p2', i)
                else:
                    self._add_to_split('train_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('train_xenv_p1', i)
                    else:
                        self._add_to_split('train_xenv_p2', i)

                continue
            else:
                pcs = []
                pcs_mmwave = []
                bug_index = 0
                if self.modality == 'mmwave':
                    keep_idxs = []
                    # for bin_fn in sorted(glob(os.path.join(d.replace('MMFi_Dataset', 'filtered_mmwave'), "frame*.bin"))):
                    for bin_fn in sorted(glob(os.path.join(d, "mmwave", "frame*.bin"))):
                        data_tmp = self._read_bin(bin_fn)
                        data_tmp[:, -1] = self._normalize_intensity(data_tmp[:, -1], 40.0)
                        data_tmp = data_tmp[:, [1, 2, 0, 3, 4]]
                        data_tmp[..., 0] = -data_tmp[..., 0]
                        #plot data for debugging
                        # print("Plotting mmwave data for debugging...")
                        # plot_data(data_tmp, None, "mmwave_only")
                        pcs.append(data_tmp)
                        keep_idx = int(os.path.basename(bin_fn).split('.')[0][5:]) - 1
                        keep_idxs.append(keep_idx)
                    kps = np.load(os.path.join(d, 'ground_truth.npy'))[keep_idxs,...]
                elif self.modality == 'dual' or self.modality == 'raw_dual':
                    # print("Processing dual or raw_dual modality...")
                    keep_idxs = []
                    for bin_fn in sorted(glob(os.path.join(d, "mmwave", "frame*.bin"))):
                        data_tmp = self._read_bin(bin_fn, True)
                        # if bug_index==0:
                        #     plot_data(data_tmp, "before_mmwave")
                        data_tmp[:, -1] = self._normalize_intensity(data_tmp[:, -1], 40.0)
                        data_tmp = data_tmp[:, [1, 2, 0, 3, 4]]
                        data_tmp[:, 1] *= -1  # negate y
                        data_tmp[:, 2] += 0.1  # z + 0.1
                        # data_tmp[..., 0] = -data_tmp[..., 0]     # negate x like LiDAR
                        # if bug_index==0:
                        #     plot_data(data_tmp, "after_mmwave")
                        #     bug_index+=1
                        pcs_mmwave.append(data_tmp)
                        keep_idx = int(os.path.basename(bin_fn).split('.')[0][5:]) - 1
                        keep_idxs.append(keep_idx)

                    kps_mm = np.load(os.path.join(d, 'ground_truth.npy'))[keep_idxs,...]
                    bug_index=0
                    for bin_fn in sorted(glob(os.path.join(d, "lidar", "frame*.bin"))):
                        data_tmp = self._read_bin(bin_fn)
                        # if bug_index==0:
                        #     plot_data(data_tmp, "before")
                        data_tmp = data_tmp[:, [1, 2, 0]]
                        data_tmp[..., 0] = -data_tmp[..., 0]
                        # if bug_index==0:
                        #     plot_data(data_tmp, "after")
                        #     bug_index+=1
                        pcs.append(data_tmp)
                    kps = np.load(os.path.join(d, 'ground_truth.npy'))
                    kps[..., 0] = kps[..., 0]
                    kps[..., 1] = -kps[..., 1]- 0.2
                    kps[..., 2] = kps[..., 2] - 0.1
                    #perform the same for kps_mm
                    kps_mm[..., 0] = kps_mm[..., 0]
                    kps_mm[..., 1] = -kps_mm[..., 1]- 0.2
                    kps_mm[..., 2] = kps_mm[..., 2] - 0.1
                elif self.modality == 'lidar':
                    for bin_fn in sorted(glob(os.path.join(d, "lidar", "frame*.bin"))):
                        data_tmp = self._read_bin(bin_fn)
                        data_tmp = data_tmp[:, [1, 2, 0]]
                        data_tmp[..., 0] = -data_tmp[..., 0]
                        pcs.append(data_tmp)
                    kps = np.load(os.path.join(d, 'ground_truth.npy'))
                    kps[..., 0] = kps[..., 0]
                    kps[..., 1] = -kps[..., 1]- 0.2
                    kps[..., 2] = kps[..., 2] - 0.1
                new_pcs = []
                mmwave_data = []
                #plot every frame before filtering
                # if self.modality == 'dual' and len(pcs)>0 and len(pcs_mmwave)>0:
                #     for pc, mm in zip(pcs, pcs_mmwave):
                #         plot_data(pc,mm, "before_filtering_dual")
                #         break
                # plot_data(pcs[0],pcs_mmwave[0], "before_filtering_lidar")
                if self.modality != 'raw_dual':
                    for pc, kp in zip(pcs, kps):
                        # print(f'Before filtering: {len(pc)} points')
                        # plot_data(pc, None, "before_filtered_lidar")
                        pc = self._filter_pcl(kp, pc, bound=0.2)
                        # plot_data(pc, None, "filtered_lidar")
                        # print(f'After filtering: {len(pc)} points')
                        new_pcs.append(pc)
                    # print("len pcs_mmwave: ", len(pcs_mmwave) if self.modality == 'dual' else 0)
                    if self.modality == 'dual':
                        for mm, kp in zip(pcs_mmwave, kps_mm):
                            # print(f'Before filtering: {len(mm)} points')
                            mm = self._filter_pcl(kp, mm, bound=0.2)
                            # plot_data(mm, "filtered_mmwave")
                            # print(f'After filtering: {len(mm)} points')
                            mmwave_data.append(mm)
                else:
                    new_pcs = pcs
                    mmwave_data = pcs_mmwave
                # # plot a frame after filtering
                # if self.modality == 'dual' and len(new_pcs)>0 and len(mmwave_data)>0:
                #     plot_data(new_pcs[0],mmwave_data[0], "after_filtering_dual")   
                if self.modality == 'dual' or self.modality == 'raw_dual':
                    self.results['sequences'].append({
                        'point_clouds': new_pcs,
                        'keypoints': kps_mm if self.modality == 'raw_dual' else kps,
                        'action': action,
                        'mmwave_data': mmwave_data 
                    })
                else:
                    self.results['sequences'].append({
                        'point_clouds': new_pcs,
                        'keypoints': kps,
                        'action': action                    
                    })

                if i in self.results['splits']['train_rdn_p3']:
                    if action in self.action_p1:
                        self._add_to_split('train_rdn_p1', i)
                    else:
                        self._add_to_split('train_rdn_p2', i)
                elif i in self.results['splits']['val_rdn_p3']:
                    if action in self.action_p1:
                        self._add_to_split('val_rdn_p1', i)
                    else:
                        self._add_to_split('val_rdn_p2', i)
                else:
                    if action in self.action_p1:
                        self._add_to_split('test_rdn_p1', i)
                    else:
                        self._add_to_split('test_rdn_p2', i)

                if subject % 5 == 0:
                    self._add_to_split('test_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('test_xsub_p1', i)
                    else:
                        self._add_to_split('test_xsub_p2', i)
                elif subject % 5 == 1:
                    self._add_to_split('val_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('val_xsub_p1', i)
                    else:
                        self._add_to_split('val_xsub_p2', i)
                else:
                    self._add_to_split('train_xsub_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('train_xsub_p1', i)
                    else:
                        self._add_to_split('train_xsub_p2', i)

                if env == 4:
                    self._add_to_split('test_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('test_xenv_p1', i)
                    else:
                        self._add_to_split('test_xenv_p2', i)
                elif env == 3:
                    self._add_to_split('val_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('val_xenv_p1', i)
                    else:
                        self._add_to_split('val_xenv_p2', i)
                else:
                    self._add_to_split('train_xenv_p3', i)
                    if action in self.action_p1:
                        self._add_to_split('train_xenv_p1', i)
                    else:
                        self._add_to_split('train_xenv_p2', i)

    def save(self):
        super().save('mmfi_' + self.modality)

    def _read_bin(self, bin_fn, load_mm=False):
        with open(bin_fn, 'rb') as f:
            raw_data = f.read()
            data_tmp = np.frombuffer(raw_data, dtype=np.float64)
            if self.modality == 'mmwave':
                data_tmp = data_tmp.copy().reshape(-1, 5)
                #seems like 3 is intensity, 4 is velocity, flip them
                data_tmp[:, [3, 4]] = data_tmp[:, [4, 3]]
            elif load_mm:
                data_tmp = data_tmp.copy().reshape(-1, 5)
                data_tmp[:, [3, 4]] = data_tmp[:, [4, 3]]
            else:
                data_tmp = data_tmp.copy().reshape(-1, 3)
        return data_tmp

    def _filter_pcl(self, bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, offset: float = 0):
        """
        Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
        """
        upper_bound = bounding_pcl[:, :3].max(axis=0) + bound
        lower_bound = bounding_pcl[:, :3].min(axis=0) - bound
        lower_bound[2] += offset

        mask_x = (target_pcl[:, 0] >= lower_bound[0]) & (
            target_pcl[:, 0] <= upper_bound[0])
        mask_y = (target_pcl[:, 1] >= lower_bound[1]) & (
            target_pcl[:, 1] <= upper_bound[1])
        mask_z = (target_pcl[:, 2] >= lower_bound[2]) & (
            target_pcl[:, 2] <= upper_bound[2])
        index = mask_x & mask_y & mask_z
        return target_pcl[index]

class MMBodyPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir):
        super().__init__(root_dir, out_dir)

    def process(self):
        train_val_dirs = glob(os.path.join(self.root_dir, 'train/sequence_*'))
        test_dirs = glob(os.path.join(self.root_dir, 'test/*/sequence_*'))

        train_val_seq_idxs = np.arange(len(train_val_dirs))
        np.random.shuffle(train_val_seq_idxs)
        num_train = int(len(train_val_seq_idxs) * 0.9)
        self.results['splits']['train'] = train_val_seq_idxs[:num_train]
        self.results['splits']['val'] = train_val_seq_idxs[num_train:]
        self.results['splits']['test'] = np.arange(len(test_dirs)) + len(train_val_dirs)

        for d in tqdm(train_val_dirs + test_dirs):
            pcs = []
            kps = []
            pc_fns = glob(os.path.join(d, 'radar', '*.npy'))
            bns = sorted([int(os.path.basename(fn).split('.')[0].split('_')[-1]) for fn in pc_fns])
            for bn in bns:
                pc = np.load(os.path.join(d, 'radar', f'frame_{bn}.npy'))
                pc[:,3:] /= np.array([5e-38, 5., 1.])
                pc[:, -1] = self._normalize_intensity(pc[:, -1], 150.0)
                kp = np.load(os.path.join(d, 'mesh', f'frame_{bn}.npz'))['joints'][:22]
                pc = self._filter_pcl(kp, pc)
                if len(pc) == 0:
                    continue
                pcs.append(pc[:, [0, 2, 1, 4, 5]])
                kps.append(kp[:, [0, 2, 1]])
            self.results['sequences'].append({
                'point_clouds': pcs,
                'keypoints': kps,
                'action': -1
            })

    def save(self):
        super().save('mmbody')

    def _filter_pcl(self, bounding_pcl: np.ndarray, target_pcl: np.ndarray, bound: float = 0.2, offset: float = 0):
        """
        Filter out the pcls of pcl_b that is not in the bounding_box of pcl_a
        """
        upper_bound = bounding_pcl[:, :3].max(axis=0) + bound
        lower_bound = bounding_pcl[:, :3].min(axis=0) - bound
        lower_bound[2] += offset

        mask_x = (target_pcl[:, 0] >= lower_bound[0]) & (
            target_pcl[:, 0] <= upper_bound[0])
        mask_y = (target_pcl[:, 1] >= lower_bound[1]) & (
            target_pcl[:, 1] <= upper_bound[1])
        mask_z = (target_pcl[:, 2] >= lower_bound[2]) & (
            target_pcl[:, 2] <= upper_bound[2])
        index = mask_x & mask_y & mask_z
        return target_pcl[index]

class MRIPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir):
        super().__init__(root_dir, out_dir)

    def process(self):
        sub_idxs = np.arange(1, 21)
        shuffled_sub_idxs = sub_idxs.copy()
        np.random.shuffle(shuffled_sub_idxs)
        train_sub_idxs = shuffled_sub_idxs[:14]
        val_sub_idxs = shuffled_sub_idxs[14:16]

        count = 0
        for idx in tqdm(sub_idxs):
            pc_fn = os.path.join(self.root_dir, f'dataset_release/aligned_data/radar/singleframe/subject{idx}.csv')
            label_fn = os.path.join(self.root_dir, f'dataset_release/aligned_data/pose_labels/subject{idx}_all_labels.cpl')
            pc_df = pd.read_csv(pc_fn)
            with open(label_fn, 'rb') as f:
                labels = pickle.load(f)
            splits = list(labels['video_label'].values())[1:13]
            for i, split in enumerate(splits):
                pcs = []
                for j in range(split[0], split[1]):
                    pc = pc_df[pc_df['Camera Frame'] == j][['X', 'Y', 'Z', 'Doppler', 'Intensity']].values
                    pc[:, -1] = self._normalize_intensity(pc[:, -1], 200.0)
                    pc = pc[:, [0, 2, 1, 3, 4]]
                    pc[:, 0] *= -1
                    pcs.append(pc)
                kps = labels['refined_gt_kps'][split[0]:split[1]].transpose(0, 2, 1)
                self.results['sequences'].append({
                    'point_clouds': pcs,
                    'keypoints': kps,
                    'action': i
                })
                if idx in train_sub_idxs:
                    self._add_to_split('train_s2_p1', count)
                elif idx in val_sub_idxs:
                    self._add_to_split('val_s2_p1', count)
                else:
                    self._add_to_split('test_s2_p1', count)
                count += 1

        seq_idxs = np.arange(len(self.results['sequences']))
        np.random.shuffle(seq_idxs)
        num_train = int(len(seq_idxs) * 0.7)
        num_val = int(len(seq_idxs) * 0.1)
        self.results['splits']['train_s1_p1'] = sorted(seq_idxs[:num_train])
        self.results['splits']['val_s1_p1'] = sorted(seq_idxs[num_train:num_train+num_val])
        self.results['splits']['test_s1_p1'] = sorted(seq_idxs[num_train+num_val:])

        for i in tqdm(seq_idxs):
            if i in self.results['splits']['train_s1_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('train_s1_p2', i)
            elif i in self.results['splits']['val_s1_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('val_s1_p2', i)
            elif i in self.results['splits']['test_s1_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('test_s1_p2', i)

            if i in self.results['splits']['train_s2_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('train_s2_p2', i)
            elif i in self.results['splits']['val_s2_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('val_s2_p2', i)
            elif i in self.results['splits']['test_s2_p1'] and self.results['sequences'][i]['action'] < 10:
                self._add_to_split('test_s2_p2', i)

    def save(self):
        super().save('mri')

class ITOPPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir, view):
        super().__init__(root_dir, out_dir)
        assert view in ['side', 'top'], 'Invalid view'
        self.view = view

    def process(self):
        train_val_data_fn = os.path.join(self.root_dir, f'ITOP_{self.view}_train_point_cloud.h5')
        train_val_labels_fn = os.path.join(self.root_dir, f'ITOP_{self.view}_train_labels.h5')
        test_data_fn = os.path.join(self.root_dir, f'ITOP_{self.view}_test_point_cloud.h5')
        test_labels_fn = os.path.join(self.root_dir, f'ITOP_{self.view}_test_labels.h5')

        train_val_data, train_val_ids = self._read_h5(train_val_data_fn, ['data', 'id'])
        train_val_labels = self._read_h5(train_val_labels_fn, ['is_valid', 'real_world_coordinates', 'segmentation'])

        test_data, test_ids = self._read_h5(test_data_fn, ['data', 'id'])
        test_labels = self._read_h5(test_labels_fn, ['is_valid', 'real_world_coordinates', 'segmentation'])

        train_val_list = self._process_split(train_val_data, train_val_ids, train_val_labels)
        test_list = self._process_split(test_data, test_ids, test_labels)

        train_val_seq_idxs = np.arange(len(train_val_list))
        np.random.shuffle(train_val_seq_idxs)
        num_train = int(len(train_val_seq_idxs) * 0.8)
        self.results['splits']['train'] = train_val_seq_idxs[:num_train]
        self.results['splits']['val'] = train_val_seq_idxs[num_train:]
        self.results['splits']['test'] = np.arange(len(test_list)) + len(train_val_list)

        for d in train_val_list + test_list:
            self.results['sequences'].append(d)

    def save(self):
        super().save('itop_' + self.view)

    def _read_h5(self, fn, keys):
        output = []
        f = h5py.File(fn, 'r')
        for key in keys:
            value = f[key][()]
            output.append(value)
        f.close()
        return tuple(output)
    
    def _segment_human(self, pc, seg):
        pc = pc[np.ravel(seg) != -1]
        return pc
    
    def _process_split(self, data, ids, labels):
        split_list = []
        last_id = None
        for pc, id, valid, kps, seg in tqdm(zip(data, ids, labels[0], labels[1], labels[2])):
            if valid == 0 and (last_id is None or len(split_list[-1]['point_clouds']) > 0):
                split_list.append({
                    'point_clouds': [],
                    'keypoints': [],
                    'action': -1
                })
            else:
                if id.decode().split('_')[0] != last_id and (last_id is None or len(split_list[-1]['point_clouds']) > 0):
                    split_list.append({
                        'point_clouds': [],
                        'keypoints': [],
                        'action': -1
                    })
                pc = self._segment_human(pc, seg)
                if len(pc) == 76800:
                    print('Skipping due to invalid segmentation')
                    continue
                
                if self.view == 'top':
                    pc = pc[..., [0, 2, 1]] * np.array([1, -1, 1])
                    kps = kps[..., [0, 2, 1]] * np.array([1, -1, 1])

                split_list[-1]['point_clouds'].append(pc)
                split_list[-1]['keypoints'].append(kps)
                last_id = id.decode().split('_')[0]

        split_list = [d for d in split_list if len(d['point_clouds']) >= 5]
        return split_list

class LiDARHuman26MPreprocessor(Preprocessor):
    def __init__(self, root_dir, out_dir):
        super().__init__(root_dir, out_dir)
        self.pc_dir = os.path.join(self.root_dir, 'labels/3d/segment')
        self.kps_dir = os.path.join(self.root_dir, 'labels/3d/pose')

    def process(self):
        with open(os.path.join(self.root_dir, 'train.txt')) as f:
            train_val_ids = f.read().splitlines()
        with open(os.path.join(self.root_dir, 'test.txt')) as f:
            test_ids = f.read().splitlines()

        np.random.shuffle(train_val_ids)
        train_ids = train_val_ids[:int(len(train_val_ids)*0.8)]
        val_ids = train_val_ids[int(len(train_val_ids)*0.8):]

        self._process_split(train_ids, 'train')
        self._process_split(val_ids, 'val')
        self._process_split(test_ids, 'test')

    def _read_point_cloud(self, filename):
        """ read XYZ point cloud from filename PLY file """
        ply_data = PlyData.read(filename)['vertex'].data
        points = np.array([[y, z, x] for x, y, z in ply_data])
        return points
    
    def _read_json(self, filename):
        with open(filename) as f:
            data = json.load(f)
        pose = np.array(data['pose'], dtype=np.float32)
        beta = np.array(data['beta'], dtype=np.float32)
        trans = np.array(data['trans'], dtype=np.float32)
        return pose, beta, trans

    def _process_split(self, ids, split_name):
        for id in tqdm(ids):
            pc_fns = sorted(glob(os.path.join(self.pc_dir, str(id), '*.ply')))

            pcs = []
            poses = []
            betas = []
            transs = []
            for pc_fn in pc_fns:
                kp_fn = pc_fn.replace('segment', 'pose').replace('ply', 'json')
                pc = self._read_point_cloud(pc_fn)
                pose, beta, trans = self._read_json(kp_fn)
                pcs.append(pc)
                poses.append(pose)
                betas.append(beta)
                transs.append(trans)
            poses = np.stack(poses)
            betas = np.stack(betas)
            transs = np.stack(transs)

            smpl = SMPL().cuda()

            batch_size = 8192
            num_batches = len(pcs) // batch_size
            if len(pcs) % batch_size != 0:
                num_batches += 1
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(pcs))
                verts = smpl(torch.from_numpy(poses[start:end]).cuda(), torch.from_numpy(betas[start:end]).cuda())
                kps = smpl.get_full_joints(verts).cpu().numpy()
                kps += transs[start:end, np.newaxis, :]
                if i == 0:
                    all_kps = kps
                else:
                    all_kps = np.concatenate([all_kps, kps], axis=0)
            # all_kps += transs[:, np.newaxis, :]

            self.results['sequences'].append({
                'point_clouds': pcs,
                'keypoints': all_kps[..., [1, 2, 0]],
                'action': -1
            })

            self._add_to_split(split_name, len(self.results['sequences']) - 1)

    def save(self):
        super().save('lidarhuman26m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, default='mmfi', choices=['mili', 'mmfi', 'mmbody', 'mri'])
    parser.add_argument('--root_dir', type=str, default='/home/ryan/MM-Fi/MMFi_Dataset', help='Path to the root directory of the dataset')
    parser.add_argument('--out_dir', type=str, default='/home/ryan/MM-Fi/LEMT/data_dual', help='Path to the output directory')
    parser.add_argument('--modality', type=str, default='dual', choices=['mmwave', 'lidar', 'dual', 'raw_dual', 'roi_bg_dual', 'bg_only_dual'])
    parser.add_argument('--localization_checkpoint', type=str, default=None, help='Path to trained localization checkpoint')
    parser.add_argument('--localization_model_name', type=str, default=None, help='Override localization model name')
    parser.add_argument('--localization_model_params', type=str, default=None, help='JSON string for localization model params')
    parser.add_argument('--split_mode', type=str, default='random', choices=['random', 'fixed_seperation'], help='MMFi split mode')
    args = parser.parse_args()

    if args.modality == 'roi_bg_dual' or args.modality == 'bg_only_dual':
        if not args.cfg:
            raise ValueError("roi_bg_dual and bg_only_dual require --cfg with localization settings")
        cfg = load_cfg(args.cfg)
        args = merge_args_cfg(args, cfg)

    localization_model_params = None
    if args.modality == 'roi_bg_dual' and args.localization_model_params:
        if isinstance(args.localization_model_params, str):
            localization_model_params = json.loads(args.localization_model_params)
        else:
            localization_model_params = args.localization_model_params
    args.seed = 0 #0 for MMFi
    if args.dataset == 'mmfi':
        preprocessor = MMFiPreprocessor(
            args.root_dir,
            args.out_dir,
            args.modality,
            localization_checkpoint=args.localization_checkpoint,
            localization_model_name=args.localization_model_name,
            localization_model_params=localization_model_params,
            localization_input=getattr(args, 'localization_input', 'mmwave'),
            split_mode=args.split_mode,
            seed = args.seed,
            background_downsample_rate=getattr(args, 'background_downsample_rate', None),
            background_update_rate=getattr(args, 'background_update_rate', None)
        )
    elif args.dataset == 'mili':
        preprocessor = MiliPointPreprocessor(args.root_dir, args.out_dir)
    elif args.dataset == 'mmbody':
        preprocessor = MMBodyPreprocessor(args.root_dir, args.out_dir)
    elif args.dataset == 'mri':
        preprocessor = MRIPreprocessor(args.root_dir, args.out_dir)
    elif args.dataset == 'itop_side':
        preprocessor = ITOPPreprocessor(args.root_dir, args.out_dir, 'side')
    elif args.dataset == 'itop_top':
        preprocessor = ITOPPreprocessor(args.root_dir, args.out_dir, 'top')
    elif args.dataset == 'lidarhuman26m':
        preprocessor = LiDARHuman26MPreprocessor(args.root_dir, args.out_dir)
    else:
        raise ValueError('Invalid dataset name')
    
    preprocessor.process()
    #check the mmwave data of preprocessor's result for seq in preprocessor.results['sequences'][:2]:
    # seq = preprocessor.results['sequences'][0]
    # if 'mmwave_data' in seq:
    #     #check the min, max of velocity and intensity of mmwave data
    #     velocities = []
    #     intensities = []
    #     for mmwave in seq['mmwave_data']:
    #         velocities.append(mmwave[:, 3])
    #         intensities.append(mmwave[:, 4])
    #     velocities = np.concatenate(velocities)
    #     intensities = np.concatenate(intensities)
    #     print(f'mmWave velocity: min {velocities.min()}, max {velocities.max()}')
    #     print(f'mmWave intensity: min {intensities.min()}, max {intensities.max()}')
        
    preprocessor.save()