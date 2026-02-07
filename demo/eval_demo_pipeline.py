import argparse
import json
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def extract_roi(point_cloud, predicted_location, edge_length):
    if point_cloud is None or point_cloud.shape[0] == 0:
        return np.zeros((0, point_cloud.shape[1] if point_cloud is not None else 3))
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
    from sklearn.neighbors import NearestNeighbors
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
        return np.zeros((0, lidar_point_cloud.shape[1] if lidar_point_cloud is not None else 3))
    from scipy.spatial import cKDTree
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

    from sklearn.neighbors import NearestNeighbors
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

    keep_indices = np.random.choice(num_points, size=max_points, replace=False)
    return point_cloud[keep_indices]


def get_centroid(point_cloud, centroid_type='median'):
    if point_cloud is None or point_cloud.shape[0] == 0:
        return np.zeros(3)
    pc_dedupe = np.unique(point_cloud[:, :3], axis=0)
    if pc_dedupe.shape[0] == 0:
        return np.zeros(3)
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


def create_background(raw_lidar_pc, hpe_output, buffer, num_points):
    if raw_lidar_pc is None or raw_lidar_pc.shape[0] == 0:
        return np.zeros((num_points, 3))
    min_coords = np.min(hpe_output, axis=0)
    max_coords = np.max(hpe_output, axis=0)
    min_bound = min_coords - buffer
    max_bound = max_coords + buffer
    lidar_xyz = raw_lidar_pc[:, :3]
    outside_bounds_mask = np.any((lidar_xyz < min_bound) | (lidar_xyz > max_bound), axis=1)
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


def mpjpe(pred, gt):
    return np.mean(np.linalg.norm(pred - gt, axis=1))


def pa_mpjpe(pred, gt):
    pred_centered = pred - np.mean(pred, axis=0)
    gt_centered = gt - np.mean(gt, axis=0)
    h = gt_centered.T @ pred_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[2, :] *= -1
        r = vt.T @ u.T
    pred_aligned = pred_centered @ r
    return np.mean(np.linalg.norm(pred_aligned - gt_centered, axis=1))


def _point_dist(a, b):
    return float(np.linalg.norm(a - b))


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_models(cfg, device):
    loc_model = create_model(_cfg_get(cfg, 'localization_model_name'), _cfg_get(cfg, 'localization_model_params'))
    loc_ckpt = torch.load(_cfg_get(cfg, 'localization_checkpoint'), map_location=device)
    loc_state = {k.replace('model.', ''): v for k, v in loc_ckpt['state_dict'].items() if k.startswith('model.')}
    loc_model.load_state_dict(loc_state, strict=False)
    loc_model.to(device)
    loc_model.eval()
    for p in loc_model.parameters():
        p.requires_grad = False

    hpe_model = create_model(_cfg_get(cfg, 'model_name'), _cfg_get(cfg, 'model_params'))
    hpe_ckpt = torch.load(_cfg_get(cfg, 'model_checkpoint'), map_location=device)
    hpe_state = {k.replace('model.', ''): v for k, v in hpe_ckpt['state_dict'].items() if k.startswith('model.')}
    hpe_model.load_state_dict(hpe_state, strict=False)
    hpe_model.to(device)
    hpe_model.eval()
    for p in hpe_model.parameters():
        p.requires_grad = False

    return loc_model, hpe_model


def evaluate_sequence(seq, loc_model, hpe_model, device):
    background_points = None
    previous_mmwave_frames = []
    previous_kps_frames = []
    previous_filtered_lidar_frames = []
    previous_extracted_mmwave_frames = []
    last_pred_kp7_abs = None

    clip_len = 5
    max_points = 128
    num_background_points = 2048
    num_new_background_points = 1024

    lidar_data = seq['point_clouds']
    mmwave_data = seq.get('mmwave_data')
    keypoints_data = seq['keypoints']

    if mmwave_data is None:
        raise ValueError("Sequence missing mmwave_data")

    mpjpe_sum = 0.0
    pampjpe_sum = 0.0
    count = 0
    per_frame_mpjpe = []
    per_frame_pampjpe = []
    per_frame_mm_counts = []
    per_frame_loc_err = []
    per_frame_centroid_err = []

    for frame_idx in range(len(lidar_data)):
        per_frame_mm_counts.append(int(mmwave_data[frame_idx].shape[0]))
        previous_mmwave_frames.append(mmwave_data[frame_idx])
        if len(previous_mmwave_frames) > clip_len:
            previous_mmwave_frames.pop(0)

        previous_kps_frames.append(keypoints_data[frame_idx])
        if len(previous_kps_frames) > clip_len:
            previous_kps_frames.pop(0)

        if len(previous_mmwave_frames) < clip_len:
            fill_frames = [previous_mmwave_frames[0]] * (clip_len - len(previous_mmwave_frames))
            current_mmwave_frames = fill_frames + previous_mmwave_frames
        else:
            current_mmwave_frames = previous_mmwave_frames

        if len(previous_kps_frames) < clip_len:
            fill_kps = [previous_kps_frames[0]] * (clip_len - len(previous_kps_frames))
            current_kps_frames = fill_kps + previous_kps_frames
        else:
            current_kps_frames = previous_kps_frames

        keypoint = get_centroid(np.concatenate(current_mmwave_frames, axis=0), centroid_type='median')

        normalized_mmwave_frames = []
        for mm_frame in current_mmwave_frames:
            mm_norm = normalize(mm_frame, keypoint)
            mm_norm = remove_outliers_box(mm_norm, radius=3.0)
            mm_norm = pad_point_cloud(mm_norm, 128)
            normalized_mmwave_frames.append(mm_norm)

        mmwave_count = int(mmwave_data[frame_idx].shape[0]) if mmwave_data[frame_idx] is not None else 0
        if mmwave_count < 3 and last_pred_kp7_abs is not None:
            predicted_location_abs = last_pred_kp7_abs.copy()
            predicted_location = predicted_location_abs - keypoint
        else:
            mmwave_input_numpy = prepare_hpe_input(normalized_mmwave_frames)
            mmwave_input_tensor = torch.from_numpy(mmwave_input_numpy).float().to(device)

            with torch.no_grad():
                predicted_location_tensor = loc_model(mmwave_input_tensor)
            predicted_location = predicted_location_tensor.detach().cpu().numpy().squeeze()
            predicted_location_abs = predicted_location + keypoint
        gt_loc = keypoints_data[frame_idx][7]
        per_frame_loc_err.append(_point_dist(predicted_location_abs, gt_loc))

        extracted_mmwave = extract_roi(mmwave_data[frame_idx], predicted_location_abs, edge_length=2.5)
        extracted_lidar = extract_roi(lidar_data[frame_idx], predicted_location_abs, edge_length=2.5)

        previous_extracted_mmwave_frames.append(extracted_mmwave)
        if len(previous_extracted_mmwave_frames) > clip_len:
            previous_extracted_mmwave_frames.pop(0)
        if len(previous_extracted_mmwave_frames) < clip_len:
            fill_extracted_mmwave = [previous_extracted_mmwave_frames[0]] * (clip_len - len(previous_extracted_mmwave_frames))
            current_extracted_mmwave_frames = fill_extracted_mmwave + previous_extracted_mmwave_frames
        else:
            current_extracted_mmwave_frames = previous_extracted_mmwave_frames

        if background_points is not None:
            filtered_lidar = remove_background(extracted_lidar, background_points, buffer=0.1)
        else:
            filtered_lidar = extracted_lidar

        previous_filtered_lidar_frames.append(filtered_lidar)
        if len(previous_filtered_lidar_frames) > clip_len:
            previous_filtered_lidar_frames.pop(0)
        if len(previous_filtered_lidar_frames) < clip_len:
            fill_filtered_lidar = [previous_filtered_lidar_frames[0]] * (clip_len - len(previous_filtered_lidar_frames))
            current_filtered_lidar_frames = fill_filtered_lidar + previous_filtered_lidar_frames
        else:
            current_filtered_lidar_frames = previous_filtered_lidar_frames

        lidar_cat = np.concatenate(current_filtered_lidar_frames, axis=0) if len(current_filtered_lidar_frames) > 0 else np.zeros((0, 3))
        if lidar_cat.shape[0] > 0:
            hpe_centroid = np.array([
                np.median(lidar_cat[:, 0]),
                np.min(lidar_cat[:, 1]),
                np.median(lidar_cat[:, 2])
            ])
        else:
            hpe_centroid = predicted_location_abs.copy()
        per_frame_centroid_err.append(_point_dist(hpe_centroid, gt_loc))

        normalized_gt_hpe = normalize(keypoints_data[frame_idx], hpe_centroid)

        hpe_frames = []
        for pc_frame, mm_frame in zip(current_filtered_lidar_frames, current_extracted_mmwave_frames):
            pc_norm = normalize(pc_frame, hpe_centroid)
            mm_norm = normalize(mm_frame, hpe_centroid)
            pc_norm = remove_outliers_box(pc_norm, radius=1.5)
            mm_norm = remove_outliers_box(mm_norm, radius=1.5)
            pc_norm = remove_outliers_radius(pc_norm, radius=0.15, min_neighbors=3)
            pc_feat = feature_transfer(pc_norm, mm_norm, mode='mmwave', knn_k=3)
            pc_padded = pad_point_cloud(pc_feat, max_points)
            hpe_frames.append(pc_padded)

        if len(hpe_frames) < clip_len:
            while len(hpe_frames) < clip_len:
                hpe_frames.insert(0, hpe_frames[0])

        HPE_input_numpy = prepare_hpe_input(hpe_frames)
        HPE_input_tensor = torch.from_numpy(HPE_input_numpy).float().to(device)

        with torch.no_grad():
            hpe_output_tensor = hpe_model(HPE_input_tensor)
        hpe_output = hpe_output_tensor.detach().cpu().numpy().squeeze()

        frame_mpjpe = mpjpe(hpe_output, normalized_gt_hpe) * 1000.0
        frame_pampjpe = pa_mpjpe(hpe_output, normalized_gt_hpe) * 1000.0
        mpjpe_sum += frame_mpjpe
        pampjpe_sum += frame_pampjpe
        count += 1
        per_frame_mpjpe.append(frame_mpjpe)
        per_frame_pampjpe.append(frame_pampjpe)

        hpe_output_vis = hpe_output + hpe_centroid
        if hpe_output_vis.shape[0] > 7:
            last_pred_kp7_abs = hpe_output_vis[7].copy()
        new_background = create_background(lidar_data[frame_idx], hpe_output_vis, buffer=0.2, num_points=num_new_background_points)
        background_points = update_background(background_points, new_background, num_background_points)

    return (
        mpjpe_sum,
        pampjpe_sum,
        count,
        per_frame_mpjpe,
        per_frame_pampjpe,
        per_frame_mm_counts,
        per_frame_loc_err,
        per_frame_centroid_err,
    )


def _plot_metric(frame_means, title, ylabel, output_path):
    x_vals = np.arange(len(frame_means))
    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, frame_means, linewidth=1.5)
    plt.xlabel('Frame Number')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_counts(frame_counts, output_path):
    x_vals = np.arange(len(frame_counts))
    plt.figure(figsize=(10, 3.5))
    plt.plot(x_vals, frame_counts, linewidth=1.2)
    plt.xlabel('Frame Number')
    plt.ylabel('Samples')
    plt.title('Samples per Frame')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to demo cfg with checkpoints')
    parser.add_argument('--pkl_path', type=str, required=True, help='Path to MMFi pkl (raw_dual)')
    parser.add_argument('--split', type=str, default='test_rdn_p3', help='Split name in pkl')
    parser.add_argument('--max_seqs', type=int, default=None, help='Limit number of sequences for quick tests')
    parser.add_argument('--skip_seqs', type=str, default='', help='Comma-separated sequence indices to skip')
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loc_model, hpe_model = load_models(cfg, device)

    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    split_indices = data['splits'][args.split]
    if isinstance(split_indices, np.ndarray):
        split_indices = split_indices.tolist()

    skip_seqs = []
    if args.skip_seqs:
        skip_seqs = [int(s.strip()) for s in args.skip_seqs.split(',') if s.strip()]

    if args.max_seqs is not None:
        split_indices = split_indices[:args.max_seqs]

    total_mpjpe = 0.0
    total_pampjpe = 0.0
    total_count = 0
    mpjpe_by_frame = {}
    pampjpe_by_frame = {}
    mm_counts_by_frame = {}
    loc_err_by_frame = {}
    centroid_err_by_frame = {}
    outlier_rows = []

    for seq_idx in split_indices:
        if seq_idx in skip_seqs:
            continue
        seq = data['sequences'][seq_idx]
        (mpjpe_sum, pampjpe_sum, count,
         per_frame_mpjpe, per_frame_pampjpe,
         per_frame_mm_counts, per_frame_loc_err,
         per_frame_centroid_err) = evaluate_sequence(seq, loc_model, hpe_model, device)
        total_mpjpe += mpjpe_sum
        total_pampjpe += pampjpe_sum
        total_count += count
        for frame_idx, val in enumerate(per_frame_mpjpe):
            mpjpe_by_frame.setdefault(frame_idx, []).append(val)
        for frame_idx, val in enumerate(per_frame_pampjpe):
            pampjpe_by_frame.setdefault(frame_idx, []).append(val)
        for frame_idx, val in enumerate(per_frame_mm_counts):
            mm_counts_by_frame.setdefault(frame_idx, []).append(val)
        for frame_idx, val in enumerate(per_frame_loc_err):
            loc_err_by_frame.setdefault(frame_idx, []).append(val)
        for frame_idx, val in enumerate(per_frame_centroid_err):
            centroid_err_by_frame.setdefault(frame_idx, []).append(val)

        for frame_idx in range(len(per_frame_mpjpe)):
            outlier_rows.append({
                'seq_idx': seq_idx,
                'frame_idx': frame_idx,
                'mpjpe': per_frame_mpjpe[frame_idx],
                'pampjpe': per_frame_pampjpe[frame_idx],
                'mmwave_points': per_frame_mm_counts[frame_idx],
                'loc_err': per_frame_loc_err[frame_idx],
                'centroid_err': per_frame_centroid_err[frame_idx],
            })

    if total_count == 0:
        print('No frames evaluated.')
        return

    print('==================== FINAL RESULTS ====================')
    print(f'MPJPE (Mean Per Joint Position Error): {total_mpjpe / total_count:.2f} mm')
    print(f'PA-MPJPE (Procrustes-Aligned MPJPE): {total_pampjpe / total_count:.2f} mm')
    print('=====================================================')

    if mpjpe_by_frame:
        max_frame = max(mpjpe_by_frame.keys())
        mpjpe_means = []
        pampjpe_means = []
        frame_counts = []
        mpjpe_stds = []
        pampjpe_stds = []
        mm_counts_means = []
        loc_err_means = []
        centroid_err_means = []
        for frame_idx in range(max_frame + 1):
            mpjpe_vals = mpjpe_by_frame.get(frame_idx, [])
            pampjpe_vals = pampjpe_by_frame.get(frame_idx, [])
            mm_vals = mm_counts_by_frame.get(frame_idx, [])
            loc_vals = loc_err_by_frame.get(frame_idx, [])
            centroid_vals = centroid_err_by_frame.get(frame_idx, [])
            frame_counts.append(len(mpjpe_vals))
            mpjpe_means.append(float(np.mean(mpjpe_vals)) if mpjpe_vals else 0.0)
            pampjpe_means.append(float(np.mean(pampjpe_vals)) if pampjpe_vals else 0.0)
            mpjpe_stds.append(float(np.std(mpjpe_vals)) if mpjpe_vals else 0.0)
            pampjpe_stds.append(float(np.std(pampjpe_vals)) if pampjpe_vals else 0.0)
            mm_counts_means.append(float(np.mean(mm_vals)) if mm_vals else 0.0)
            loc_err_means.append(float(np.mean(loc_vals)) if loc_vals else 0.0)
            centroid_err_means.append(float(np.mean(centroid_vals)) if centroid_vals else 0.0)

        if max_frame >= 20:
            mpjpe_tail = [v for v in mpjpe_means[20:] if v > 0.0]
            pampjpe_tail = [v for v in pampjpe_means[20:] if v > 0.0]
            if mpjpe_tail and pampjpe_tail:
                print('==================== FRAME >= 20 ====================')
                print(f'MPJPE (Frame >= 20): {float(np.mean(mpjpe_tail)):.2f} mm')
                print(f'PA-MPJPE (Frame >= 20): {float(np.mean(pampjpe_tail)):.2f} mm')
                print('=====================================================')

        output_dir = os.path.join(os.path.dirname(__file__), 'display_images')
        os.makedirs(output_dir, exist_ok=True)
        _plot_metric(
            mpjpe_means,
            'MPJPE by Frame (Average)',
            'MPJPE (mm)',
            os.path.join(output_dir, 'mpjpe_by_frame.png')
        )
        _plot_metric(
            pampjpe_means,
            'PA-MPJPE by Frame (Average)',
            'PA-MPJPE (mm)',
            os.path.join(output_dir, 'pampjpe_by_frame.png')
        )
        _plot_counts(
            frame_counts,
            os.path.join(output_dir, 'samples_by_frame.png')
        )
        _plot_metric(
            mm_counts_means,
            'Average mmWave Points by Frame',
            'mmWave Points',
            os.path.join(output_dir, 'mmwave_points_by_frame.png')
        )
        _plot_metric(
            loc_err_means,
            'Localization Error by Frame (Average)',
            'Error (m)',
            os.path.join(output_dir, 'localization_error_by_frame.png')
        )
        _plot_metric(
            centroid_err_means,
            'Centroid Error by Frame (Average)',
            'Error (m)',
            os.path.join(output_dir, 'centroid_error_by_frame.png')
        )

        csv_path = os.path.join(output_dir, 'metrics_by_frame.csv')
        with open(csv_path, 'w', encoding='ascii') as f:
            f.write('frame,mpjpe_mean,mpjpe_std,pampjpe_mean,pampjpe_std,count,mmwave_points_mean,loc_err_mean,centroid_err_mean\n')
            for frame_idx in range(max_frame + 1):
                f.write(
                    f"{frame_idx},{mpjpe_means[frame_idx]:.6f},{mpjpe_stds[frame_idx]:.6f},"
                    f"{pampjpe_means[frame_idx]:.6f},{pampjpe_stds[frame_idx]:.6f},"
                    f"{frame_counts[frame_idx]},{mm_counts_means[frame_idx]:.6f},"
                    f"{loc_err_means[frame_idx]:.6f},{centroid_err_means[frame_idx]:.6f}\n"
                )

        outlier_rows.sort(key=lambda r: r['mpjpe'], reverse=True)
        outlier_path = os.path.join(output_dir, 'outliers_by_mpjpe.csv')
        with open(outlier_path, 'w', encoding='ascii') as f:
            f.write('seq_idx,frame_idx,mpjpe,pampjpe,mmwave_points,loc_err,centroid_err\n')
            for row in outlier_rows[:200]:
                f.write(
                    f"{row['seq_idx']},{row['frame_idx']},{row['mpjpe']:.6f},"
                    f"{row['pampjpe']:.6f},{row['mmwave_points']},"
                    f"{row['loc_err']:.6f},{row['centroid_err']:.6f}\n"
                )


if __name__ == '__main__':
    main()
