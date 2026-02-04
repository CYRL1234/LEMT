import numpy as np
import torch
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN
from miniball import get_bounding_ball

class AddNoisyPoints():
    def __init__(self, add_std=0.01, num_added=32, zero_centered=True, center_range=1.5):
        self.add_std = add_std
        self.num_added = num_added
        self.zero_centered = zero_centered
        self.center_range = center_range

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            if self.zero_centered:
                noise = np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][i].shape[1]))
            else:
                noise_center = np.random.uniform(-self.center_range, self.center_range, sample['point_clouds'][i].shape[1])
                noise = np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][i].shape[1])) + noise_center
            sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i], noise], axis=0)
        
        return sample
    
class AddPointsAroundJoint():
    def __init__(self, add_std=0.1, max_num2add=1, num_added=32):
        self.add_std = add_std
        self.max_num2add = max_num2add
        self.num_added = num_added

    def __call__(self, sample):
        num_joints = sample['keypoints'][0].shape[0]
        num2add = np.random.randint(1, self.max_num2add)
        idxs2add = np.random.choice(num_joints, num2add, replace=False)

        new_pcs = []
        for i in range(len(sample['keypoints'])):
            pc = sample['point_clouds'][i]
            for idx in idxs2add:
                add_point = sample['keypoints'][i][idx]
                if add_point.shape[-1] < pc.shape[-1]:
                    add_point = np.concatenate([add_point, np.zeros(pc.shape[-1]-add_point.shape[-1])], axis=-1)
                add_points = add_point[np.newaxis, :].repeat(self.num_added, axis=0) + np.random.normal(0, self.add_std, (self.num_added, sample['point_clouds'][-1].shape[-1]))
                pc = np.concatenate([pc, add_points], axis=0)
            new_pcs.append(pc)

        sample['point_clouds'] = new_pcs
        return sample
    
class RemoveOutliers():
    def __init__(self, outlier_type='statistical', num_neighbors=3, std_multiplier=1.0, radius=1.0, min_neighbors=2, data_type='pointcloud'):
        self.outlier_type = outlier_type
        self.num_neighbors = num_neighbors
        self.std_multiplier = std_multiplier
        self.radius = radius
        self.min_neighbors = min_neighbors
        self.data_type = data_type
        if outlier_type not in ['statistical', 'radius', 'cluster', 'box']:
            raise ValueError('outlier_type must be "statistical" or "radius" or "cluster" or "box"')

    def __call__(self, sample):
        if self.data_type == 'pointcloud':
            data_key = 'point_clouds'
        elif self.data_type == 'mmwave':
            data_key = 'mmwave_data'
        for i in range(len(sample[data_key])):
            if sample[data_key][i] is None or sample[data_key][i].shape[0] == 0:
                continue

            if self.outlier_type == 'statistical':
                neighbors = NearestNeighbors(n_neighbors=self.num_neighbors+1).fit(sample[data_key][i][...,:3])
                distances, _ = neighbors.kneighbors(sample[data_key][i][...,:3])
                mean_dist = np.mean(distances[:, 1:], axis=1)
                std_dist = np.std(distances[:, 1:], axis=1)
                dist_threshold = mean_dist + self.std_multiplier * std_dist
                inliers = np.where(distances[:, 1:] < dist_threshold[:, np.newaxis])

            elif self.outlier_type == 'radius':
                neighbors = NearestNeighbors(radius=self.radius).fit(sample[data_key][i][...,:3])
                distances, _ = neighbors.radius_neighbors(sample[data_key][i][...,:3], return_distance=True)
                inliers = np.where([len(d) >= self.min_neighbors for d in distances])

            elif self.outlier_type == 'cluster':
                clusterer = DBSCAN(min_samples=self.min_neighbors)
                inliers = clusterer.fit_predict(sample[data_key][i][...,:3]) != -1
                if np.sum(inliers) == 0:
                    inliers[0] = True

            # elif self.outlier_type == 'box':
            #     inliers = np.where(np.all(np.abs(sample[data_key][i][...,:2] - np.array([[0, 1]])) < self.radius, axis=1))
            #     if 'mmwave_data' in sample:
            #         inliers_mmwave = np.where(np.all(np.abs(sample['mmwave_data'][i][...,:2] - np.array([[0, 1]])) < self.radius, axis=1))
            #         sample['mmwave_data'][i] = sample['mmwave_data'][i][inliers_mmwave]
            
            # else:
            #     raise ValueError('You should never reach here!')
            
            # if len(inliers[0]) == 0:
            #     sample[data_key][i] = sample[data_key][i][:1]
            # else:
            #     sample[data_key][i] = sample[data_key][i][inliers]
            elif self.outlier_type == 'box':
                mask = np.all(np.abs(sample[data_key][i][...,:2] - np.array([[0, 1]])) < self.radius, axis=1)
                inliers = np.where(mask)[0]
                if data_key != 'mmwave_data' and 'mmwave_data' in sample:
                    mm_mask = np.all(np.abs(sample['mmwave_data'][i][...,:2] - np.array([[0, 1]])) < self.radius, axis=1)
                    sample['mmwave_data'][i] = sample['mmwave_data'][i][mm_mask]
            
            else:
                raise ValueError('You should never reach here!')
            
            if isinstance(inliers, tuple):
                inliers = inliers[0]

            if len(inliers) == 0:
                if sample[data_key][i].shape[0] == 0:
                    continue
                sample[data_key][i] = sample[data_key][i][:1]
            else:
                sample[data_key][i] = sample[data_key][i][inliers]

        return sample

class RemoveFloor():
    def __init__(self, buffer=0.1):
        self.buffer = buffer

    def __call__(self, sample):
        if 'point_clouds' not in sample:
            return sample

        new_pcs = []
        for i in range(len(sample['point_clouds'])):
            pc = sample['point_clouds'][i]
            if pc is None or pc.shape[0] == 0:
                new_pcs.append(pc)
                continue
            min_y = np.min(pc[:, 1])
            threshold = min_y + self.buffer
            filtered = pc[pc[:, 1] >= threshold]
            if filtered.shape[0] == 0:
                filtered = pc[:1]
            new_pcs.append(filtered)

        sample['point_clouds'] = new_pcs

        # Apply the same floor removal to mmWave if present
        if 'mmwave_data' in sample:
            new_mms = []
            for i in range(len(sample['mmwave_data'])):
                mm = sample['mmwave_data'][i]
                if mm is None or mm.shape[0] == 0:
                    new_mms.append(mm)
                    continue
                min_y = np.min(mm[:, 1])
                threshold = min_y + self.buffer
                filtered = mm[mm[:, 1] >= threshold]
                if filtered.shape[0] == 0:
                    filtered = mm[:1]
                new_mms.append(filtered)
            sample['mmwave_data'] = new_mms

        return sample

class Pad():
    def __init__(self, max_len, pad_type='repeat', mm_len = 256):
        self.max_len = max_len
        self.mm_len = mm_len
        self.pad_type = pad_type
        if pad_type not in ['zero', 'repeat']:
            raise ValueError('pad_type must be "zero" or "repeat"')

    # def __call__(self, sample):
    #     for i in range(len(sample['point_clouds'])):
    #         cur_len = sample['point_clouds'][i].shape[0]
    #         if cur_len == 0:
    #             # add random points if the point cloud is empty
    #             sample['point_clouds'][i] = np.random.normal(0, 1, (self.max_len, sample['point_clouds'][i].shape[1]))
    #         elif cur_len >= self.max_len:
    #             indices = np.random.choice(cur_len, self.max_len, replace=False)
    #             sample['point_clouds'][i] = sample['point_clouds'][i][indices]
    #         else:
    #             if self.pad_type == 'zero':
    #                 sample['point_clouds'][i] = np.pad(sample['point_clouds'][i], ((0, self.max_len - sample['point_clouds'][i].shape[0]), (0, 0)), mode='constant')
    #             elif self.pad_type == 'repeat':
    #                 repeat = self.max_len // cur_len
    #                 residue = self.max_len % cur_len
    #                 repeated_parts = [sample['point_clouds'][i] for _ in range(repeat)]
    #                 if residue > 0:
    #                     indices = np.random.choice(cur_len, residue, replace=False)
    #                     repeated_parts.append(sample['point_clouds'][i][indices])
                    
    #                 sample['point_clouds'][i] = np.concatenate(repeated_parts, axis=0)
    #             else:
    #                 raise ValueError('You should never reach here! pad_type must be "zero" or "repeat"')
        

    #     # pad / crop mmwave_data if present, using analogous logic
    #     if 'mmwave_data' in sample:
    #         for i in range(len(sample['mmwave_data'])):
    #             cur_len = sample['mmwave_data'][i].shape[0]
    #             if cur_len == 0:
    #                 sample['mmwave_data'][i] = np.random.normal(0, 1, (self.mm_len, sample['mmwave_data'][i].shape[1]))
    #             elif cur_len >= self.mm_len:
    #                 indices = np.random.choice(cur_len, self.mm_len, replace=False)
    #                 sample['mmwave_data'][i] = sample['mmwave_data'][i][indices]
    #             else:
    #                 if self.pad_type == 'zero':
    #                     sample['mmwave_data'][i] = np.pad(sample['mmwave_data'][i], ((0, self.mm_len - sample['mmwave_data'][i].shape[0]), (0, 0)), mode='constant')
    #                 elif self.pad_type == 'repeat':
    #                     repeat = self.mm_len // cur_len
    #                     residue = self.mm_len % cur_len
    #                     repeated_parts = [sample['mmwave_data'][i] for _ in range(repeat)]
    #                     if residue > 0:
    #                         indices = np.random.choice(cur_len, residue, replace=False)
    #                         repeated_parts.append(sample['mmwave_data'][i][indices])
                        
    #                     sample['mmwave_data'][i] = np.concatenate(repeated_parts, axis=0)
    #                 else:
    #                     raise ValueError('You should never reach here! pad_type must be "zero" or "repeat"')


    #     sample['point_clouds'] = np.stack(sample['point_clouds'])
    #     if 'mmwave_data' in sample:
    #         sample['mmwave_data'] = np.stack(sample['mmwave_data'])
    #     return sample
    def __call__(self, sample):
        # --- Process Point Clouds ---
        new_pcs = []
        # Handle the object array from FeatureTransfer by converting to a list
        point_clouds_list = list(sample['point_clouds'])

        for pc in point_clouds_list:
            cur_len, num_features = pc.shape

            if cur_len == 0:
                # Add random points if the point cloud is empty
                new_pcs.append(np.random.normal(0, 1, (self.max_len, num_features)))
            elif cur_len >= self.max_len:
                indices = np.random.choice(cur_len, self.max_len, replace=False)
                new_pcs.append(pc[indices])
            else:
                if self.pad_type == 'zero':
                    padded_pc = np.pad(pc, ((0, self.max_len - cur_len), (0, 0)), mode='constant')
                    new_pcs.append(padded_pc)
                elif self.pad_type == 'repeat':
                    repeat = self.max_len // cur_len
                    residue = self.max_len % cur_len
                    
                    repeated_parts = [pc] * repeat
                    if residue > 0:
                        indices = np.random.choice(cur_len, residue, replace=False)
                        repeated_parts.append(pc[indices])
                    
                    new_pcs.append(np.concatenate(repeated_parts, axis=0))
        
        sample['point_clouds'] = np.stack(new_pcs)

        # --- Process mmWave Data (if it exists) ---
        if 'mmwave_data' in sample:
            new_mms = []
            mmwave_data_list = list(sample['mmwave_data'])

            for mm in mmwave_data_list:
                cur_len, num_features = mm.shape

                if cur_len == 0:
                    new_mms.append(np.random.normal(0, 1, (self.mm_len, num_features)))
                elif cur_len >= self.mm_len:
                    indices = np.random.choice(cur_len, self.mm_len, replace=False)
                    new_mms.append(mm[indices])
                else:
                    if self.pad_type == 'zero':
                        padded_mm = np.pad(mm, ((0, self.mm_len - cur_len), (0, 0)), mode='constant')
                        new_mms.append(padded_mm)
                    elif self.pad_type == 'repeat':
                        repeat = self.mm_len // cur_len
                        residue = self.mm_len % cur_len
                        
                        repeated_parts = [mm] * repeat
                        if residue > 0:
                            indices = np.random.choice(cur_len, residue, replace=False)
                            repeated_parts.append(mm[indices])
                        
                        new_mms.append(np.concatenate(repeated_parts, axis=0))
            
            sample['mmwave_data'] = np.stack(new_mms)

        return sample
    
class MultiFrameAggregate():
    def __init__(self, num_frames):
        self.num_frames = num_frames
        assert num_frames % 2 == 1, 'num_frames must be odd'
        self.offset = (num_frames - 1) // 2

    def __call__(self, sample):
        total_frames = len(sample['point_clouds'])
        if self.num_frames <= total_frames:
            sample['point_clouds'] = [np.concatenate(sample['point_clouds'][i-self.offset:i+self.offset]) for i in range(self.offset, total_frames-self.offset)]
            if 'keypoints' in sample:
                sample['keypoints'] = sample['keypoints'][self.offset:-self.offset]
        return sample

class MultiFrameAggregate():
    def __init__(self, num_frames):
        self.num_frames = num_frames
        assert num_frames % 2 == 1, 'num_frames must be odd'
        self.offset = (num_frames - 1) // 2

    def __call__(self, sample):
        total_frames = len(sample['point_clouds'])
        if self.num_frames <= total_frames:
            sample['point_clouds'] = [np.concatenate(sample['point_clouds'][i-self.offset:i+self.offset]) for i in range(self.offset, total_frames-self.offset)]
            # sample['point_clouds'] = [np.concatenate(sample['point_clouds'][np.maximum(0, i-self.offset):np.minimum(i+self.offset+1, total_frames-1)]) for i in range(total_frames)]
            if 'keypoints' in sample:
                sample['keypoints'] = sample['keypoints'][self.offset:-self.offset]
        # print('multi frame aggregate', len(sample['point_clouds']), len(sample['keypoints']))
        return sample

class RandomScale():
    def __init__(self, scale_min=0.9, scale_max=1.1):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample):
        scale = np.random.uniform(self.scale_min, self.scale_max)
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] *= scale
        if 'keypoints' in sample:
            sample['keypoints'] *= scale
        sample['scale'] = scale
        return sample
    
class RandomRotate():
    def __init__(self, angle_min=-np.pi, angle_max=np.pi, deg=False):
        self.angle_min = angle_min
        self.angle_max = angle_max

        if deg:
            angle_min = np.pi * angle_min / 180
            angle_max = np.pi * angle_max / 180

    def __call__(self, sample):
        angle_1 = np.random.uniform(self.angle_min, self.angle_max)
        angle_2 = np.random.uniform(self.angle_min, self.angle_max)
        rot_matrix = np.array([[np.cos(angle_1), -np.sin(angle_1), 0], [np.sin(angle_1), np.cos(angle_1), 0], [0, 0, 1]]) @ np.array([[np.cos(angle_2), 0, np.sin(angle_2)], [0, 1, 0], [-np.sin(angle_2), 0, np.cos(angle_2)]])
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] = sample['point_clouds'][i][...,:3] @ rot_matrix
        
        if 'mmwave_data' in sample:
            for i in range(len(sample['mmwave_data'])):
                sample['mmwave_data'][i][...,:3] = sample['mmwave_data'][i][...,:3] @ rot_matrix
        
        if 'keypoints' in sample:
            sample['keypoints'] = sample['keypoints'] @ rot_matrix
        sample['rotation_matrix'] = rot_matrix
        return sample
    
class RandomTranslate():
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, sample):
        translate = np.random.uniform(-self.translate_range, self.translate_range, 3)
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] += translate
        if 'keypoints' in sample:
            sample['keypoints'] += translate
        sample['translate'] = translate
        return sample

class RandomJitter():
    def __init__(self, jitter_std=0.01):
        self.jitter_std = jitter_std

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] += np.random.normal(0, self.jitter_std, sample['point_clouds'][i][...,:3].shape)
        return sample

class RandomJitterKeypoints():
    def __init__(self, jitter_std=0.01):
        self.jitter_std = jitter_std

    def __call__(self, sample):
        for i in range(len(sample['keypoints'])):
            sample['keypoints'][i][...,:3] += np.random.normal(0, self.jitter_std, sample['keypoints'][i][...,:3].shape)
        return sample
    
class RandomDrop():
    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            drop_indices = np.random.choice(sample['point_clouds'][i].shape[0], int(sample['point_clouds'][i].shape[0] * self.drop_prob), replace=False)
            sample['point_clouds'][i] = np.delete(sample['point_clouds'][i], drop_indices, axis=0)
        return sample
    
class GetCentroid():
    def __init__(self, centroid_type='minball', data_type='pointcloud'):
        self.centroid_type = centroid_type
        self.data_type = data_type
        if centroid_type not in ['none', 'zonly', 'mean', 'median', 'minball', 'dataset_median', 'kps', 'xz', 'lidar_human']:
            raise ValueError('centroid_type must be "mean" or "minball"')
        
    def __call__(self, sample):
        if self.data_type == 'pointcloud':
            pc_cat = np.concatenate(sample['point_clouds'], axis=0)
            pc_dedupe = np.unique(pc_cat[...,:3], axis=0)
            pc_dedupe = pc_dedupe[np.all(np.isfinite(pc_dedupe), axis=1)]
            if pc_dedupe.shape[0] == 0:
                sample['centroid'] = np.zeros(3)
                return sample
            if self.centroid_type == 'none':
                centroid = np.zeros(3)
            elif self.centroid_type == 'zonly':
                centroid = np.zeros(3)
                centroid[2] = np.median(pc_dedupe[...,2])
            elif self.centroid_type == 'mean':
                centroid = np.mean(pc_dedupe[...,:3], axis=0)
            elif self.centroid_type == 'median':
                centroid = np.median(pc_dedupe[...,:3], axis=0)
            elif self.centroid_type == 'minball':
                try:
                    centroid, _ = get_bounding_ball(pc_dedupe)
                except:
                    print('Error in minball')
                    centroid = np.mean(pc_dedupe[...,:3], axis=0)
            elif self.centroid_type == 'kps':
                kps_cat = np.concatenate(sample['keypoints'], axis=0)
                centroid = np.array([np.median(kps_cat[:, 0]), np.min(kps_cat[:, 1]), np.median(kps_cat[:, 2])])
            elif self.centroid_type == 'lidar_human':
                lidar_cat = np.concatenate(sample['point_clouds'], axis=0)
                lidar_dedupe = np.unique(lidar_cat[...,:3], axis=0)
                lidar_dedupe = lidar_dedupe[np.all(np.isfinite(lidar_dedupe), axis=1)]
                if lidar_dedupe.shape[0] == 0:
                    centroid = np.zeros(3)
                else:
                    centroid = np.array([np.median(lidar_dedupe[:, 0]), np.min(lidar_dedupe[:, 1]), np.median(lidar_dedupe[:, 2])])
            elif self.centroid_type == 'xz':
                centroid = np.array([np.median(pc_dedupe[:, 0]), 0, np.median(pc_dedupe[:, 2])])
            else:
                raise ValueError('You should never reach here! centroid_type must be "mean" or "minball"')
        elif self.data_type == 'mmwave':
            mm_cat = np.concatenate(sample['mmwave_data'], axis=0)
            mm_dedupe = np.unique(mm_cat[...,:3], axis=0)
            mm_dedupe = mm_dedupe[np.all(np.isfinite(mm_dedupe), axis=1)]
            if mm_dedupe.shape[0] == 0:
                sample['centroid'] = np.zeros(3)
                return sample
            if self.centroid_type == 'none':
                centroid = np.zeros(3)
            elif self.centroid_type == 'zonly':
                centroid = np.zeros(3)
                centroid[2] = np.median(mm_dedupe[...,2])
            elif self.centroid_type == 'mean':
                centroid = np.mean(mm_dedupe[...,:3], axis=0)
            elif self.centroid_type == 'median':
                centroid = np.median(mm_dedupe[...,:3], axis=0)
            elif self.centroid_type == 'minball':
                try:
                    centroid, _ = get_bounding_ball(mm_dedupe)
                except:
                    print('Error in minball')
                    centroid = np.mean(mm_dedupe[...,:3], axis=0)
            elif self.centroid_type == 'kps':
                kps_cat = np.concatenate(sample['keypoints'], axis=0)
                centroid = np.array([np.median(kps_cat[:, 0]), np.min(kps_cat[:, 1]), np.median(kps_cat[:, 2])])
            elif self.centroid_type == 'xz':
                centroid = np.array([np.median(mm_dedupe[:, 0]), 0, np.median(mm_dedupe[:, 2])])
            else:
                raise ValueError('You should never reach here! centroid_type must be "mean" or "minball"')
        else:
            raise ValueError('data_type must be "pointcloud" or "mmwave"')
        
        sample['centroid'] = centroid

        return sample
    
class Normalize():
    def __init__(self, feat_scale=None):
        self.feat_scale = feat_scale

    def __call__(self, sample):
        for i in range(len(sample['point_clouds'])):
            sample['point_clouds'][i][...,:3] -= sample['centroid'][np.newaxis]
            if 'mmwave_data' in sample:
                sample['mmwave_data'][i][...,:3] -= sample['centroid'][np.newaxis]
            if self.feat_scale:
                sample['point_clouds'][i][...,3:] /= np.array(self.feat_scale)[np.newaxis][np.newaxis]
                sample['mmwave_data'][i][...,3:] /= np.array(self.feat_scale)[np.newaxis][np.newaxis]
                sample['feat_scale'] = self.feat_scale
        
        if 'raw_point_clouds' in sample:
            for i in range(len(sample['raw_point_clouds'])):
                sample['raw_point_clouds'][i][...,:3] -= sample['centroid'][np.newaxis]

        if 'raw_mmwave_data' in sample:
            for i in range(len(sample['raw_mmwave_data'])):
                sample['raw_mmwave_data'][i][...,:3] -= sample['centroid'][np.newaxis]
        if 'keypoints' in sample:
            sample['keypoints'] -= sample['centroid'][np.newaxis][np.newaxis]
        return sample
    
class FeatureTransfer():
    def __init__(self, feature_idx=[3,4,5], knn_k=3, transfer_type='empty'):
        self.feature_idx = feature_idx
        self.knn_k = knn_k
        self.transfer_type = transfer_type

    def __call__(self, sample):
        # ensure we can replace per-frame arrays with larger-feature arrays
        if isinstance(sample['point_clouds'], np.ndarray):
            sample['point_clouds'] = [pc for pc in sample['point_clouds']]
        
        #ensure valid transfer type
        if self.transfer_type not in ['mmwave', 'empty']:
            raise ValueError('invalid transfer_type"')
        
        # if transfer empty features
        if self.transfer_type == 'empty':
            new_pcs = []
            for i in range(len(sample['point_clouds'])):
                num_points = sample['point_clouds'][i].shape[0]
                empty_feat = np.zeros((num_points, len(self.feature_idx)))
                new_pc = np.concatenate([sample['point_clouds'][i], empty_feat], axis=1)
                new_pcs.append(new_pc)
            try:
                sample['point_clouds'] = np.stack(new_pcs)
            except ValueError:
                # fallback: heterogeneous frames -> ndarray of objects
                sample['point_clouds'] = np.array(new_pcs, dtype=object)
            return sample

        # if transfer mmwave features
        if self.transfer_type == 'mmwave':
            new_pcs = []
            for i in range(len(sample['point_clouds'])):
                pc_xyz = sample['point_clouds'][i][...,:3]
                mmwave_xyz = sample['mmwave_data'][i][...,:3]
                mmwave_feat = sample['mmwave_data'][i][...,self.feature_idx]

                if pc_xyz.shape[0] == 0:
                    num_points = sample['point_clouds'][i].shape[0]
                    empty_feat = np.zeros((num_points, len(self.feature_idx)))
                    new_pc = np.concatenate([sample['point_clouds'][i], empty_feat], axis=1)
                    new_pcs.append(new_pc)
                    continue

                if mmwave_xyz.shape[0] == 0:
                    num_points = sample['point_clouds'][i].shape[0]
                    empty_feat = np.zeros((num_points, len(self.feature_idx)))
                    new_pc = np.concatenate([sample['point_clouds'][i], empty_feat], axis=1)
                    new_pcs.append(new_pc)
                    continue

                all_xyz = np.concatenate([pc_xyz, mmwave_xyz], axis=0)

                # find the vertices of the bounding cube of all points
                min_xyz = np.min(all_xyz, axis=0)
                max_xyz = np.max(all_xyz, axis=0)

                # expand the cube a bit
                expand_ratio = 0.1
                min_xyz = min_xyz - (max_xyz - min_xyz) * expand_ratio
                max_xyz = max_xyz + (max_xyz - min_xyz) * expand_ratio

                # add 0 doppler speed and intensity to the vertices
                cube_vertices = np.array([[min_xyz[0], min_xyz[1], min_xyz[2]],
                                        [min_xyz[0], min_xyz[1], max_xyz[2]],
                                        [min_xyz[0], max_xyz[1], min_xyz[2]],
                                        [min_xyz[0], max_xyz[1], max_xyz[2]],
                                        [max_xyz[0], min_xyz[1], min_xyz[2]],
                                        [max_xyz[0], min_xyz[1], max_xyz[2]],
                                        [max_xyz[0], max_xyz[1], min_xyz[2]],
                                        [max_xyz[0], max_xyz[1], max_xyz[2]]])
                cube_vertex_feat = np.zeros((8, len(self.feature_idx)))
                mmwave_xyz = np.concatenate([mmwave_xyz, cube_vertices], axis=0)
                mmwave_feat = np.concatenate([mmwave_feat, cube_vertex_feat], axis=0)

                neighbors = NearestNeighbors(n_neighbors=self.knn_k).fit(mmwave_xyz)
                distances, indices = neighbors.kneighbors(pc_xyz)

                weights = 1 / (distances + 1e-8)
                weights = weights / np.sum(weights, axis=1, keepdims=True)

                transferred_feat = np.sum(mmwave_feat[indices] * weights[..., np.newaxis], axis=1)
                new_pc = np.concatenate([sample['point_clouds'][i], transferred_feat], axis=1)
                new_pcs.append(new_pc)
                # sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i], transferred_feat], axis=1)
            # sample['point_clouds'] = new_pcs
            try:
                sample['point_clouds'] = np.stack(new_pcs)
            except ValueError:
                # fallback: heterogeneous frames -> ndarray of objects
                # print('Warning: heterogeneous point cloud sizes after mmwave feature transfer; using object array.')
                sample['point_clouds'] = np.array(new_pcs, dtype=object)

            return sample
    

class RawFeatureTransfer():
    def __init__(self, feature_idx=[3,4,5], knn_k=3, transfer_type='empty'):
        self.feature_idx = feature_idx
        self.knn_k = knn_k
        self.transfer_type = transfer_type

    def __call__(self, sample):
        # This transform operates on the raw data before other augmentations
        source_pc_key = 'raw_point_clouds'
        source_mm_key = 'raw_mmwave_data'

        # ensure we can replace per-frame arrays with larger-feature arrays
        if isinstance(sample[source_pc_key], np.ndarray):
            sample[source_pc_key] = [pc for pc in sample[source_pc_key]]
        
        #ensure valid transfer type
        if self.transfer_type not in ['mmwave', 'empty']:
            raise ValueError('invalid transfer_type"')
        
        # if transfer empty features
        if self.transfer_type == 'empty':
            new_pcs = []
            for i in range(len(sample[source_pc_key])):
                num_points = sample[source_pc_key][i].shape[0]
                empty_feat = np.zeros((num_points, len(self.feature_idx)))
                new_pc = np.concatenate([sample[source_pc_key][i], empty_feat], axis=1)
                new_pcs.append(new_pc)
            sample[source_pc_key] = new_pcs
            return sample

        # # if transfer mmwave features
        # if self.transfer_type == 'mmwave':
        #     new_pcs = []
        #     for i in range(len(sample[source_pc_key])):
        #         # Skip if there are no points to process
        #         if sample[source_pc_key][i].shape[0] == 0:
        #             # Add empty feature columns to match expected dimensions
        #             num_feat_to_add = len(self.feature_idx)
        #             current_feats = sample[source_pc_key][i].shape[1]
        #             new_pc = np.zeros((0, current_feats + num_feat_to_add))
        #             new_pcs.append(new_pc)
        #             continue

        #         pc_xyz = sample[source_pc_key][i][...,:3]
        #         mmwave_xyz = sample[source_mm_key][i][...,:3]
        #         mmwave_feat = sample[source_mm_key][i][...,self.feature_idx]
                
        #         # Skip if mmwave data is empty
        #         if mmwave_xyz.shape[0] == 0:
        #             num_points = sample[source_pc_key][i].shape[0]
        #             empty_feat = np.zeros((num_points, len(self.feature_idx)))
        #             new_pc = np.concatenate([sample[source_pc_key][i], empty_feat], axis=1)
        #             new_pcs.append(new_pc)
        #             continue

        #         all_xyz = np.concatenate([pc_xyz, mmwave_xyz], axis=0)

        #         # find the vertices of the bounding cube of all points
        #         min_xyz = np.min(all_xyz, axis=0)
        #         max_xyz = np.max(all_xyz, axis=0)

        #         # expand the cube a bit
        #         expand_ratio = 0.1
        #         min_xyz = min_xyz - (max_xyz - min_xyz) * expand_ratio
        #         max_xyz = max_xyz + (max_xyz - min_xyz) * expand_ratio

        #         # add 0 doppler speed and intensity to the vertices
        #         cube_vertices = np.array([[min_xyz[0], min_xyz[1], min_xyz[2]],
        #                                 [min_xyz[0], min_xyz[1], max_xyz[2]],
        #                                 [min_xyz[0], max_xyz[1], min_xyz[2]],
        #                                 [min_xyz[0], max_xyz[1], max_xyz[2]],
        #                                 [max_xyz[0], min_xyz[1], min_xyz[2]],
        #                                 [max_xyz[0], min_xyz[1], max_xyz[2]],
        #                                 [max_xyz[0], max_xyz[1], min_xyz[2]],
        #                                 [max_xyz[0], max_xyz[1], max_xyz[2]]])
        #         cube_vertex_feat = np.zeros((8, len(self.feature_idx)))
        #         mmwave_xyz = np.concatenate([mmwave_xyz, cube_vertices], axis=0)
        #         mmwave_feat = np.concatenate([mmwave_feat, cube_vertex_feat], axis=0)

        #         neighbors = NearestNeighbors(n_neighbors=self.knn_k).fit(mmwave_xyz)
        #         distances, indices = neighbors.kneighbors(pc_xyz)

        #         weights = 1 / (distances + 1e-8)
        #         weights = weights / np.sum(weights, axis=1, keepdims=True)

        #         transferred_feat = np.sum(mmwave_feat[indices] * weights[..., np.newaxis], axis=1)
        #         new_pc = np.concatenate([sample[source_pc_key][i], transferred_feat], axis=1)
        #         new_pcs.append(new_pc)

        #     sample[source_pc_key] = new_pcs
        #     return sample
        if self.transfer_type == 'mmwave':
            new_pcs = []
            for i in range(len(sample[source_pc_key])):
                pc_xyz = sample[source_pc_key][i][...,:3]
                mmwave_xyz = sample[source_mm_key][i][...,:3]
                mmwave_feat = sample[source_mm_key][i][...,self.feature_idx]
                all_xyz = np.concatenate([pc_xyz, mmwave_xyz], axis=0)

                # find the vertices of the bounding cube of all points
                min_xyz = np.min(all_xyz, axis=0)
                max_xyz = np.max(all_xyz, axis=0)

                # expand the cube a bit
                expand_ratio = 0.1
                min_xyz = min_xyz - (max_xyz - min_xyz) * expand_ratio
                max_xyz = max_xyz + (max_xyz - min_xyz) * expand_ratio

                # add 0 doppler speed and intensity to the vertices
                cube_vertices = np.array([[min_xyz[0], min_xyz[1], min_xyz[2]],
                                        [min_xyz[0], min_xyz[1], max_xyz[2]],
                                        [min_xyz[0], max_xyz[1], min_xyz[2]],
                                        [min_xyz[0], max_xyz[1], max_xyz[2]],
                                        [max_xyz[0], min_xyz[1], min_xyz[2]],
                                        [max_xyz[0], min_xyz[1], max_xyz[2]],
                                        [max_xyz[0], max_xyz[1], min_xyz[2]],
                                        [max_xyz[0], max_xyz[1], max_xyz[2]]])
                cube_vertex_feat = np.zeros((8, len(self.feature_idx)))
                mmwave_xyz = np.concatenate([mmwave_xyz, cube_vertices], axis=0)
                mmwave_feat = np.concatenate([mmwave_feat, cube_vertex_feat], axis=0)

                neighbors = NearestNeighbors(n_neighbors=self.knn_k).fit(mmwave_xyz)
                distances, indices = neighbors.kneighbors(pc_xyz)

                weights = 1 / (distances + 1e-8)
                weights = weights / np.sum(weights, axis=1, keepdims=True)

                transferred_feat = np.sum(mmwave_feat[indices] * weights[..., np.newaxis], axis=1)
                new_pc = np.concatenate([sample[source_pc_key][i], transferred_feat], axis=1)
                new_pcs.append(new_pc)
                # sample['point_clouds'][i] = np.concatenate([sample['point_clouds'][i], transferred_feat], axis=1)
            # sample['point_clouds'] = new_pcs
            try:
                sample[source_pc_key] = np.stack(new_pcs)
            except ValueError:
                # fallback: heterogeneous frames -> ndarray of objects
                # print('Warning: heterogeneous point cloud sizes after mmwave feature transfer; using object array.')
                sample[source_pc_key] = np.array(new_pcs, dtype=object)

            return sample