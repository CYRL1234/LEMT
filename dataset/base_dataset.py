import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain

class BaseDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train', ratio=1):
        super().__init__()
        
        self.data_path = data_path
        self.transform = transform

        with open(data_path, 'rb') as f:
            self.all_data = pickle.load(f)

        if isinstance(split, str):
            self.split_ = self.all_data['splits'][split]
        elif isinstance(split, list):
            self.split_ = [self.all_data['splits'][s] for s in split]
            self.split_ = list(chain(*self.split_))

        self.split = self.split_[:int(len(self.split_) * ratio)]
        self.data = [self.all_data['sequences'][i] for i in self.split]
        self.seq_lens = [len(seq['keypoints']) for seq in self.data]

    def __len__(self):
        return np.sum(self.seq_lens)
    
    def __getitem__(self, idx):
        seq_idx = 0
        global_idx = idx
        while idx >= self.seq_lens[seq_idx]:
            idx -= self.seq_lens[seq_idx]
            seq_idx += 1
        sample = deepcopy(self.data[seq_idx])

        sample['dataset_name'] = self.data_path.split('/')[-1].split('.')[0]
        sample['sequence_index'] = seq_idx
        sample['global_index'] = global_idx
        sample['index'] = idx
        sample['centroid'] = np.array([0.,0.,0.])
        sample['radius'] = 1.
        sample['scale'] = 1.
        sample['translate'] = np.array([0.,0.,0.])
        sample['rotation_matrix'] = np.eye(3)

        # Store raw data before augmentation
        if 'point_clouds' in sample:
            sample['raw_point_clouds'] = deepcopy(sample['point_clouds'])
        if 'mmwave_data' in sample:
            sample['raw_mmwave_data'] = deepcopy(sample['mmwave_data'])

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    @staticmethod
    def collate_fn(batch):
        batch_data = {}
        # Define keys to be collated
        keys = ['point_clouds', 'keypoints', 'centroid', 'radius', 'sequence_index', 'index', 'global_index']
        if 'mmwave_data' in batch[0]:
            keys.append('mmwave_data')
        if 'raw_point_clouds' in batch[0]:
            keys.append('raw_point_clouds')
        if 'raw_mmwave_data' in batch[0]:
            keys.append('raw_mmwave_data')

        for key in keys:
            if 'raw' in key:
                # For raw data (list of np.ndarrays), create a list of lists of tensors.
                # This handles variable numbers of points per frame.
                batch_data[key] = [[torch.from_numpy(frame).float() for frame in sample[key]] for sample in batch]
            else:
                # For padded data, stack into a single tensor.
                batch_data[key] = torch.from_numpy(np.stack([sample[key] for sample in batch], axis=0))

        return batch_data