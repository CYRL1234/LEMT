import numpy as np
from copy import deepcopy
import torch
from misc.skeleton import coco2simplecoco, mmbody2simplecoco, mmfi2simplecoco, \
                          itop2simplecoco, mmfi2itop, mmbody2itop, ITOPSkeleton

class ToTensor():
    def _array_to_tensor(self, data, dtype=torch.float):
        return torch.from_numpy(data).to(dtype)
    
    def _item_to_tensor(self, data, dtype=torch.float):
        return torch.tensor([data], dtype=dtype)

    def __call__(self, sample):
        for key in ['point_clouds', 'keypoints', 'centroid', 'translate', 'rotation_matrix', 'mmwave_data']:
            if key in sample:
                # print("Key: ", key, " Type: ", type(sample[key]))
                if isinstance(sample[key], list):
                    # Convert list to numpy array first
                    sample[key] = np.array(sample[key])
                # print( key,": ", type(sample[key]), len(sample[key]) )
                sample[key] = self._array_to_tensor(sample[key])

        for key in ['action', 'sequence_index', 'index', 'global_index']:
            if key in sample:
                sample[key] = self._item_to_tensor(sample[key], dtype=torch.long)

        for key in ['radius', 'scale']:
            if key in sample:
                sample[key] = self._item_to_tensor(sample[key], dtype=torch.float)

        return sample
    
class ToSimpleCOCO():
    def __call__(self, sample):
        if sample['dataset_name'] in ['mmbody', 'lidarhuman26m', 'hmpear']:
            transfer_func = mmbody2simplecoco
        elif sample['dataset_name'] == 'mri':
            transfer_func = coco2simplecoco
        elif sample['dataset_name'] in ['mmfi', 'mmfi_lidar']:
            transfer_func = mmfi2simplecoco
        elif sample['dataset_name'] in ['itop_side', 'itop_top']:
            transfer_func = itop2simplecoco
        else:
            raise NotImplementedError
        
        sample['keypoints'] = transfer_func(sample['keypoints'])
        return sample
    
class ToITOP():
    def __call__(self, sample):
        if sample['dataset_name'] in ['mmfi', 'mmfi_lidar']:
            transfer_func = mmfi2itop
        elif sample['dataset_name'] in ['mmbody', 'lidarhuman26m', 'hmpear']:
            transfer_func = mmbody2itop
        elif sample['dataset_name'] in ['itop_side', 'itop_top']:
            transfer_func = lambda x: x
        else:
            raise ValueError('You should never reach here! dataset_name must be "mmbody", "mri", "mmfi", "itop_side" or "itop_top"')
        
        sample['keypoints'] = transfer_func(sample['keypoints'])
        return sample
    
class ReduceKeypointLen():
    def __init__(self, only_one=False, keep_type='middle', frame_to_reduce=1, indexs_to_keep=None):
        self.only_one = only_one
        assert keep_type in ['middle', 'start', 'end'], 'keep_type must be "middle", "start" or "end"'
        self.keep_type = keep_type
        self.frame_to_reduce = frame_to_reduce
        self.indexs_to_keep = indexs_to_keep

    def __call__(self, sample):
        if self.only_one:
            num_frames = len(sample['point_clouds'])
            if self.keep_type == 'middle':
                keep_idx = (num_frames - 1) // 2
            elif self.keep_type == 'start':
                keep_idx = 0
            else:
                keep_idx = num_frames - 1
            sample['keypoints'] = sample['keypoints'][keep_idx:keep_idx+1]
        elif self.indexs_to_keep is not None:
            sample['keypoints'] = sample['keypoints'][self.indexs_to_keep]
        else:
            sample['keypoints'] = sample['keypoints'][self.frame_to_reduce:-self.frame_to_reduce]
        return sample