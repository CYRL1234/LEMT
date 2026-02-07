import argparse
import os
import numpy as np
from glob import glob


def _read_bin(path, dims):
    data = np.fromfile(path, dtype=np.float64)
    if data.size == 0:
        return np.zeros((0, dims), dtype=np.float64)
    return data.reshape(-1, dims)


def inspect_sequence(root_dir):
    lidar_dir = os.path.join(root_dir, 'lidar')
    mmwave_dir = os.path.join(root_dir, 'mmwave')

    lidar_files = sorted(glob(os.path.join(lidar_dir, 'frame*.bin')))
    mmwave_files = sorted(glob(os.path.join(mmwave_dir, 'frame*.bin')))

    if not lidar_files:
        print(f'No LiDAR files found in {lidar_dir}')
    if not mmwave_files:
        print(f'No mmWave files found in {mmwave_dir}')

    def frame_idx(path):
        name = os.path.basename(path)
        return int(name.split('.')[0][5:]) - 1

    lidar_map = {frame_idx(p): p for p in lidar_files}
    mmwave_map = {frame_idx(p): p for p in mmwave_files}

    all_indices = sorted(set(lidar_map.keys()).union(mmwave_map.keys()))

    print(f'Sequence: {root_dir}')
    print(f'Frames (union): {len(all_indices)}')
    print('frame, lidar_shape, mmwave_shape, lidar_count, mmwave_count')

    for idx in all_indices:
        lidar_path = lidar_map.get(idx)
        mmwave_path = mmwave_map.get(idx)

        lidar = _read_bin(lidar_path, 3) if lidar_path else np.zeros((0, 3), dtype=np.float64)
        mmwave = _read_bin(mmwave_path, 5) if mmwave_path else np.zeros((0, 5), dtype=np.float64)

        print(
            f'{idx}, {lidar.shape}, {mmwave.shape}, '
            f'{lidar.shape[0]}, {mmwave.shape[0]}'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Path to sequence folder (E*/S*/A*)')
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        raise ValueError(f'Invalid directory: {args.root_dir}')

    inspect_sequence(args.root_dir)


if __name__ == '__main__':
    main()
