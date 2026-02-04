import numpy as np
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from LEMT.misc.utils import load_cfg, merge_args_cfg, torch2numpy
from LEMT.model.model_api import create_model
from LEMT.model.metrics import calulate_error
from LEMT.dataset.data_api import create_dataset


def recover_data(data, centroid, radius):
    data = data.clone().detach()
    data[..., :3] = data[..., :3] * radius.unsqueeze(-2).unsqueeze(-2) + centroid.unsqueeze(-2).unsqueeze(-2)
    return torch2numpy(data)


def add_mmwave_features(x, mmwave_data, knn_k=3):
    """
    Add mmWave features (doppler/intensity) to LiDAR points via kNN transfer.
    x: [B, L, N, 3]
    mmwave_data: [B, L, M, 5]
    returns: [B, L, N, 5]
    """
    if x.shape[-1] >= 5:
        return x

    xyz = x[..., :3]
    mm_xyz = mmwave_data[..., :3]
    mm_feat = mmwave_data[..., 3:5]

    B, L, N, _ = xyz.shape
    M = mm_xyz.shape[2]

    xyz_flat = xyz.reshape(B * L, N, 3)
    mm_xyz_flat = mm_xyz.reshape(B * L, M, 3)
    mm_feat_flat = mm_feat.reshape(B * L, M, 2)

    dists = torch.cdist(xyz_flat, mm_xyz_flat)
    knn = torch.topk(dists, k=knn_k, largest=False)
    idx = knn.indices
    dist_k = knn.values

    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 2)
    mm_feat_exp = mm_feat_flat.unsqueeze(1).expand(-1, N, -1, 2)
    mm_feat_knn = torch.gather(mm_feat_exp, 2, idx_exp)

    weights = 1.0 / (dist_k + 1e-8)
    weights = weights / torch.sum(weights, dim=2, keepdim=True)
    transferred = torch.sum(mm_feat_knn * weights.unsqueeze(-1), dim=2)

    transferred = transferred.reshape(B, L, N, 2)
    return torch.cat([x, transferred], dim=-1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    dataset, collate_fn = create_dataset(
        args.test_dataset['name'],
        args.test_dataset['params'],
        args.test_pipeline
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_eva,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    model = create_model(args.model_name, args.model_params)
    checkpoint_path = getattr(args, 'model_checkpoint', None) or getattr(args, 'checkpoint_path', None)
    if checkpoint_path is None:
        raise ValueError("No checkpoint provided. Set model_checkpoint or checkpoint_path in the config.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    fusion_mode = args.model_params.get('fusion_mode', 'none')

    all_preds = []
    all_gts = []

    max_batches = args.max_batches if args.max_batches is not None else float('inf')

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            x = batch['point_clouds'].float().to(device)
            y = batch['keypoints'].float().to(device)
            c = batch['centroid'].float().to(device)
            r = batch['radius'].float().to(device)

            if x.shape[-1] == 3 and 'mmwave_data' in batch:
                mm_raw = batch['mmwave_data'].float().to(device)
                x = add_mmwave_features(x, mm_raw, knn_k=3)

            if fusion_mode == 'dual':
                mm = batch['mmwave_data'].float().to(device)
                y_hat = model(x, mm)
            else:
                y_hat = model(x)

            y_rec = recover_data(y, c, r)
            y_hat_rec = recover_data(y_hat, c, r)

            all_preds.append(y_hat_rec)
            all_gts.append(y_rec)

    if not all_preds:
        print("No batches processed.")
        return

    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)

    mpjpe, pampjpe = calulate_error(preds, gts, reduce=True)
    print(f"MPJPE: {mpjpe:.4f} m")
    print(f"PA-MPJPE: {pampjpe:.4f} m")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, required=True)
    parser.add_argument('--max_batches', type=int, default=10)
    parser.add_argument('-e', '--batch_size_eva', type=int, default=16)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
