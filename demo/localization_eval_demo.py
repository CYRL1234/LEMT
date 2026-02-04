import numpy as np
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from LEMT.misc.utils import load_cfg, merge_args_cfg, torch2numpy
from LEMT.model.model_api import create_model
from LEMT.dataset.data_api import create_dataset


def recover_points(points, centroid, radius):
    points = points.clone().detach()
    points = points * radius.unsqueeze(-1) + centroid
    return torch2numpy(points)


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

    all_err_norm = []
    all_err_abs = []

    max_batches = args.max_batches if args.max_batches is not None else float('inf')

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            mm = batch['mmwave_data'].float().to(device)
            kps = batch['keypoints'].float().to(device)
            c = batch['centroid'].float().to(device)
            r = batch['radius'].float().to(device)

            y = kps[:, 0, 7, :]
            y_hat = model(mm)

            err_norm = torch.norm(y_hat - y, dim=1)
            all_err_norm.append(err_norm.cpu().numpy())

            y_abs = recover_points(y, c, r)
            y_hat_abs = recover_points(y_hat, c, r)
            err_abs = np.linalg.norm(y_hat_abs - y_abs, axis=1)
            all_err_abs.append(err_abs)

    if not all_err_norm:
        print("No batches processed.")
        return

    err_norm = np.concatenate(all_err_norm, axis=0)
    err_abs = np.concatenate(all_err_abs, axis=0)

    print(f"Localization L2 error (normalized): {err_norm.mean():.4f} m")
    print(f"Localization L2 error (absolute): {err_abs.mean():.4f} m")


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
