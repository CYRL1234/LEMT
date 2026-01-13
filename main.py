import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import os

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from dataset.data_api import LitDataModule
from model.model_api import LitModel
from misc.utils import load_cfg, merge_args_cfg

def main(args):
    # --- Define an absolute path for logs ---
    root_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root_dir, 'logs')
    # ---

    if args.train_module == 'HPE Module':
        dm = LitDataModule(hparams=args)
        model = LitModel(hparams=args)
        monitor = 'val_mpjpe'
        filename = args.model_name+'-{epoch}-{val_mpjpe:.4f}'

        callbacks = [
            ModelCheckpoint(
                monitor=monitor,
                dirpath=os.path.join(log_dir, args.exp_name, args.version),
                filename=filename,
                save_top_k=1,
                save_last=True,
                mode='min'),
            RichProgressBar(refresh_rate=20)
        ]

        logger = TensorBoardLogger(save_dir=log_dir, 
                                name=args.exp_name,
                                version=args.version)
        logger.log_hyperparams(args)

        trainer = L.Trainer(
            fast_dev_run=args.dev,
            logger=logger, # wandb_logger if wandb_on else None,
            max_epochs=args.epochs,
            devices=1 if args.predict else args.gpus, # Use 1 GPU for prediction
            accelerator="gpu",
            sync_batchnorm=args.sync_batchnorm,
            num_nodes=args.num_nodes,
            gradient_clip_val=args.clip_grad,
            strategy=DDPStrategy(find_unused_parameters=False) if args.strategy == 'ddp' else args.strategy,
            callbacks=callbacks,
            precision=args.precision,
            benchmark=args.benchmark
    )
    elif args.train_module == 'Human Localization Module':
        dm = LitDataModule(hparams=args)
        model = LitModel(hparams=args)
        monitor = 'val_loss'
        filename = args.model_name+'-{epoch}-{val_loss:.4f}'

        callbacks = [
            ModelCheckpoint(
                monitor=monitor,
                dirpath=os.path.join(log_dir, args.exp_name, args.version),
                filename=filename,
                save_top_k=1,
                save_last=True,
                mode='min'),
            RichProgressBar(refresh_rate=20)
        ]

        logger = TensorBoardLogger(save_dir=log_dir, 
                                name=args.exp_name,
                                version=args.version)
        logger.log_hyperparams(args)

        trainer = L.Trainer(
            fast_dev_run=args.dev,
            logger=logger,
            max_epochs=args.epochs,
            devices=1 if args.predict else args.gpus, # Use 1 GPU for prediction
            accelerator="gpu",
            sync_batchnorm=args.sync_batchnorm,
            num_nodes=args.num_nodes,
            gradient_clip_val=args.clip_grad,
            strategy=DDPStrategy(find_unused_parameters=False) if args.strategy == 'ddp' else args.strategy,
            callbacks=callbacks,
            precision=args.precision,
            benchmark=args.benchmark
    )

    if bool(args.test):
        trainer.test(model, datamodule=dm)
    elif bool(args.predict):
        predictions = trainer.predict(model, datamodule=dm, return_predictions=True)
        save_path = os.path.join(log_dir, args.exp_name, args.version, f'{args.model_name}_predictions.pt')
        print(f'Saving predictions to {save_path}')
        torch.save(predictions, save_path)
    else:
        trainer.fit(model, datamodule=dm)
        if args.dev == 0:
            trainer.test(ckpt_path="best", datamodule=dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/test.yaml')
    parser.add_argument('-g', '--gpus', type=int, default=None,
                        help='number of gpus to use (default: all available)')
    parser.add_argument('-d', "--dev", type=int, default=0, help='fast_dev_run for debug')
    parser.add_argument('-n', "--num_nodes", type=int, default=1)
    parser.add_argument('-w', "--num_workers", type=int, default=4)
    parser.add_argument('-b', "--batch_size", type=int, default=2048)
    parser.add_argument('-e', "--batch_size_eva", type=int, default=1000, help='batch_size for evaluation')
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument("--model_ckpt_dir", type=str, default="./model_ckpt/")
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--exp_name', type=str, default='fasternet')
    parser.add_argument("--version", type=str, default="0")
    parser.add_argument("--train_module", type=str, default="HPE Module")

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)

    main(args)