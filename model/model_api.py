import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle

from model.metrics import calulate_error
from misc.lr_scheduler import LinearWarmupCosineAnnealingLR
from misc.utils import torch2numpy, import_with_str, delete_prefix_from_state_dict, exists_and_is_true
from misc.skeleton import ITOPSkeleton, JOINT_COLOR_MAP, MMFiSkeleton
from misc.vis import visualize_sample
from loss.unsup import UnsupLoss

def create_model(model_name, model_params):
    if model_params is None:
        model_params = {}
    model_class = import_with_str('model', model_name)
    model = model_class(**model_params)
    return model

def create_loss(loss_name, loss_params):
    if loss_params is None:
        loss_params = {}
    if loss_name == 'UnsupLoss':
        loss_class = UnsupLoss
        loss = loss_class(**loss_params)
    else:
        loss_class = import_with_str('torch.nn', loss_name)
        loss = loss_class(**loss_params)
    return loss

def create_optimizer(optim_name, optim_params, mparams):
    if optim_params is None:
        optim_params = {}
    optim_class = import_with_str('torch.optim', optim_name)
    optimizer = optim_class(mparams, **optim_params)
    return optimizer
    
def create_scheduler(sched_name, sched_params, optimizer):
    if sched_params is None:
        sched_params = {}
    if sched_name == 'LinearWarmupCosineAnnealingLR':
        sched_class = LinearWarmupCosineAnnealingLR
    else:
        sched_class = import_with_str('torch.optim.lr_scheduler', sched_name)
    scheduler = sched_class(optimizer, **sched_params)
    return scheduler

class LitModel(L.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.model = create_model(self.hparams.model_name, self.hparams.model_params)

        if exists_and_is_true(self.hparams, 'lemt'):
            self.model_teacher = create_model(self.hparams.model_name, self.hparams.model_params)

        # Load pretrained localization model for cascaded HPE
        if self.hparams.train_module == 'Cascaded HPE Module':
            if hasattr(self.hparams, 'localization_checkpoint') and self.hparams.localization_checkpoint is not None:
                print(f"Loading pretrained localization model from: {self.hparams.localization_checkpoint}")
                self.localization_model = create_model(self.hparams.localization_model_name, self.hparams.localization_model_params)
                checkpoint = torch.load(self.hparams.localization_checkpoint, map_location=self.device)
                # Extract only the model state dict
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
                self.localization_model.load_state_dict(state_dict, strict=False)
                self.localization_model.eval()  # Set to evaluation mode
                # Freeze localization model parameters
                for param in self.localization_model.parameters():
                    param.requires_grad = False
                print("Localization model loaded and frozen.")
            else:
                raise ValueError("Cascaded HPE Module requires localization_checkpoint in config")

            # Initialize state for managing sequences, background, and previous frames
            self.last_sequence_index = None
            self.background_size = 1024
            self.new_background_sample_size = 512
            self.background_data = None # Single background model
            self.previous_frames = [[] for _ in range(self.hparams.batch_size)]
            self.roi_cube_size = getattr(self.hparams, 'roi_cube_size', 2.0)  # Default 2.0m cube

            # --- Sequence-based state for robust batch processing ---
            # State is stored in dictionaries keyed by the sequence_index from the data.
            self.sequence_background_data = {}
            self.sequence_previous_frames = {}
            # ---

            # Dynamically get max_points for HPE from the Pad augmentation step
            self.hpe_max_points = 128  # Default value
            if hasattr(self.hparams, 'train_pipeline'):
                for step in self.hparams.train_pipeline:
                    if isinstance(step, dict) and 'Pad' in step:
                        self.hpe_max_points = step['Pad'].get('max_len', self.hpe_max_points)
                        break
            print(f"HPE model will pad sequences to: {self.hpe_max_points} points.")

        if hparams.checkpoint_path is not None:
            self.load_state_dict(torch.load(hparams.checkpoint_path, map_location=self.device)['state_dict'], strict=False)
        self.loss_fn = create_loss(self.hparams.loss_name, self.hparams.loss_params)

    def _recover_data(self, data, centroid, radius):
        # print(data.shape, centroid.shape, radius.shape)
        data[..., :3] = data[..., :3] * radius.unsqueeze(-2).unsqueeze(-2) + centroid.unsqueeze(-2).unsqueeze(-2)
        data = torch2numpy(data)
        return data
    
    def _recover_all(self, x, y, y_hat, c, r):
        x = self._recover_data(x.clone().detach(), c, r)
        y = self._recover_data(y.clone().detach(), c, r)
        y_hat = self._recover_data(y_hat.clone().detach(), c, r)
        return x, y, y_hat
    
    def _filter_by_roi(self, point_clouds_list, mmwave_data_list, center_location, cube_size):
        """
        Filter points to be within a Region of Interest (ROI) cube.
        
        Args:
            point_clouds_list: list of tensors [B, L, N, C] or list of list of tensors for raw data
            mmwave_data_list: list of tensors [B, L, N, C] or list of list of tensors for raw data
            center_location: tensor [B, 3] - predicted center of the ROI
            cube_size: float - side length of the ROI cube
        """
        filtered_pcs = []
        filtered_mms = []
        half_size = cube_size / 2

        is_raw = not isinstance(point_clouds_list, torch.Tensor)

        for b in range(len(point_clouds_list)):
            center = center_location[b]
            min_bound = center - half_size
            max_bound = center + half_size
            
            batch_pcs = []
            batch_mms = []

            # Determine the number of frames in the sequence for this batch item
            num_frames = len(point_clouds_list[b]) if is_raw else point_clouds_list.shape[1]

            for l in range(num_frames):
                pc = point_clouds_list[b][l] if is_raw else point_clouds_list[b, l]
                
                # Filter LiDAR points
                if pc.shape[0] > 0:
                    mask = torch.all((pc[:, :3] >= min_bound) & (pc[:, :3] <= max_bound), dim=1)
                    filtered_pc = pc[mask]
                    if filtered_pc.shape[0] == 0:
                        filtered_pc = pc[:1] # Keep at least one point to avoid errors
                    batch_pcs.append(filtered_pc)
                else:
                    batch_pcs.append(pc)

                # Filter mmWave points
                if b < len(mmwave_data_list):
                    mm = mmwave_data_list[b][l] if is_raw else mmwave_data_list[b, l]
                    if mm.shape[0] > 0:
                        mask_mm = torch.all((mm[:, :3] >= min_bound) & (mm[:, :3] <= max_bound), dim=1)
                        filtered_mm = mm[mask_mm]
                        if filtered_mm.shape[0] == 0:
                            filtered_mm = mm[:1]
                        batch_mms.append(filtered_mm)
                    else:
                        batch_mms.append(mm)

            filtered_pcs.append(batch_pcs)
            if batch_mms:
                filtered_mms.append(batch_mms)
                
        return filtered_pcs, filtered_mms
    
    def _background_reduction(self, point_cloud, background_model, distance_threshold=0.1):
        """
        Remove background points from a single point cloud frame.
        point_cloud: [N, C]
        background_model: [M, C]
        """
        if background_model is None or background_model.shape[0] == 0 or point_cloud.shape[0] == 0:
            return point_cloud

        pc_xyz = point_cloud[:, :3]
        bg_xyz = background_model[:, :3]

        # For each point in pc_xyz, find the minimum distance to any point in bg_xyz
        distances = torch.cdist(pc_xyz, bg_xyz).min(dim=1)[0]
        
        # Keep points that are further than the threshold
        foreground_mask = distances > distance_threshold
        
        return point_cloud[foreground_mask]

    def _update_background(self, old_background, new_background_points):
        """
        Update the background model with new points.
        """
        if new_background_points.shape[0] > self.new_background_sample_size:
            indices = torch.randperm(new_background_points.shape[0])[:self.new_background_sample_size]
            sampled_new_points = new_background_points[indices]
        else:
            sampled_new_points = new_background_points

        if old_background is None or old_background.shape[0] == 0:
            # Initialize background
            if sampled_new_points.shape[0] < self.background_size:
                # Pad if not enough points
                pad_size = self.background_size - sampled_new_points.shape[0]
                padding = torch.zeros(pad_size, sampled_new_points.shape[1], device=sampled_new_points.device)
                return torch.cat([sampled_new_points, padding], dim=0)
            return sampled_new_points

        # Concatenate and resample
        combined_background = torch.cat([old_background, sampled_new_points], dim=0)
        
        if combined_background.shape[0] > self.background_size:
            indices = torch.randperm(combined_background.shape[0])[:self.background_size]
            return combined_background[indices]
        else:
            return combined_background


    def _calculate_loss(self, batch):

        if self.hparams.train_module == 'Cascaded HPE Module':
            # We must process the batch item-by-item to maintain state correctly.
            batch_size = batch['keypoints'].shape[0]
            all_y_hat = []
            # print("Batch size in Cascaded HPE Module:", batch_size)
            # --- Loop through each item in the batch ---
            for i in range(batch_size):
                # --- State management using sequence_index as the key ---
                sequence_key = batch['sequence_index'][i].item()
                # print("Processing sequence key:", sequence_key)
                # Initialize state for a new sequence if not seen before
                if sequence_key not in self.sequence_background_data:
                    self.sequence_background_data[sequence_key] = None
                    self.sequence_previous_frames[sequence_key] = []

                # Localization preprocessing (match mmfi_human_localization.yml)
                raw_mm_frames = batch['raw_mmwave_data'][i]
                seq_len = len(batch['raw_point_clouds'][i])
                if len(raw_mm_frames) >= seq_len:
                    mm_frames = raw_mm_frames[-seq_len:]
                else:
                    pad_frame = raw_mm_frames[0] if raw_mm_frames else torch.zeros_like(batch['raw_point_clouds'][i][0])
                    mm_frames = [pad_frame] * (seq_len - len(raw_mm_frames)) + list(raw_mm_frames)

                c_hpe = batch['centroid'][i]
                mm_abs_cat = torch.cat([f[:, :3] + c_hpe for f in mm_frames], dim=0)
                finite_mask = torch.isfinite(mm_abs_cat).all(dim=1)
                if finite_mask.any():
                    c_loc = torch.median(mm_abs_cat[finite_mask], dim=0).values
                else:
                    c_loc = torch.zeros_like(c_hpe)

                padded_mm_frames = []
                for f in mm_frames:
                    mm_abs = f.clone()
                    mm_abs[:, :3] = mm_abs[:, :3] + c_hpe
                    mm_norm = mm_abs.clone()
                    mm_norm[:, :3] = mm_abs[:, :3] - c_loc
                    xy = mm_norm[:, :2]
                    center = torch.tensor([0.0, 1.0], device=xy.device)
                    mask = torch.all(torch.abs(xy - center) < 3.0, dim=1)
                    mm_filt = mm_norm[mask]
                    if mm_filt.shape[0] == 0:
                        mm_filt = mm_norm[:1]
                    padded_mm_frames.append(self._pad_single_frame(mm_filt, self.hpe_max_points))

                mmwave_loc_input = torch.stack(padded_mm_frames, dim=0).unsqueeze(0)

                with torch.no_grad():
                    # Step 1: Predict human location for the single item
                    predicted_location = self.localization_model(mmwave_loc_input).squeeze(0)

                # Convert localization output to HPE space
                pred_abs = predicted_location + c_loc
                predicted_location_hpe = pred_abs - c_hpe

                # --- ROI Filtering and Background Subtraction ---
                # Step 2: Filter ONLY THE LAST FRAME by ROI for efficiency
                last_pc_frame = batch['raw_point_clouds'][i][-1]
                last_mm_frame = batch['raw_mmwave_data'][i][-1]
                filtered_pc_list, filtered_mm_list = self._filter_by_roi(
                    [[last_pc_frame]], [[last_mm_frame]], predicted_location_hpe.unsqueeze(0), self.roi_cube_size
                )
                current_roi_pc_frame = filtered_pc_list[0][0]
                current_roi_mm_frame = filtered_mm_list[0][0] if filtered_mm_list else last_mm_frame[:1]

                # Step 3: Background Subtraction using the sequence's specific background
                background_model = self.sequence_background_data[sequence_key]
                foreground_points = self._background_reduction(current_roi_pc_frame, background_model)

                # Step 4: Apply HPE preprocessing (match mmfi_feature_fusion.yml)
                # Box outlier removal on LiDAR + mmWave
                center = torch.tensor([0.0, 1.0], device=foreground_points.device)
                if foreground_points.shape[0] > 0:
                    mask_pc = torch.all(torch.abs(foreground_points[:, :2] - center) < 1.5, dim=1)
                    foreground_points = foreground_points[mask_pc] if mask_pc.any() else foreground_points[:1]
                if current_roi_mm_frame.shape[0] > 0:
                    mask_mm = torch.all(torch.abs(current_roi_mm_frame[:, :2] - center) < 1.5, dim=1)
                    current_roi_mm_frame = current_roi_mm_frame[mask_mm] if mask_mm.any() else current_roi_mm_frame[:1]

                # Radius outlier removal on LiDAR
                if foreground_points.shape[0] > 0:
                    dists = torch.cdist(foreground_points[:, :3], foreground_points[:, :3])
                    counts = (dists <= 0.15).sum(dim=1)
                    mask_r = counts >= 3
                    foreground_points = foreground_points[mask_r] if mask_r.any() else foreground_points[:1]

                # Feature transfer from mmWave to LiDAR
                if current_roi_mm_frame.shape[0] > 0 and foreground_points.shape[0] > 0:
                    pc_xyz = foreground_points[:, :3]
                    mm_xyz = current_roi_mm_frame[:, :3]
                    mm_feat = current_roi_mm_frame[:, 3:5]
                    dists = torch.cdist(pc_xyz, mm_xyz)
                    knn = torch.topk(dists, k=min(3, mm_xyz.shape[0]), largest=False)
                    idx = knn.indices
                    dist_k = knn.values
                    mm_feat_exp = mm_feat.unsqueeze(0).expand(pc_xyz.shape[0], -1, -1)
                    idx_exp = idx.unsqueeze(-1).expand(-1, -1, 2)
                    mm_feat_knn = torch.gather(mm_feat_exp, 1, idx_exp)
                    weights = 1.0 / (dist_k + 1e-8)
                    weights = weights / torch.sum(weights, dim=1, keepdim=True)
                    transferred = torch.sum(mm_feat_knn * weights.unsqueeze(-1), dim=1)
                    foreground_points = torch.cat([foreground_points, transferred], dim=1)
                else:
                    if foreground_points.shape[0] > 0:
                        zeros = torch.zeros((foreground_points.shape[0], 2), device=foreground_points.device)
                        foreground_points = torch.cat([foreground_points, zeros], dim=1)

                # Step 5: Pad the single processed frame
                padded_foreground_frame = self._pad_single_frame(foreground_points, self.hpe_max_points)
                # Step 5: Update and aggregate frames for this specific sequence
                # --- Debug: Print structure of raw point clouds ---
                if i == 0 and self.global_step == 0:
                    try:
                        batch_len = len(batch['raw_point_clouds'])
                        seq_len__item0 = len(batch['raw_point_clouds'][0])
                        frame0_shape = batch['raw_point_clouds'][0][0].shape
                        print(f"\n[Debug] Raw point clouds structure: Batch size={batch_len}, Seq Len={seq_len__item0}, First Frame Shape={frame0_shape}\n")
                    except (IndexError, AttributeError) as e:
                        print(f"\n[Debug] Could not print raw point cloud structure. Error: {e}\n")
                # ---
                seq_len = len(batch['raw_point_clouds'][i])
                self.sequence_previous_frames[sequence_key].append(padded_foreground_frame)
                if len(self.sequence_previous_frames[sequence_key]) > seq_len:
                    self.sequence_previous_frames[sequence_key].pop(0)

                aggregated_frames = self.sequence_previous_frames[sequence_key]
                if len(aggregated_frames) < seq_len:
                    padding_needed = seq_len - len(aggregated_frames)
                    pad_frame = aggregated_frames[0] if aggregated_frames else torch.zeros_like(padded_foreground_frame)
                    padded_sequence = [pad_frame] * padding_needed + aggregated_frames
                else:
                    padded_sequence = aggregated_frames
                
                hpe_input_sequence = torch.stack(padded_sequence, dim=0)

                # Step 6: HPE Prediction for the single item
                y_hat_single = self.model(hpe_input_sequence.unsqueeze(0))
                all_y_hat.append(y_hat_single)

                # Step 7: Background Update for this specific sequence
                with torch.no_grad():
                    # For the background update, we still need the unfiltered last frame
                    # to determine what is NOT the human.
                    unfiltered_last_frame_for_bg = batch['raw_point_clouds'][i][-1]
                    predicted_keypoints_for_bg = y_hat_single.squeeze(0)
                    min_bound, max_bound = self._get_human_bounding_box(predicted_keypoints_for_bg)
                    
                    human_mask = torch.all((unfiltered_last_frame_for_bg[:, :3] >= min_bound) & (unfiltered_last_frame_for_bg[:, :3] <= max_bound), dim=1)
                    new_background_points = unfiltered_last_frame_for_bg[~human_mask]
                    self.sequence_background_data[sequence_key] = self._update_background(background_model, new_background_points)

            # --- After loop, combine results for loss calculation ---
            y_hat = torch.cat(all_y_hat, dim=0)
            y = batch['keypoints']
            loss = self.loss_fn(y_hat, y)
            loss_dict = {'loss': loss.item()}

        elif exists_and_is_true(self.hparams, 'lemt_pl'):
            x_sup, y_sup = batch['point_clouds'], batch['keypoints']
            x_unsup = batch['point_clouds_unsup']
            x_lidar, y_lidar = batch['point_clouds_ref'], batch['keypoints_ref']

            y_hat_sup = self.model(x_sup)
            y_hat_lidar = self.model(x_lidar)
            loss_sup = F.mse_loss(y_hat_sup, y_sup) + F.mse_loss(y_hat_lidar, y_lidar)
            
            with torch.no_grad():
                y_hat_unsup0 = self.model(x_unsup[:, :-1])
                y_hat_unsup1 = self.model(x_unsup[:, 1:])
            loss_dynamic, loss_static = self.loss_fn(x_unsup, y_hat_unsup0, y_hat_unsup1)

            loss = loss_sup + self.hparams.unsup_weight * (loss_dynamic + loss_static)
            loss_dict = {'loss_sup': loss_sup.item(), 'loss_unsup': (loss_dynamic + loss_static).item(), 'loss': loss.item()}
            y_hat = y_hat_sup

        elif exists_and_is_true(self.hparams, 'lemt_train'):
            x_sup, y_sup = batch['point_clouds'], batch['keypoints']
            x_pl, y_pl = batch['point_clouds_pl'], batch['keypoints_pl']
            x_lidar, y_lidar = batch['point_clouds_ref'], batch['keypoints_ref']

            y_hat_sup = self.model(x_sup)
            y_hat_pl = self.model(x_pl)
            y_hat_lidar = self.model(x_lidar)
            loss_sup = F.mse_loss(y_hat_sup, y_sup)
            loss_pl = F.mse_loss(y_hat_pl, y_pl)
            loss_lidar = F.mse_loss(y_hat_lidar, y_lidar)

            loss = loss_sup + self.hparams.pl_weight * loss_pl + self.hparams.lidar_weight * loss_lidar
            loss_dict = {'loss_sup': loss_sup.item(), 'loss_pl': loss_pl.item(), 'loss_lidar': loss_lidar.item(), 'loss': loss.item()}
            y_hat = y_hat_sup
        
        elif exists_and_is_true(self.hparams, 'lemt'):
            x_sup, y_sup = batch['point_clouds'], batch['keypoints']
            x_unsup = batch['point_clouds_unsup']
            x_lidar, y_lidar = batch['point_clouds_ref'], batch['keypoints_ref']

            y_hat_sup = self.model(x_sup)
            y_hat_lidar = self.model(x_lidar)

            yt_hat_sup = self.model_teacher(x_sup)
            yt_hat_lidar = self.model_teacher(x_lidar)

            loss_sup = F.mse_loss(y_hat_sup, y_sup) + F.mse_loss(yt_hat_sup, y_sup)
            loss_lidar = F.mse_loss(y_hat_lidar, y_lidar) + F.mse_loss(yt_hat_lidar, y_lidar)

            x_unsup0 = x_unsup[:, :-1]
            x_unsup1 = x_unsup[:, 1:]

            with torch.no_grad():
                yt_hat_unsup0 = self.model_teacher(x_unsup0)
                yt_hat_unsup1 = self.model_teacher(x_unsup1)

            y_hat_unsup0 = self.model(x_unsup0)
            y_hat_unsup1 = self.model(x_unsup1)

            loss_pseudo = F.mse_loss(y_hat_unsup0, yt_hat_unsup0.detach()) + F.mse_loss(y_hat_unsup1, yt_hat_unsup1.detach())
            loss_dynamic, loss_static = self.loss_fn(x_unsup, y_hat_unsup0, y_hat_unsup1)

            loss = loss_sup + self.hparams.w_lidar * loss_lidar + \
                              self.hparams.w_pseudo * loss_pseudo + \
                              self.hparams.w_dynamic * loss_dynamic + \
                              self.hparams.w_static * loss_static
            
            loss_dict = {'loss_sup': loss_sup.item(), 'loss_lidar': loss_lidar.item(), 
                         'loss_pseudo': loss_pseudo.item(), 'loss_dynamic': loss_dynamic.item(), 
                         'loss_static': loss_static.item(), 'loss': loss.item()}
            y_hat = y_hat_sup

        elif self.hparams.train_module == 'Human Localization Module':
            input_type = getattr(self.hparams, 'localization_input', 'mmwave')
            if input_type in ('lidar', 'feature_transferred_lidar'):
                data = batch['point_clouds']
            else:
                data = batch['mmwave_data']
            # print(batch['keypoints'].shape)
            y = batch['keypoints'][:, 0, 7, :] #batch['keypoints'] is shape (16,1,17,3)
            y_hat = self.model(data)
            loss = self.loss_fn(y_hat, y)
            loss_dict = {'loss': loss.item()}
        else: # Corresponds to 'HPE Module'
            # Get fusion mode from model params---
            fusion_mode = self.hparams.model_params.get('fusion_mode', 'none')
            #---
            x, y = batch['point_clouds'], batch['keypoints']
            
            #####Debug######
            # print("fusion_mode:", fusion_mode)
            #---
            if fusion_mode == 'dual':
                # Get mmwave data and ensure it exists
                mm = batch.get('mmwave_data')
                assert mm is not None, "fusion_mode=='dual' requires mmwave_data in batch"
                y_hat = self.model(x, mm)  # Pass both inputs to model
            else:
            #---
                input_modality = getattr(self.hparams, 'input_modality', None)
                if input_modality is None:
                    input_modality = self.hparams.model_params.get('input_modality', 'lidar')

                if input_modality == 'mmwave':
                    mm = batch.get('mmwave_data')
                    assert mm is not None, "feature_transferred_lidar input requires mmwave_data in batch"
                    y_hat = self.model(mm)  # Pass both inputs
                else:
                    y_hat = self.model(x)  # Original single-modality forward

            loss = self.loss_fn(y_hat, y)
            loss_dict = {'loss': loss.item()}

        return loss, loss_dict, y_hat
    
    def _visualize(self, x, y, y_hat):
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        y_hat_cpu = y_hat.cpu()
        sample = x_cpu[0][0][:, [0, 2, 1]], y_cpu[0][0][:, [0, 2, 1]], y_hat_cpu[0][0][:, [0, 2, 1]]
        fig = visualize_sample(sample, edges=MMFiSkeleton.bones, point_size=2, joint_size=25, linewidth=2, padding=0.1)
        tb = self.logger.experiment
        tb.add_figure('val_sample', fig, global_step=self.global_step)
        plt.close(fig)
        plt.clf()

    def training_step(self, batch, batch_idx):
        x, y, c, r = batch['point_clouds'], batch['keypoints'], batch['centroid'], batch['radius']
        loss, loss_dict, y_hat = self._calculate_loss(batch)
        #---
        x = batch['point_clouds']
        #---
        log_dict = {f'train_{k}': v for k, v in loss_dict.items()}
        if self.hparams.train_module in ['HPE Module', 'Cascaded HPE Module']:
            x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
            mpjpe, pampjpe = calulate_error(y_hat_rec, y_rec)
            log_dict = {**log_dict, 'train_mpjpe': mpjpe, 'train_pampjpe': pampjpe}

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, c, r = batch['point_clouds'], batch['keypoints'], batch['centroid'], batch['radius']
        loss, loss_dict, y_hat = self._calculate_loss(batch)
        log_dict = {f'val_{k}': v for k, v in loss_dict.items()}
        
        if self.hparams.train_module in ['HPE Module', 'Cascaded HPE Module']:
            x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
            mpjpe, pampjpe = calulate_error(y_hat_rec, y_rec)
            log_dict = {**log_dict, 'val_mpjpe': mpjpe, 'val_pampjpe': pampjpe}
            if batch_idx == 0:
                self._visualize(x, y, y_hat)

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, c, r = batch['point_clouds'], batch['keypoints'], batch['centroid'], batch['radius']
        loss, loss_dict, y_hat = self._calculate_loss(batch)
        log_dict = {f'test_{k}': v for k, v in loss_dict.items()}
        if self.hparams.train_module in ['HPE Module', 'Cascaded HPE Module']:
            x_rec, y_rec, y_hat_rec = self._recover_all(x, y, y_hat, c, r)
            mpjpe, pampjpe = calulate_error(y_hat_rec, y_rec)
            log_dict = {**log_dict, 'test_mpjpe': mpjpe, 'test_pampjpe': pampjpe}

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, c, r = batch['point_clouds'], batch['centroid'], batch['radius']
        
        if self.hparams.train_module == 'Human Localization Module':
            mmwave_data = batch['mmwave_data']
            y_hat = self.model(mmwave_data)
            # For localization, y_hat is the location, not keypoints.
            # _recover_all might not be appropriate here as it expects keypoint shapes.
            # We return the direct output.
            return y_hat

        elif self.hparams.train_module == 'Cascaded HPE Module':
            with torch.no_grad():
                predicted_location = self.localization_model(batch['mmwave_data']).squeeze()
            # Placeholder for ROI filtering
            y_hat = self.model(x)
        
        else: # 'HPE Module'
            y_hat = self.model(x)

        _, _, y_hat_rec = self._recover_all(x, y_hat, y_hat, c, r)
        return y_hat_rec

    def configure_optimizers(self):
        optim_params = self.hparams.optim_params if self.hparams.optim_params is not None else {}
        optimizer = create_optimizer(self.hparams.optim_name, optim_params, self.model.parameters())

        if self.hparams.sched_name is None:
            return optimizer
        else:
            sched_params = self.hparams.sched_params if self.hparams.sched_params is not None else {}
            scheduler = create_scheduler(self.hparams.sched_name, sched_params, optimizer)
            return [optimizer], [scheduler]

    def _get_human_bounding_box(self, keypoints, buffer=0.2):
        """
        Get a bounding box from keypoints with a buffer.
        keypoints: [1, 17, 3]
        """
        if keypoints.dim() > 2:
            keypoints = keypoints.squeeze(0) # Shape: [17, 3]
        
        min_coords = torch.min(keypoints, dim=0)[0] - buffer
        max_coords = torch.max(keypoints, dim=0)[0] + buffer
        return min_coords, max_coords
    
    def _pad_single_frame(self, frame, max_points):
        """
        Pads a single frame to a fixed number of points.
        """
        num_channels = 5 # Default to 5 for (x,y,z,doppler,intensity)
        if frame.shape[0] == 0:
            # Handle empty frames by creating a zero tensor with the correct number of channels
            return torch.zeros((max_points, num_channels), device=self.device)

        if frame.shape[0] > max_points:
            frame = frame[:max_points, :]
        
        pad_size = max_points - frame.shape[0]
        # Ensure the padding matches the frame's channel count, just in case
        if frame.shape[1] != num_channels:
             print(f"Warning: Frame has {frame.shape[1]} channels, but padding with {num_channels}.")
        padded_frame = F.pad(frame, (0, 0, 0, pad_size))
        return padded_frame

    def _pad_sequences_for_hpe(self, sequences, max_points=256):
        """
        Pad sequences to a fixed number of points for the HPE model.
        """
        processed_sequences = []
        for seq in sequences:
            padded_frames = []
            for frame in seq:
                if frame.shape[0] == 0:
                    # Handle empty frames by creating a zero tensor
                    padded_frame = torch.zeros((max_points, 3), device=self.device)
                else:
                    if frame.shape[0] > max_points:
                        frame = frame[:max_points, :]
                    
                    pad_size = max_points - frame.shape[0]
                    # Use the correct number of channels from the frame
                    num_channels = frame.shape[1]
                    padded_frame = F.pad(frame, (0, 0, 0, pad_size))
                
                padded_frames.append(padded_frame)
            processed_sequences.append(torch.stack(padded_frames, dim=0))
        
        return torch.stack(processed_sequences, dim=0).float()