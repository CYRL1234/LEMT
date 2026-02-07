import torch
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from .point_4d_convolution import *
from .transformer import *
from .featurePropagation import * 
from .keyPointRegressionHead import *
# from torchvision.models import resnet18


class P4Transformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features=3, fusion_mode = 'none', fusion_type = 'early', lidar_features = 3, mmwave_features = 5, lidar_weight = 1.0, mmwave_weight = 1.0):                                                 # output
        super().__init__()

        assert fusion_mode in ['dual', 'none'], "Invalid fusion mode"
        assert fusion_type in ['early', 'late'], "Invalid fusion type"

        #updat fusion parameters
        self.fusion_mode = fusion_mode
        self.fusion_type = fusion_type
        self.features = features
        self.mmwave_features = mmwave_features
        self.dim = dim

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')
        #if dual fusion
        if fusion_mode == 'dual':
            self.mm_tube_embedding = P4DConv(in_planes=mmwave_features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')
            
            if fusion_type == 'early':
                # Early fusion: concatenate features, project back to dim
                # self.fuse_linear = nn.Linear(2 * dim, dim)
                pass
            elif fusion_type == 'late':
                # Late fusion: separate transformers, then fuse outputs
                self.lidar_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
                self.mmwave_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
                # Fuse transformer outputs (after max pooling)
                # self.late_fuse_linear = nn.Linear(2 * dim, dim)
                self.fusion_gate = nn.Sequential(
                    nn.Linear(2 * dim, dim),
                    nn.Sigmoid()
                )
        
        # if fusion_mode != 'dual':
        #     self.feature_propagation = FeaturePropagation(k=3)
        #     self.keypoint_head = KeyPointRegressionHead(dim, output_dim // 3)
        #     self.temporal_aggregator = nn.Sequential(
        #         # Convolve over the time dimension (L=5). The number of channels is C=4.
        #         nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5), # L=5 -> 1
        #         nn.ReLU(),
        #         nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        #     )
        #     self.output_projection = nn.Linear(17, 17*3) # Project to final output dim
        #     self.emb_norm = nn.LayerNorm(dim)

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else nn.Identity()
        #TODO: try LiDAR and mmWave feature fusion here

#-------- Transformer ----------------#
        if fusion_mode != 'dual' or fusion_type != 'late':
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

        # if fusion_mode == 'dual':
        #     self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, mlp_dim),
        #     nn.GELU(),
        #     nn.Linear(mlp_dim, output_dim),
        # )

    def forward(self, input, mmwave_input=None):   
        if self.fusion_mode == 'dual': 

            if mmwave_input is None:
                raise AssertionError("fusion_mode=='dual' requires mmwave_input")
                                                 # [B, L, N, 3]
            device = input.get_device()

            lidar_feat = input[:,:,:,5-self.features:] if input.shape[-1] > 3 else None
            # feats_l: [B, L, C, n], C==dim
            #performs 4D convolution (spatial + temporal)
            # Output:
            # xyzs_l: [B, L, n, 3] - downsampled coordinates
            # feats_l: [B, L, C, n] - learned features (C=dim)
            xyzs_l, feats_l = self.tube_embedding(input[:,:,:,:3], (lidar_feat.clone().permute(0,1,3,2) if lidar_feat is not None else None))
            
            mm_feat = mmwave_input[:,:,:,3:]  # doppler/intensity part
            # feats_m: [B, L, C, n], C==dim
            xyzs_m, feats_m = self.mm_tube_embedding(mmwave_input[:,:,:,:3], mm_feat.clone().permute(0,1,3,2))
            # example early fusion: concatenate features along channel dim
            # reshape/permute to same layout before concat as rest of pipeline expects
            # ...existing code to reshape xyzs / feats...
            # # TODO: concatenate or fuse features as needed
            # features = torch.cat([feats_l, feats_m], dim=-1)  # adapt dims accordingly
            # Use LiDAR geometry for positional encoding
            if self.fusion_type == 'early':
                #Build temporal position encoding based on LiDAR coordinates
                xyzs = xyzs_l  # or fuse coordinates if needed
                #fuse coordinates
                xyzs = torch.cat([xyzs_l, xyzs_m], dim=2)  # [B, L, n_l + n_m, 3]
                
                # Build time-augmented coords -> [B, L*n, 4]
                xyzts = []
                xyzs_split = torch.split(xyzs, 1, dim=1) #original [B, L, n, 3], now split into L tensors of [B,1,n,3]
                xyzs_split = [z.squeeze(1).contiguous() for z in xyzs_split] #remove dim 1, now list of L tensors of [B,n,3]
                for t, xyz in enumerate(xyzs_split):
                    tt = torch.ones((xyz.size(0), xyz.size(1), 1), dtype=torch.float32, device=device) * (t + 1)
                    xyzts.append(torch.cat((xyz, tt), dim=2))
                xyzts = torch.stack(xyzts, dim=1)  # [B, L, n, 4]
                B, L, n, _ = xyzts.shape
                xyzts = xyzts.view(B, L * n, 4)
                xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L*n, dim]

                # #Feature fusion
                # # Prepare features to [B, L*n, C], fuse along feature dimension, then project to dim
                # B, L, C, n = feats_l.shape
                # feats_l = feats_l.permute(0, 1, 3, 2).reshape(B, L * n, C)  # [B, L, C, n] to [B, L*n, C or dim]
                # feats_m = feats_m.permute(0, 1, 3, 2).reshape(B, L * n, C)  
                # features = torch.cat([feats_l, feats_m], dim=-1)            # [B, L*n, 2*dim]
                # features = self.fuse_linear(features)                        # [B, L*n, dim]

                # Feature fusion - reshape to match LiDAR spatial structure
                B, L, C, n_l = feats_l.shape
                Bm, Lm, Cm, n_m = feats_m.shape
                
                # Reshape LiDAR features
                feats_l = feats_l.permute(0, 1, 3, 2).reshape(B, L * n_l, C)  # [B, L*n_l, dim]
                feats_m = feats_m.permute(0, 1, 3, 2).reshape(B, L * n_m, Cm)  # [B, L*n_m, dim]
                #concatenate feats_l and feats_m along the second dimension
                features = torch.cat([feats_l, feats_m], dim=1)  # [B, L*n_l + L*n_m, dim]  

                # # Interpolate or pool mmWave features to match LiDAR's n_l points
                # if n_m != n_l:
                #     # Option 1: Interpolate mmWave to match LiDAR point count
                #     feats_m = feats_m.permute(0, 1, 3, 2)  # [B, L, n_m, dim]
                #     feats_m = feats_m.reshape(B * L, n_m, Cm)  # [B*L, n_m, dim]
                #     feats_m = torch.nn.functional.interpolate(
                #         feats_m.permute(0, 2, 1),  # [B*L, dim, n_m]
                #         size=n_l,
                #         mode='linear',
                #         align_corners=False
                #     ).permute(0, 2, 1)  # [B*L, n_l, dim]
                #     feats_m = feats_m.reshape(B, L * n_l, Cm)  # [B, L*n_l, dim]
                # else:
                #     feats_m = feats_m.permute(0, 1, 3, 2).reshape(B, L * n_m, Cm)
                
                # Now concatenate along feature dimension

                # features = self.fuse_linear(features)              # [B, L*n_l, dim] no need fuse layer
                embedding = xyzts + features
                if self.emb_relu:
                    embedding = self.emb_relu(embedding)
                output = self.transformer(embedding)
                output_ = torch.max(output, dim=1, keepdim=False)[0]

            elif self.fusion_type == 'late':
                # Late fusion: separate transformers, then fuse outputs
                
                # LiDAR branch
                xyzts_l = []
                xyzs_l_split = torch.split(xyzs_l, 1, dim=1)
                xyzs_l_split = [z.squeeze(1).contiguous() for z in xyzs_l_split]
                for t, xyz in enumerate(xyzs_l_split):
                    tt = torch.ones((xyz.size(0), xyz.size(1), 1), dtype=torch.float32, device=device) * (t + 1)
                    xyzts_l.append(torch.cat((xyz, tt), dim=2))
                xyzts_l = torch.stack(xyzts_l, dim=1)
                B, L, n_l, _ = xyzts_l.shape
                xyzts_l = xyzts_l.view(B, L * n_l, 4)
                xyzts_l = self.pos_embedding(xyzts_l.permute(0, 2, 1)).permute(0, 2, 1)
                
                feats_l = feats_l.permute(0, 1, 3, 2).reshape(B, L * n_l, -1)
                embedding_l = xyzts_l + feats_l
                if self.emb_relu:
                    embedding_l = self.emb_relu(embedding_l)
                
                # mmWave branch
                xyzts_m = []
                xyzs_m_split = torch.split(xyzs_m, 1, dim=1)
                xyzs_m_split = [z.squeeze(1).contiguous() for z in xyzs_m_split]
                for t, xyz in enumerate(xyzs_m_split):
                    tt = torch.ones((xyz.size(0), xyz.size(1), 1), dtype=torch.float32, device=device) * (t + 1)
                    xyzts_m.append(torch.cat((xyz, tt), dim=2))
                xyzts_m = torch.stack(xyzts_m, dim=1)
                B, L, n_m, _ = xyzts_m.shape
                xyzts_m = xyzts_m.view(B, L * n_m, 4)
                xyzts_m = self.pos_embedding(xyzts_m.permute(0, 2, 1)).permute(0, 2, 1)
                
                feats_m = feats_m.permute(0, 1, 3, 2).reshape(B, L * n_m, -1)
                embedding_m = xyzts_m + feats_m
                if self.emb_relu:
                    embedding_m = self.emb_relu(embedding_m)
                
                # Separate transformers
                output_l = self.lidar_transformer(embedding_l)
                output_m = self.mmwave_transformer(embedding_m)
                
                # Max pooling
                output_l_ = torch.max(output_l, dim=1, keepdim=False)[0]
                output_m_ = torch.max(output_m, dim=1, keepdim=False)[0]
                
                # Late fusion: concatenate and project
                #adjust the weights of output_l_ and output_m_ 
                # output_l_ = output_l_ * self.lidar_weight
                # output_m_ = output_m_ * self.mmwave_weight
                
                # Late fusion with gating mechanism
                combined_features = torch.cat([output_l_, output_m_], dim=-1)
                gate = self.fusion_gate(combined_features)
                
                # Apply gate to fuse features
                #print("gate shape: ", gate.shape)
                output_ = gate * output_l_ + (1 - gate) * output_m_
                #print("output_ shape: ", output_.shape)
                # output_ = torch.cat([output_l_, output_m_], dim=-1)
                # output_ = self.late_fuse_linear(output_)
        else:                                                                                                          # [B, L, N, 3]
            device = input.get_device()
            # if input.shape[-1] > 3:
            #     input_feat = input[:,:,:,3:4]
            # else:
            # original [:,:,:,2:]
            input_feat = input[:,:,:,5-self.features:]
                
            xyzs, features = self.tube_embedding(input[:,:,:,:3], input_feat.clone().permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n]

            # print('xyzs: ', xyzs.max().item(), xyzs.min().item())
            # print('features: ', features.max().item(), features.min().item())

            xyzts = []
            xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
            xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
            for t, xyz in enumerate(xyzs):
                t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
                xyzt = torch.cat(tensors=(xyz, t), dim=2)
                xyzts.append(xyzt)
            xyzts = torch.stack(tensors=xyzts, dim=1)
            xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

            features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
            features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
            xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

            embedding = xyzts + features
            #TODO: concatenate LiDAR and mmWave sequence, don't mix up channel and sequence
            if self.emb_relu:
                embedding = self.emb_relu(embedding)

            output = self.transformer(embedding)
            output_ = torch.max(output, dim=1, keepdim=False)[0]

            # device = input.get_device() if input.is_cuda else torch.device('cpu')
    
            # # Step 1: Extract coordinates and dummy feature (z-coordinate)
            # input_coords = input[:,:,:,:3]  # [B, L, N, 3] (x,y,z)
            # # print('input_coords shape: ', input_coords.shape)
            # input_feat = input[:,:,:,3:] if input.shape[-1] > 3 else input[:,:,:,2:3]  # Fix: Valid dummy feature
            
            # # Step 2: Point 4D Convolution (subsample + embed local spatio-temporal features)
            # xyzs_sub, feats_sub = self.tube_embedding(
            #     input_coords, 
            #     input_feat.permute(0,1,3,2)  # [B, L, C_feat, N] → match P4DConv input
            # )  # xyzs_sub: [B, L, N', 3], feats_sub: [B, L, dim, N']

            # # Step 3: 4D Coordinate Embedding (per paper Eq. 3)
            # B, L, N_prime, _ = xyzs_sub.shape
            # # Generate 4D (x,y,z,t) coordinates
            # # print('xyzs_sub shape: ', xyzs_sub.shape)
            # xyzts_list = []
            # for t_idx, xyz in enumerate(torch.split(xyzs_sub, 1, dim=1)):
            #     xyz = xyz.squeeze(1)  # [B, N', 3]
            #     t = torch.ones((xyz.shape[0], xyz.shape[1], 1), device=device) * (t_idx + 1)
            #     xyzts_list.append(torch.cat((xyz, t), dim=2))  # [B, N', 4]
            # xyzts = torch.stack(xyzts_list, dim=1)  # [B, L, N', 4]
            # # print('xyzts shape: ', xyzts.shape)
            # # Flatten + Conv1d embedding
            # xyzts_flat = xyzts.reshape(B, L*N_prime, 4)  # [B, L×N', 4]
            # xyzts_conv1d = xyzts_flat.permute(0, 2, 1)  # [B, 4, L×N'] → Conv1d input
            # xyzts_emb = self.pos_embedding(xyzts_conv1d).permute(0, 2, 1)  # [B, L×N', dim]
            # # print('xyzts_emb shape: ', xyzts_emb.shape)
            # # Step 4: Merge embedding and P4Conv features (paper Eq. 3)
            # feats_sub_flat = feats_sub.permute(0,1,3,2).reshape(B, L*N_prime, self.dim)  # [B, L×N', dim]
            # embedding = xyzts_emb + feats_sub_flat  # [B, L×N', dim]
            # embedding = self.emb_relu(embedding)
            # embedding = self.emb_norm(embedding)  # Stabilize transformer input
            # # print('embedding shape: ', embedding.shape)
            # # Step 5: Transformer (video-level attention)
            # transformer_output = self.transformer(embedding)  # [B, L×N', dim]
            # # In P4Transformer.forward (fusion_mode='none'):
            # # After transformer, reshape output to [B, L, N_sub, dim] (matches FeaturePropagation's input)
            # B, L, N_prime, _ = xyzts.shape  # N_prime = N_sub (subsampled points per frame)
            # transformer_output_reshaped = transformer_output.reshape(B, L, N_prime, self.dim)
            # # print('transformer_output_reshaped shape: ', transformer_output_reshaped.shape)
            # # Step 6: Feature Propagation (upsample to original high-res points)
            # # Call FeaturePropagation (inputs now match the corrected forward method)
            # feats_propagated = self.feature_propagation(
            #     xyz_subsampled=xyzs_sub,  # [B, L, N_sub, 3] → Correct
            #     feats_subsampled=transformer_output_reshaped,  # [B, L, N_sub, dim] → Correct
            #     xyz_original=input_coords  # [B, L, N, 3] → Correct (original high-res points)
            # )
            # # print('feats_propagated shape: ', feats_propagated.shape)
            # #reshape to [B, L*N, dim]
            # keypoints_raw = self.keypoint_head(feats_propagated)  # [B, L, J, C] -> [B, 5, 17, 4]

            # # Reshape for temporal aggregation
            # B, L, J, C = keypoints_raw.shape
            # # Permute to [B, J, C, L] to apply Conv1d across time
            # keypoints_for_agg = keypoints_raw.permute(0, 2, 3, 1) # B, 17, 4, 5
            
            # # Reshape to combine batch and keypoint dims
            # keypoints_reshaped = keypoints_for_agg.reshape(B * J, C, L) # B*17, 4, 5

            # # Apply temporal aggregation
            # aggregated_features = self.temporal_aggregator(keypoints_reshaped) # B*17, 1, 1
            
            # # Reshape back and project to final coordinate space
            # aggregated_features = aggregated_features.view(B, J) # B, 17
            # keypoints_final = self.output_projection(aggregated_features) # B, 17*3
            
            # # Reshape to the target shape [B, 1, 17, 3]
            # keypoints_final = keypoints_final.view(B, 1, J, 3)

            # return keypoints_final

            
    #-------- Transformer ----------------#
        #Final MLP head
        output = self.mlp_head(output_)
        output = output.reshape(output.shape[0], 1, output.shape[-1]//3, 3) # B 1 J 3
        # print('output after mlp_head: ', output.max().item(), output.min().item())
        return output

class HumanLocalizeP4Transformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, output_dim, features=3
                 ):                                                 # output
        super().__init__()

        

        self.tube_embedding = P4DConv(in_planes=features, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 1],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')
        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else nn.Identity()
        #TODO: try LiDAR and mmWave feature fusion here
        self.features = features
        self.dim = dim
#-------- Transformer ----------------#
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
        )

    def forward(self, input, mmwave_input=None):
        device = input.get_device()
        
        if input.shape[-1] != 5:
            raise RuntimeError(f"HumanLocalizeP4Transformer expects 5-channel input (x,y,z,doppler,intensity), got {input.shape[-1]}")
        if self.features != 3:
            raise RuntimeError(f"HumanLocalizeP4Transformer expects features=3 for [z,doppler,intensity], got {self.features}")
        input_feat = input[:, :, :, -self.features:]
            
        xyzs, features = self.tube_embedding(input[:,:,:,:3], input_feat.clone().permute(0,1,3,2))                                             # [B, L, n, 3], [B, L, C, n]

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)
        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts + features
        #TODO: concatenate LiDAR and mmWave sequence, don't mix up channel and sequence
        if self.emb_relu:
            embedding = self.emb_relu(embedding)

        output = self.transformer(embedding)
        output_ = torch.max(output, dim=1, keepdim=False)[0]
        #Final MLP head
        output = self.mlp_head(output_)
        output = output.reshape(output.shape[0], 1, output.shape[-1]//3, 3) # B 1 J 3
        # print('output after mlp_head: ', output.max().item(), output.min().item())
        return output.squeeze(1).squeeze(1)