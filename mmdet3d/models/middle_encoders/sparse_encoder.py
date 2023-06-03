from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from mmdet3d.ops.pointnet2_stack import voxel_query_utils 
from ..builder import MIDDLE_ENCODERS
from mmdet3d.models.utils import utils,spconv_utils
from torch.nn import functional as F
import torch
from mmdet3d.ops import DynamicScatter,ball_query,gather_points
from mmdet3d.ops import grouping_operation 
from mmdet3d.models.voxel_encoders.voxel_encoder import HardSimpleVFE
@MIDDLE_ENCODERS.register_module()
class SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own
        max_range = [5, 51, 51]
        radius = 0.2
        nsample = 5
        self.voxel_query = voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample)
        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')
        self.voxel_size =  [0.1, 0.1, 0.2]
        self.point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.radar_voxel_size = [0.8,0.8,0.2]
        self.vfe = HardSimpleVFE(7)
        #self.vx = voxel_size[0]
        #self.vy = voxel_size[1]
        #self.vz = voxel_size[2]
        #self.x_offset = self.vx / 2 + point_cloud_range[0]
        #self.y_offset = self.vy / 2 + point_cloud_range[1]
        #self.z_offset = self.vz / 2 + point_cloud_range[2]
    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size, voxel_features2, coors2):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])

        """
        #voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample)
        batch_size = out.batch_size
        #feature = out.feature
        coors = out.indices
        coors[:, 1] = 0
        cur_voxel_xyz = utils.get_voxel_centers(
                coors[:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
        cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            cur_voxel_xyz_batch_cnt[bs_idx] = (coors[:, 0] == bs_idx).sum()

        v2p_ind_tensor = spconv_utils.generate_voxel2pinds(out,coors2)

        radar_voxel_xyz = utils.get_voxel_centers(
                coors2[:, 1:4],
                downsample_times=1,
                voxel_size=self.radar_voxel_size,
                point_cloud_range=self.point_cloud_range
            )
        
        radar_voxel_xyz_batch_cnt = radar_voxel_xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            radar_voxel_xyz_batch_cnt[bs_idx] = (coors2[:, 0] == bs_idx).sum()

        

        #xyz (Tensor): (B, N, 3) xyz coordinates of the features.
        #center_xyz (Tensor): (B, npoint, 3) centers of the ball query.
        idx = ball_query(0,2,4,radar_voxel_xyz.unsqueeze(0).contiguous(),cur_voxel_xyz.unsqueeze(0).contiguous())
        #features (Tensor): (B, C, N) tensor of features to group.
        #indices (Tensor): (B, npoint, nsample) the indicies of
        ###voxel_features2 = torch.cat([radar_voxel_xyz,voxel_features2],dim=1)

        #grouped_xyz = grouping_operation(radar_voxel_xyz.unsqueeze_(0).permute(0,2,1).contiguous(),idx.contiguous()).squeeze(0).permute(1,0,2)
        #voxel_features2 = voxel_features2.squeeze()
        #voxel_features2 = self.vfe(voxel_features2,num_points2,coors2)
        grouped_features = grouping_operation(voxel_features2.unsqueeze_(0).permute(0,2,1).contiguous(),idx.contiguous()).squeeze(0).permute(1,0,2)
        flag = idx.sum(dim=2).permute(1,0)
        flag[flag>0]=1
        ####拼接回去
        grouped_features =  F.max_pool2d(grouped_features, kernel_size=[1, grouped_features.size(2)])
        grouped_features =  grouped_features.squeeze().contiguous()
        grouped_features =  grouped_features*flag
        #out.features=torch.cat([out.features,grouped_features[:,[0,1,3,4]].contiguous()],dim=1)
        out.features=torch.cat([out.features,grouped_features.contiguous()],dim=1)

        """
        #features_ls.append(f_center)
        
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        ###
        spatial_features = spatial_features.view(N, C * D, H, W)  #torch.Size([8, 128, 2, 128, 128])   ##([8, 160, 128, 128])

        return spatial_features

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = spconv.SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
