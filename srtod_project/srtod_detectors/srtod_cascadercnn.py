# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Union

from mmdet.structures import SampleList

import copy
import warnings

from .srtod_twostagedetector import SRTOD_TwoStageDetector

import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
            #nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up_direct(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv(x)
   
class RH(nn.Module):
    def __init__(self, in_channels=256, out_channels=3):
        super(RH, self).__init__()
        self.up1 = Up_direct(in_channels,128)
        self.up2 = Up_direct(128,64)
        self.out_conv = OutConv(64, out_channels)
    
    def forward(self, x):
        P0 = self.up1(x)
        P0 = self.up2(P0)
        r_img = self.out_conv(P0)
        return r_img


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DGFE(nn.Module):
    def __init__(self, gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max']):
        super(DGFE, self).__init__()
    
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types


    def forward(self, x, difference_map, learnable_thresh):

        #Filtration
        difference_map_mask = (torch.sign(difference_map - learnable_thresh) + 1)*0.5

        #resize
        feat_difference_map = torch.nn.functional.interpolate(difference_map_mask,size=(x.shape[2],x.shape[3])) #nearest

        #reweighting
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        feat_diff_mat = feat_difference_map.repeat(1,x.shape[1],1,1)
        x_out = x * scale
        x_out = torch.mul(x_out,feat_diff_mat) + x_out

        return x_out


@MODELS.register_module()
class SRTOD_CascadeRCNN(SRTOD_TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 loss_res: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 learnable_thresh: float = 0.0156862,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.loss_res = MODELS.build(loss_res)
        self.rh = RH()
        self.dgfe = DGFE()


        self.learnable_thresh = torch.nn.Parameter(torch.tensor(learnable_thresh), requires_grad=True)

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             img_inputs: Tensor) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        #SR-TOD
        #image self-reconstruction
        r_img = self.rh(x[0].clone())

        #difference map
        difference_map = torch.sum(torch.abs(r_img - img_inputs),dim=1,keepdim=True) / 3 

        # print("thresh:", self.learnable_thresh * 255)
            
        x = list(x) 
        x[0] = self.dgfe(x[0], difference_map, self.learnable_thresh)
        x = tuple(x) 

        losses = dict()

        #reconstruction image loss
        loss_res = self.loss_res(r_img,img_inputs)
        losses['loss_res'] = loss_res

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                img_inputs: Tensor,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(batch_inputs)
        # If there are no pre-defined proposals, use RPN to get proposals

        #SR-TOD
        #image self-reconstruction
        r_img = self.rh(x[0].clone())

        #difference map
        difference_map = torch.sum(torch.abs(r_img - img_inputs),dim=1,keepdim=True) / 3 
       
        # print("thresh:", self.learnable_thresh * 255)
        
        x = list(x) 
        x[0] = self.dgfe(x[0], difference_map, self.learnable_thresh)
        x = tuple(x) 


        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    
