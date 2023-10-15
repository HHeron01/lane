# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.losses import FocalLoss, SmoothL1Loss, binary_cross_entropy
from .axis_aligned_iou_loss import AxisAlignedIoULoss, axis_aligned_iou_loss
from .chamfer_distance import ChamferDistance, chamfer_distance
from .multibin_loss import MultiBinLoss
from .paconv_regularization_loss import PAConvRegularizationLoss
from .rotated_iou_loss import RotatedIoU3DLoss
from .uncertain_smooth_l1_loss import UncertainL1Loss, UncertainSmoothL1Loss
from mmdet3d.models.multiview.losses.bevlane_loss import OffsetLoss, RegL1Loss, Lane_FocalLoss#, FocalLoss#, BevLaneLoss
from mmdet3d.models.multiview.losses.bevlane_loss import LanePushPullLoss, IoULoss
from mmdet3d.models.lane2d_tasks.losses.laneaf_loss import LaneAFLoss

__all__ = [
    'FocalLoss', 'SmoothL1Loss', 'binary_cross_entropy', 'ChamferDistance',
    'chamfer_distance', 'axis_aligned_iou_loss', 'AxisAlignedIoULoss',
    'PAConvRegularizationLoss', 'UncertainL1Loss', 'UncertainSmoothL1Loss',
    'MultiBinLoss', 'RotatedIoU3DLoss', 'OffsetLoss', 'RegL1Loss',
    'Lane_FocalLoss', 'LanePushPullLoss', 'IoULoss', 'LaneAFLoss', #'FocalLoss',
]