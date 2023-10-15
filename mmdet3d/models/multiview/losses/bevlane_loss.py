import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.models.losses.utils import weighted_loss
from mmdet3d.models.builder import LOSSES

@LOSSES.register_module()
class OffsetLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        # self.loss_func = torch.nn.MSELoss(reduction="mean")
    def forward(self, logits, labels, mask=None):
        if mask is None:
            off_logits = logits
            off_labels = labels
        else:
            bool_mask = mask > 0
            if len(logits.shape) == 4 and len(mask.shape) == 3:
                channel = logits.shape[1]
                # expand mask shape [batch_size, h, w] to [batch_size, channel, h, w]
                bool_mask = bool_mask.unsqueeze(1)
                bool_mask = torch.repeat_interleave(bool_mask, repeats=channel, dim=1)

            if len(logits.shape) == 4 and len(labels.shape) == 3:
                channel = logits.shape[1]
                labels = labels.unsqueeze(1)
                labels = torch.repeat_interleave(labels, repeats=channel, dim=1)

            # print("logits:", logits.shape, labels.shape)
            off_logits = torch.masked_select(logits, bool_mask)
            off_labels = torch.masked_select(labels, bool_mask)

        loss = F.l1_loss(off_logits, off_labels, reduction=self.reduction)
        return loss

@LOSSES.register_module()
class Lane_FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=[0.5, 0.5], size_average=True):
        super(Lane_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([1 - alpha, alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.eps = 1e-10

    def forward(self, outputs, targets):
        # outputs, targets = torch.sigmoid(outputs.view(-1, 1)), targets.view(-1, 1).long() # (N, 1)
        outputs, targets = torch.sigmoid(outputs.reshape(-1, 1)), targets.reshape(-1, 1).long() # (N, 1)
        outputs = torch.cat((1 - outputs, outputs), dim=1) # (N, 2)

        pt = outputs.gather(1, targets).view(-1)
        logpt = torch.log(outputs + self.eps)
        logpt = logpt.gather(1, targets).view(-1)

        if self.alpha is not None:
            if self.alpha.type() != outputs.data.type():
                self.alpha = self.alpha.type_as(outputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        # inputs = inputs.squeeze(1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        else:
            raise NotImplementedError

@LOSSES.register_module()
class RegL1Loss(nn.Module):
    def __init__(self, ignore_index=255):
        super(RegL1Loss, self).__init__()
        self.ignore_index = ignore_index
        self.fg_threshold = 0.2

    def forward(self, output, target, mask):
        if mask is not None:
            _mask = mask.detach().clone()
            _mask[mask == self.ignore_index] = 0.
            _mask[_mask <= self.fg_threshold] = 0.
            _mask[_mask > self.fg_threshold] = 1.0
            loss = F.l1_loss(output * _mask, target * _mask, reduction='mean')
        else:
            loss = F.l1_loss(output, target, reduction='mean')
        return loss


@LOSSES.register_module()
class OhemLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(OhemLoss, self).__init__()
        self.ignore_index = ignore_index
        self.fg_threshold = 0.2

        self.smooth_l1_sigma = 1.0
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')  # reduce=False

    def forward(self, inputs, targets):
        # inputs = inputs.squeeze(1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        else:
            raise NotImplementedError

    def ohem_loss(self, batch_size, cls_pred, cls_target, loc_pred, loc_target):
        """    Arguments:
         batch_size (int): number of sampled rois for bbox head training
         loc_pred (FloatTensor): [R, 4], location of positive rois
         loc_target (FloatTensor): [R, 4], location of positive rois
         pos_mask (FloatTensor): [R], binary mask for sampled positive rois
         cls_pred (FloatTensor): [R, C]
         cls_target (LongTensor): [R]
         Returns:
               cls_loss, loc_loss (FloatTensor)
        """

        ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
        ohem_loc_loss = self.smooth_l1_loss(loc_pred, loc_target).sum(dim=1)

        print(ohem_cls_loss.shape, ohem_loc_loss.shape)
        loss = ohem_cls_loss + ohem_loc_loss

        sorted_ohem_loss, idx = torch.sort(loss, descending=True)
        # 再对loss进行降序排列

        keep_num = min(sorted_ohem_loss.size()[0], batch_size)
        # 得到需要保留的loss数量

        if keep_num < sorted_ohem_loss.size()[0]:
            # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留

            keep_idx_cuda = idx[:keep_num]  # 保留到需要keep的数目
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
            ohem_loc_loss = ohem_loc_loss[keep_idx_cuda]  # 分类和回归保留相同的数目

        cls_loss = ohem_cls_loss.sum() / keep_num
        loc_loss = ohem_loc_loss.sum() / keep_num  # 然后分别对分类和回归loss求均值
        return cls_loss, loc_loss



@LOSSES.register_module()
class LanePushPullLoss(nn.Module):
    """
    An embedding loss to min var of per cluster and max distance between different clusters.

    So, for easier cluster, margin_var should be small, and margin_dist should be larger

    Inputs:
    featmap: prediction of network, [b,N,h,w], float tensor
    gt: gt, [b,N,h,w], long tensor, all val >= ignore_label will NOT be contributed to loss.

    loss = var_weight * var_loss + dist_weight * dist_loss

    Args:
        var_weight (float):
        dist_weight (float):
        margin_var (float): margin for var, any var < this margin will NOT be counted in loss
        margin_dist (float): margin for distance, any distance > this margin will NOT be counted in loss
        ignore_label: val in gt >= this arg, will be ignored.
    """

    def __init__(self, var_weight, dist_weight, margin_var, margin_dist, ignore_label):
        super(LanePushPullLoss, self).__init__()
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.margin_var = margin_var
        self.margin_dist = margin_dist
        self.ignore_label = ignore_label

    def forward(self, featmap, gt):
        assert (featmap.shape[2:] == gt.shape[2:])
        pull_loss = []
        push_loss = []
        C = gt[gt < self.ignore_label].max().item()
        # [B, N, H, W] = fm, [B, 1, H, W]  = gt
        # TODO not an optimized implement here. Should not expand B dim.
        for b in range(featmap.shape[0]):
            bfeat = featmap[b]
            bgt = gt[b][0]
            instance_centers = {}
            for i in range(1, int(C) + 1):
                instance_mask = bgt == i
                if instance_mask.sum() == 0:
                    continue
                pos_featmap = bfeat[:, instance_mask].T.contiguous()  # mask_num x N
                instance_center = pos_featmap.mean(dim=0, keepdim=True)  # N x mask_num (mean)-> N x 1
                instance_centers[i] = instance_center
                # TODO xxx
                instance_loss = torch.clamp(torch.cdist(pos_featmap, instance_center) - self.margin_var, min=0.0)
                pull_loss.append(instance_loss.mean())
            for i in range(1, int(C) + 1):
                for j in range(1, int(C) + 1):
                    if i == j:
                        continue  # No need to push
                    if i not in instance_centers or j not in instance_centers:
                        continue
                    instance_loss = torch.clamp(
                        2 * self.margin_dist - torch.cdist(instance_centers[i], instance_centers[j]), min=0.0)
                    push_loss.append(instance_loss)
        if len(pull_loss) > 0:
            pull_loss = torch.cat([item.unsqueeze(0) for item in pull_loss]).mean() * self.var_weight
        else:
            pull_loss = 0.0 * featmap.mean()  # Fake loss

        if len(push_loss) > 0:
            push_loss = torch.cat([item.unsqueeze(0) for item in push_loss]).mean() * self.dist_weight
        else:
            push_loss = 0.0 * featmap.mean()  # Fake loss
        return push_loss + pull_loss
@LOSSES.register_module()
class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        targets = targets.float()
        num = torch.sum(outputs * targets * mask)
        den = torch.sum(outputs * mask + targets * mask - outputs * targets * mask)
        return 1 - num / den

class BevLaneLoss(nn.Module):
    def __int__(self):
        super(BevLaneLoss, self).__init__()
        # self.off_loss = OffsetLoss()
        # self.seg_loss = FocalLoss()
        # self.haf_loss = RegL1Loss()
        # self.vaf_loss = RegL1Loss()

    def forward(self, labels, preds):
        binary_seg, embedding, haf_pred, vaf_pred, off_pred = preds
        # seg_mask, haf_mask, vaf_mask, mask_offset = labels
        total_loss = dict()
        #............
        # only for test
        seg_mask, _, haf_mask, vaf_mask, mask_offset = preds
        #...........
        device = seg_mask.device
        seg_mask, haf_mask, vaf_mask, mask_offset = seg_mask.to(device), haf_mask.to(device), vaf_mask.to(device), mask_offset.to(device)

        # haf_loss = self.haf_loss(haf_pred, haf_mask, seg_mask)
        # vaf_loss = self.vaf_loss(vaf_pred, vaf_mask, seg_mask)
        # seg_loss = self.seg_loss(binary_seg, seg_mask)

        haf_loss = RegL1Loss().forward(haf_pred, vaf_mask, seg_mask)
        vaf_loss = RegL1Loss().forward(vaf_pred, haf_mask, seg_mask)
        seg_loss = Lane_FocalLoss().forward(binary_seg, haf_mask)

        total_loss['loss'] = seg_loss + haf_loss + vaf_loss

        # total_loss['loss'] = torch.tensor(0.5).requires_grad_(True)
        return total_loss




