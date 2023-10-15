      
# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Author  :   wangjing
@Version :   0.1
@License :   (C)Copyright 2019-2035
@Desc    :   None
"""
import torch
import torch.nn as nn
from mmdet3d.models.builder import HEADS, build_loss
import numpy as np
import cv2


@HEADS.register_module()
class LaneAF2DHead(nn.Module):
    def __init__(
        self,
        num_classes=1,
        in_channel=64,
        debug=False,
        seg_loss=dict(type='Lane_FocalLoss'),
        haf_loss=dict(type='RegL1Loss'),
        vaf_loss=dict(type='RegL1Loss'),
    ):

        super().__init__()
        self.num_classes = num_classes
        self.inner_channel = in_channel

        self.seg_loss = build_loss(seg_loss)
        self.haf_loss = build_loss(haf_loss)
        self.vaf_loss = build_loss(vaf_loss)

        self.binary_seg = nn.Sequential(
            nn.Conv2d(in_channel, self.inner_channel, 1),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU(),
            nn.Conv2d(self.inner_channel, self.num_classes, 1),
            # nn.Sigmoid(),
        )

        self.haf_head = nn.Sequential(
            nn.Conv2d(in_channel, self.inner_channel, 1),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU(),
            nn.Conv2d(self.inner_channel, 1, 1),
        )
        self.vaf_head = nn.Sequential(
            nn.Conv2d(in_channel, self.inner_channel, 1),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU(),
            nn.Conv2d(self.inner_channel, 2, 1),
        )

        self.debug = debug
        if self.debug:
            self.debug_loss = nn.CrossEntropyLoss()

    def loss(self, preds_dicts, gt_labels, **kwargs):
        loss_dict = dict()
        binary_seg, haf_pred, vaf_pred, topdown = preds_dicts
        gt_mask, mask_haf, mask_vaf = gt_labels
        device = binary_seg.device
        maps = gt_mask.to(device)
        mask_haf = torch.unsqueeze(mask_haf, 1)

        haf_loss = self.haf_loss(haf_pred, mask_haf, binary_seg)
        #print("haf_shape:", haf_pred.shape, mask_haf.shape, binary_seg.shape)
        
        vaf_loss = self.vaf_loss(vaf_pred, mask_vaf, binary_seg)
        #print("vaf_shape:", vaf_pred.shape, mask_vaf.shape, binary_seg.shape)

        seg_loss = self.seg_loss(binary_seg, maps)

        loss_dict['haf_loss'] = haf_loss * 10.0
        loss_dict['vaf_loss'] = vaf_loss * 10.0
        loss_dict['seg_loss'] = seg_loss * 10.0

        loss_dict['loss'] = (2 * haf_loss + 2 * vaf_loss + 8 * seg_loss) * 10.0

        return loss_dict

    def forward(self, topdown):

        binary_seg = self.binary_seg(topdown)
        haf = self.haf_head(topdown)
        vaf = self.vaf_head(topdown)

        lane_head_output = binary_seg, haf, vaf, topdown

        return lane_head_output

    def tensor2image(self, tensor, mean, std):
        mean = mean[..., np.newaxis, np.newaxis] # (nc, 1, 1)
        mean = np.tile(mean, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)
        std = std[..., np.newaxis, np.newaxis] # (nc, 1, 1)
        std = np.tile(std, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)

        image = 255.0*(std*tensor[0].cpu().float().numpy() + mean) # (nc, H, W)
        if image.shape[0] == 1:
            image = np.tile(image, (3, 1, 1))
        image = np.transpose(image, (1, 2, 0)) # (C, H, W) to (H, W, C)
        image = image[:, :, ::-1] # RGB to BGR
        return image.astype(np.uint8) # (H, W, C)

    
    def decodeAFs(BW, VAF, HAF, fg_thresh=128, err_thresh=5, viz=False):
        output = np.zeros_like(BW, dtype=np.uint8) # initialize output array
        lane_end_pts = [] # keep track of latest lane points
        next_lane_id = 1 # next available lane ID

        if viz:
            im_color = cv2.applyColorMap(BW, cv2.COLORMAP_JET)
            cv2.imshow('BW', im_color)
            ret = cv2.waitKey(0)

        # start decoding from last row to first
        for row in range(BW.shape[0]-1, -1, -1):
            cols = np.where(BW[row, :] > fg_thresh)[0] # get fg cols
            clusters = [[]]
            if cols.size > 0:
                prev_col = cols[0]

            # parse horizontally
            for col in cols:
                if col - prev_col > err_thresh: # if too far away from last point
                    clusters.append([])
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                if HAF[row, prev_col] >= 0 and HAF[row, col] >= 0: # keep moving to the right
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                elif HAF[row, prev_col] >= 0 and HAF[row, col] < 0: # found lane center, process VAF
                    clusters[-1].append(col)
                    prev_col = col
                elif HAF[row, prev_col] < 0 and HAF[row, col] >= 0: # found lane end, spawn new lane
                    clusters.append([])
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                elif HAF[row, prev_col] < 0 and HAF[row, col] < 0: # keep moving to the right
                    clusters[-1].append(col)
                    prev_col = col
                    continue

            # parse vertically
            # assign existing lanes
            assigned = [False for _ in clusters]
            C = np.Inf*np.ones((len(lane_end_pts), len(clusters)), dtype=np.float64)
            for r, pts in enumerate(lane_end_pts): # for each end point in an active lane
                for c, cluster in enumerate(clusters):
                    if len(cluster) == 0:
                        continue
                    # mean of current cluster
                    cluster_mean = np.array([[np.mean(cluster), row]], dtype=np.float32)
                    # get vafs from lane end points
                    vafs = np.array([VAF[int(round(x[1])), int(round(x[0])), :] for x in pts], dtype=np.float32)
                    vafs = vafs / np.linalg.norm(vafs, axis=1, keepdims=True)
                    # get predicted cluster center by adding vafs
                    pred_points = pts + vafs*np.linalg.norm(pts - cluster_mean, axis=1, keepdims=True)
                    # get error between prediceted cluster center and actual cluster center
                    error = np.mean(np.linalg.norm(pred_points - cluster_mean, axis=1))
                    C[r, c] = error
            # assign clusters to lane (in acsending order of error)
            row_ind, col_ind = np.unravel_index(np.argsort(C, axis=None), C.shape)
            for r, c in zip(row_ind, col_ind):
                if C[r, c] >= err_thresh:
                    break
                if assigned[c]:
                    continue
                assigned[c] = True
                # update best lane match with current pixel
                output[row, clusters[c]] = r+1
                lane_end_pts[r] = np.stack((np.array(clusters[c], dtype=np.float32), row*np.ones_like(clusters[c])), axis=1)
            # initialize unassigned clusters to new lanes
            for c, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    continue
                if not assigned[c]:
                    output[row, cluster] = next_lane_id
                    lane_end_pts.append(np.stack((np.array(cluster, dtype=np.float32), row*np.ones_like(cluster)), axis=1))
                    next_lane_id += 1

        if viz:
            im_color = cv2.applyColorMap(40*output, cv2.COLORMAP_JET)
            cv2.imshow('Output', im_color)
            ret = cv2.waitKey(0)

        return output

    def get_lane(self, preds_dicts):

        binary_seg, haf_pred, vaf_pred, topdown = preds_dicts

        # convert to arrays
        mask_out = self.tensor2image(torch.sigmoid(binary_seg).repeat(1, 3, 1, 1).detach(), 
            np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
        vaf_out = np.transpose(vaf_pred[0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        haf_out = np.transpose(haf_pred[0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

        # decode AFs to get lane instances
        seg_out = self.decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=5)
        return seg_out
    
    def create_viz(img, seg, mask, vaf, haf):
        scale = 8
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        #seg_large = cv2.resize(seg, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        #seg_large_color = cv2.applyColorMap(40*seg_large, cv2.COLORMAP_JET)
        #img[seg_large > 0, :] = seg_large_color[seg_large > 0, :]
        img = np.ascontiguousarray(img, dtype=np.uint8)
        seg_color = cv2.applyColorMap(40*seg, cv2.COLORMAP_JET)
        rows, cols = np.nonzero(seg)
        for r, c in zip(rows, cols):
            img = cv2.arrowedLine(img, (c*scale, r*scale),(int(c*scale+vaf[r, c, 0]*scale*0.75), 
                int(r*scale+vaf[r, c, 1]*scale*0.5)), seg_color[r, c, :].tolist(), 1, tipLength=0.4)
        return img
