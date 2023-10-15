import torch
from mmcv.runner.hooks import HOOKS, Hook
from PIL import Image
import cv2
import os
import numpy as np
from mmdet3d.core.hook.openlane_vis_func import LaneVisFunc
from mmdet3d.core.hook.anchor_openlane_vis_func import Anchor_LaneVisFunc
from mmdet3d.core.hook.vrm_openlane_vis_func import Vrm_LaneVisFunc

@HOOKS.register_module()
class BevLaneVisAllHook(Hook):
    """
    process for visualization our lane3d output
    """
    def __init__(self) -> None:
        super().__init__()
        # if write:
        #     assert log_dir is not None
        # self.log_dir = log_dir
        # self.freq = freq
        # self.dst_rank = dst_rank
        # self.use_offset = use_offset
        self.vis_func= LaneVisFunc()
        self.anchor_vis_func = Anchor_LaneVisFunc()
        self.vir_vis_func0 = Vrm_LaneVisFunc()                 
    def after_train_epoch(self, runner):
        if runner.data_batch.get("maps"):
            self.vis_func(runner)
        if runner.data_batch.get("targets"):
            self.anchor_vis_func(runner)
        if runner.data_batch.get("maps_2d", None) is not None:
            self.vis_func1 = LaneVisFunc(use_off_z=False, use_offset=False)
            self.vis_func1.vis_func_2d(runner)
        if runner.data_batch.get("lables_2d", None) is not None:
            self.anchor_vis_func1 = Anchor_LaneVisFunc(use_off_z=False, use_offset=False)
            self.anchor_vis_func1.vis_2d_result(runner)
        # elif runner.data_batch.get("ipm_gt_segment"):
        if runner.model_name == 'BEVLaneForward':
            self.vir_vis_func(runner)
        elif runner.model_name == 'VRM_BEVLane':
            self.vir_vis_func(runner)
            
            
    def after_val_epoch(self, runner):
        if runner.data_batch.get("maps"):
            self.vis_func(runner)
        if runner.data_batch.get("targets"):
            self.anchor_vis_func(runner)
        if runner.data_batch.get("maps_2d", None) is not None:
            self.vis_func1 = LaneVisFunc(use_off_z=False, use_offset=False)
            self.vis_func1.vis_func_2d(runner)
        if runner.data_batch.get("lables_2d", None) is not None:
            self.anchor_vis_func1 = Anchor_LaneVisFunc(use_off_z=False, use_offset=False)
            self.anchor_vis_func1.visu_2d_result(runner)
        elif runner.model_name == 'VRM_BEVLane':
            self.vir_vis_func(runner)


    def vis(self, runner, mode='train'):
        pass
        # print('*' * 50)
        # *_, inputs = storager.get_data('inputs')
        # *_, results = storager.get_data('results')
        # print("results:", len(results))

        # NOTE results["prediction"], results["loss"]
        # NOTE predictions[1] is lanehead pred
        # print("results:", results["prediction"][1][1].shape)
        # disp_img = self.vis_func(inputs, results["prediction"][1])
        #
        # if self.write:
        #     print("write: epoc={}, train_iter={}".format(self.trainer.epoch, self.trainer.train_iter))
        #     img = Image.fromarray(disp_img)
        #     img.save(self.log_dir + '/' + str(self.trainer.epoch % 200).zfill(6) + '.jpg')