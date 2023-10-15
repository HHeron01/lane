import json
# from utils.utils import *
import copy
import os.path as ops
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.multiview_datasets.instance_bevlane.openlane_extract import OpenLaneSegMask
from mmdet3d.datasets.multiview_datasets.image import img_transform, normalize_img
from mmdet.datasets import DATASETS
import cv2
from ..coord_util import ego2image,IPM2ego_matrix
from ..standard_camera_cpu import Standard_camera
from scipy.interpolate import interp1d


@DATASETS.register_module()
class Virtual_Cam_OpenLane_Dataset_v2(Dataset):
    def __init__(self, images_dir, json_file_dir, data_config=None, grid_config=None,
                 virtual_camera_config=None,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=False):
        super(Virtual_Cam_OpenLane_Dataset_v2, self).__init__()
        width_range = (grid_config['x'][0], grid_config['x'][1])
        depth_range = (grid_config['y'][0], grid_config['y'][1])
        self.width_res = grid_config['x'][2]
        self.depth_res = grid_config['y'][2]
        self.IMG_ORIGIN_W, self.IMG_ORIGIN_H = data_config['src_size'] #1920 * 1280
        self.x_min = grid_config['x'][0]
        self.x_max = grid_config['x'][1]
        self.y_min = grid_config['y'][0]
        self.y_max = grid_config['y'][1]
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False)

        self.zoff = 1.08
        self.use_valid_flag = use_valid_flag
        self.CLASSES = CLASSES
        self.is_train = not test_mode  # 1 - test_mode
        self.data_config = data_config
        self.grid = self.make_grid()

        self.images_dir = images_dir
        self.json_file_dir = json_file_dir
        self.samples = self.init_dataset_3D(json_file_dir)
        self.mask_extract = OpenLaneSegMask(width_range=width_range,
            depth_range=depth_range,
            width_res=self.width_res,
            depth_res=self.depth_res)

        # only support equal res
        self.matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.y_max / self.depth_res), int(self.x_max / self.width_res)),
            m_per_pixel=self.width_res)

        # self.matrix_IPM2ego = IPM2ego_matrix(
        #     ipm_center=(int(self.x_max / self.width_res), int(self.y_max / self.depth_res)),
        #     m_per_pixel=self.width_res)

        self.virtual_camera_config = virtual_camera_config

        self.lane3d_thick = 1
        self.lane2d_thick = 3
        self.lane_length_threshold = 3

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self.samples), dtype=np.uint8)



    def __len__(self):
        return len(self.samples) #// 20

    def make_grid(self):
        xcoords = torch.linspace(self.x_min, self.x_max, int((self.x_max - self.x_min) / self.width_res))
        ycoords = torch.linspace(self.y_min, self.y_max, int((self.y_max - self.y_min) / self.depth_res))
        yy, xx = torch.meshgrid(ycoords, xcoords)
        return torch.stack([xx, yy, torch.full_like(xx, self.zoff)], dim=-1)

    def prune_3d_lane_by_visibility(self, lane_3d, visibility):
        lane_3d = lane_3d[visibility > 0, ...]
        return lane_3d

    def prune_3d_lane_by_range(self, lane_3d, x_min, x_max):
        # TODO: solve hard coded range later
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

        # remove lane points out of x range
        lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                         lane_3d[:, 0] < x_max), ...]
        return lane_3d

    def data_filter(self, gt_lanes, gt_visibility, gt_category):
        gt_lanes = [self.prune_3d_lane_by_visibility(np.array(gt_lane), np.array(gt_visibility[k])) for k, gt_lane in
                    enumerate(gt_lanes)]
        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes)
                       if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]
        gt_lanes = [lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1] and lane[-1, 1] > self.y_samples[0]]

        gt_lanes = [self.prune_3d_lane_by_range(np.array(lane), self.x_min, self.x_max) for lane in gt_lanes]

        gt_category = [gt_category[k] for k, lane in enumerate(gt_lanes) if lane.shape[0] > 1]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        return gt_category, gt_lanes

    def sample_augmentation(self):
        fW, fH = self.data_config['input_size']
        resize = (fW / self.IMG_ORIGIN_W, fH / self.IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def get_seg_mask(self, gt_lanes, gt_category, gt_visibility):

        seg_mask = torch.from_numpy(np.array([]))
        haf_mask = torch.from_numpy(np.array([]))
        vaf_mask = torch.from_numpy(np.array([]))
        mask_offset = torch.from_numpy(np.array([]))

        gt_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z = self.mask_extract(gt_lanes, gt_category, gt_visibility)

        return gt_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z

    def perspective(self, matrix, vector):
        """Applies perspective projection to a vector using projection matrix."""
        # tmp = torch.zeros_like(vector)
        # tmp[..., 0] = vector[..., 0]
        # tmp[..., 1] = vector[..., 2]
        # tmp[..., 2] = vector[..., 1]
        # vector = tmp
        vector = vector.unsqueeze(-1)
        homogeneous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
        homogeneous = homogeneous.squeeze(-1)
        b = (homogeneous[..., -1] > 0).unsqueeze(-1)
        b = torch.cat((b, b, b), -1)
        b[..., -1] = True
        homogeneous = homogeneous * b.float()
        return homogeneous[..., :-1] / homogeneous[..., [-1]], b.float()


    def get_data_info(self, index, debug=True):
        label_json = self.samples[index]
        # label_file_path = self.json_file_dir + '/' + label_json
        label_file_path = ops.join(self.json_file_dir, label_json)

        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []
        extrinsics = []
        undists = []

        #'/home/slj/Documents/workspace/mmdet3d/data/openlane/example/image/validation/segment-260994483494315994_2797_545_2817_545_with_camera_labels/150723473494688800.jpg'
        with open(label_file_path, 'r') as fr:
            info_dict = json.loads(fr.read())

        image_path = ops.join(self.images_dir, info_dict['file_path'])
        assert ops.exists(image_path), '{:s} not exist'.format(image_path)

        img = Image.open(image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (960, 640))
        cv2.imwrite("./test_vis/pic.jpg", image)

        extrinsic = np.array(info_dict['extrinsic'])
        intrinsic = np.array(info_dict['intrinsic'])
        gt_lanes_packeds = info_dict['lane_lines']
        resize, resize_dims = self.sample_augmentation()
        img, post_rot, post_tran = img_transform(img, resize, resize_dims)

        R_vg = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]], dtype=float)
        extrinsic[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), extrinsic[:3, :3]),
            R_vg), R_gc)
        extrinsic[0:2, 3] = 0.0

        gt_lanes, gt_visibility, gt_category = [], [], []

        for j, gt_lane_packed in enumerate(gt_lanes_packeds):
            # A GT lane can be either 2D or 3D
            # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
            lane = np.array(gt_lane_packed['xyz'])
            lane_visibility = np.array(gt_lane_packed['visibility'])

            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            cam_representation = np.linalg.inv(
                np.array([[0, 0, 1, 0],
                          [-1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]], dtype=float))
            lane = np.matmul(extrinsic, np.matmul(cam_representation, lane))
            lane = lane[0:3, :].T

            gt_lanes.append(lane)
            gt_visibility.append(lane_visibility)
            gt_category.append(gt_lane_packed['category'])

        gt_category, gt_lanes = self.data_filter(gt_lanes, gt_visibility, gt_category)

        ipm_lanes = []
        for gt_lane in gt_lanes:
            gt_lane = gt_lane.T
            gt_lane_x_y = gt_lane[:2, :][::-1]
            gt_lane_z = gt_lane[2, :]

            ipm_points = np.linalg.inv(self.matrix_IPM2ego[:, :2]) @ (gt_lane_x_y -
                                                                      self.matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_,  np.array([gt_lane_z])], axis=0)
            ipm_lanes.append(res_points.T)
        # ipm_lanes = ipm_lanes.T
        # if self.use_virtual_camera:
        #     sc = Standard_camera(self.vc_intrinsic, self.vc_extrinsics, self.vc_image_shape,
        #                          cam_intrinsic, cam_extrinsics, (image.shape[0], image.shape[1]))
        #     trans_matrix = sc.get_matrix(height=0)
        #     image = cv2.warpPerspective(image, trans_matrix, self.vc_image_shape)
        #     image_gt = cv2.warpPerspective(image_gt, trans_matrix, self.vc_image_shape)

        img = normalize_img(img)
        trans.append(torch.Tensor(extrinsic[:3, 3]))
        rots.append(torch.Tensor(extrinsic[:3, :3]))
        extrinsics.append(torch.tensor(extrinsic).float())
        intrins.append(torch.cat((torch.Tensor(intrinsic), torch.zeros((3, 1))), dim=1).float())
        post_trans.append(post_tran)
        post_rots.append(post_rot)
        imgs.append(img)
        undists.append(torch.zeros(7))

        imgs, trans, rots, intrins, post_trans, post_rots, undists, extrinsics = torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(
            post_trans), torch.stack(post_rots), torch.stack(undists), torch.stack(extrinsics)
        extrinsics = torch.linalg.inv(extrinsics)# change cam2glob to glob2cam

        # gt_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z = self.get_seg_mask(gt_lanes, gt_category, gt_visibility)
        ipm_lanes, mask_seg, mask_offset, mask_haf, mask_vaf, mask_z = self.get_seg_mask(ipm_lanes, gt_category, gt_visibility)

        mask_seg[mask_seg > 0] = 1
        mask_seg = torch.from_numpy(np.array(mask_seg, dtype=np.uint8))
        mask_haf = torch.from_numpy(np.array(mask_haf)).float()
        mask_vaf = np.transpose(mask_vaf, (2, 0, 1))
        mask_vaf = torch.from_numpy(np.array(mask_vaf)).float()
        mask_offset = torch.from_numpy(np.array(mask_offset)).float()
        mask_offset = mask_offset.permute(2, 0, 1)
        mask_z = torch.from_numpy(np.array(mask_z)).float()
        mask_z = mask_z.permute(2, 0, 1)

        if debug:
            visu_path = './test_vis'
            calib = np.matmul(intrins, extrinsics)
            for gt_lane in gt_lanes:
                gt_lane = torch.tensor(gt_lane).float()
                img_points, _ = self.perspective(calib, gt_lane)

                post_img_points = []
                for img_point in img_points:
                    img_point = torch.matmul(post_rots[0, :2, :2], img_point) + post_trans[0, :2]
                    post_img_points.append(img_point.detach().cpu().numpy())
                post_img_points = np.array(post_img_points)
                x_2d, y_2d = post_img_points[:, 0].astype(np.int32), post_img_points[:, 1].astype(np.int32)
                for k in range(1, img_points.shape[0]):
                    image = cv2.line(image, (x_2d[k - 1], y_2d[k - 1]),
                                     (x_2d[k], y_2d[k]), (0, 0, 255), 4)
            cv2.imwrite(visu_path + "/debug_img.jpg", image)

        input_dict = dict(
            imgs=imgs,
            trans=trans,
            rots=rots,
            extrinsics=extrinsics,
            intrins=intrins,
            undists=undists,
            post_trans=post_trans,
            post_rots=post_rots,
            semantic_masks=mask_seg,
            mask_offset=mask_offset,
            mask_haf=mask_haf,
            mask_vaf=mask_vaf,
            mask_z=mask_z,
            gt_lanes=gt_lanes,
            grid=self.grid,
            drop_idx=torch.tensor([]),
            file_path=info_dict['file_path'],
        )

        return input_dict

    def init_dataset_3D(self, json_file_dir):
        filter_samples = []
        samples = glob.glob(json_file_dir + '**/*.json', recursive=True)
        for i, sample in enumerate(samples):
            label_file_path = ops.join(self.json_file_dir, sample)
            with open(label_file_path, 'r') as fr:
                info_dict = json.loads(fr.read())
            image_path = ops.join(self.images_dir, info_dict['file_path'])
            if not ops.exists(image_path):
                # print('{:s} not exist'.format(image_path))
                continue
            # if i < 1014:
            #     continue
            filter_samples.append(sample)
            if len(filter_samples) > 8:
                break
            # print("image_path:", image_path)

        # return samples
        return filter_samples

    def __getitem__(self, idx):
        input_dict = self.get_data_info(idx)
        data = self.pipeline(input_dict)
        return data
        # return input_dict

if __name__ == '__main__':
    data_config = {
        'cams': [],
        'Ncams': 1,
        'input_size': (960, 640),
        'src_size': (1920, 1280),
        'thickness': 5,
        'angle_class': 36,

        # Augmentation
        'resize': (-0.06, 0.11),
        'rot': (-5.4, 5.4),
        'flip': True,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.00,

    }

    grid_config = {
        'x': [-10.0, 10.0, 0.15],
        'y': [3.0, 103.0, 0.5],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 1.0],
    }
    # images_dir = '/home/slj/Documents/workspace/mmdet3d/data/openlane/example/image'
    # json_file_dir = '/home/slj/Documents/workspace/mmdet3d/data/openlane/example/annotations/segment-260994483494315994_2797_545_2817_545_with_camera_labels'

    images_dir = '/home/slj/data/openlane/openlane_all/images'
    json_file_dir = '/home/slj/data/openlane/openlane_all/lane3d_300/training/'

    dataset = OpenLane_Dataset(images_dir, json_file_dir, data_config=data_config, grid_config=grid_config,
                 test_mode=False, pipeline=None, CLASSES=None, use_valid_flag=True)

    for idx in tqdm(range(dataset.__len__())):
        input_dict = dataset.__getitem__(idx)
        print(idx)
