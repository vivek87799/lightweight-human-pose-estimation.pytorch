import argparse
import cv2
import time
import numpy as np
import pyrealsense2 as rs

import sys
sys.path.append('/home/metratec/Development/vivek_thesis/pose_estimation/lightweight_human_pose_estimation')

from realsense_device_manager_package.realsense_device_manager import DeviceManager
from realsense_device_manager_package.helper_functions import pixel_to_camera_coordinate

import torch

from threading import Thread

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class VideoGet:
    def __init__(self, args=None):
        self.args = args
        self.device_manager = DeviceManager()
        self.device_manager.enable_emitter(True)
        self.device_manager.enable_all_devices()
        self.frames_np = self.device_manager.poll_frames()
        self.stopped = False
        print("available devices..", self.device_manager.available_devices)

    def start(self):
        print("start..")
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            print("getting..")
            self.frames_np = self.device_manager.poll_frames()

    def stop(self):
        self.stopped = True


class VideoInfer:
    def __init__(self, args, frame_color=None, frame_depth=None):
        print("infer thread..")
        self.args = args
        self.cpu = self.args.cpu
        self.track = self.args.track
        self.frame_color = frame_color
        self.frame_depth = frame_depth
        self.stopped = False
        self.net = PoseEstimationWithMobileNet()
        self.checkpoint = torch.load(self.args.checkpoint_path, map_location='cpu')
        load_state(self.net, self.checkpoint)

        self.net = self.net.eval()
        if not self.args.cpu:
            print("moving to cuda")
            self.net = self.net.cuda()

        self.height_size = self.args.height_size
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.delay = 33
        self.all_keypoints = []
        self.frame_vis = None
        print("infer thread..")

    def start(self):
        Thread(target=self.infer, args=()).start()
        return self

    def infer(self):
        while not self.stopped:
            print("infering..")
            start_time = time.time()

            img = self.frame_color.copy()
            orig_img = img.copy()

            heatmaps, pafs, scale, pad = self.infer_fast(self.net, img, self.height_size, self.stride,
                                                         self.upsample_ratio, self.cpu)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(self.num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                         total_keypoints_num)

            pose_entries, self.all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)

            for kpt_id in range(self.all_keypoints.shape[0]):
                self.all_keypoints[kpt_id, 0] = (self.all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[
                    1]) / scale
                self.all_keypoints[kpt_id, 1] = (self.all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[
                    0]) / scale
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(self.num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(self.all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(self.all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)

            for pose in current_poses:
                pose.draw(img)
            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            fps = 1/(time.time()-start_time)
            cv2.putText(img, str(fps), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            # img_all.vstack(img)
            self.all_keypoints.tolist()
            self.frame_vis = img
            # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
            # cv2.waitKey(1)

    def stop(self):
        self.stopped = True
        pass

    def infer_fast(self, net, img, net_input_height_size, stride, upsample_ratio, cpu,
                   pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu:
            tensor_img = tensor_img.cuda()

        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()


    ####
    ####
    video_getter = VideoGet(args).start()
    video_infer = {}
    for device, frames in video_getter.frames_np.items():
        print("device..", device)
        video_infer[device] = VideoInfer(args, frames[rs.stream.color]).start()

    while True:
        print("pass..")
        for device, frames in video_getter.frames_np.copy().items():
            video_infer[device].frame_color = frames[rs.stream.color]
            video_infer[device].frame_depth = frames[rs.stream.depth]

        infer_img = []
        for device, infer in video_infer.items():
            print(type(infer.all_keypoints))
            if infer.frame_vis is not None:
                print("device pose..")
                infer_img.append(infer.frame_vis.copy())
                # infer.frame_vis = []
            else:
                print("device empty pose..")
                infer_img.append(infer.frame_color.copy())

        if len(infer_img):
            print("vis..")
            cv2.imshow("lightweight_pose", np.hstack(tuple(infer_img)))
            cv2.waitKey(1)