import argparse

import cv2
import numpy as np
import pyrealsense2 as rs
import time
from realsense_device_manager_package.realsense_device_manager import DeviceManager
from realsense_device_manager_package.helper_functions import pixel_to_camera_coordinate

from person_reid.person_reid import PersonReid

import torch

from threading import Thread
from mqtt_manager.tojson import ToJson
from mqtt_manager.mqtt_client import MqttClient


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
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            self.frames_np = self.device_manager.poll_frames()

    def stop(self):
        self.stopped = True


class VideoInfer:
    def __init__(self, args, frames_np=None):
        print("infer thread..")
        self.args = args
        self.cpu = self.args.cpu
        self.track = self.args.track
        self.frames_np = frames_np
        self.frame_color = []
        self.frame_depth = None
        self.stopped = False
        self.intrinsics_devices = None
        self.depth_scale = 0
        self.pose3d_json = {}

        self.net = PoseEstimationWithMobileNet()
        self.checkpoint = torch.load(self.args.checkpoint_path, map_location='cpu')
        load_state(self.net, self.checkpoint)

        self.net = self.net.eval()
        if not self.args.cpu:
            print("moving to cuda")
            self.net = self.net.cuda()
        
        # Load person reid model
        self.person_reid = PersonReid(self.args.reid_model)

        self.height_size = self.args.height_size
        self.stride = 8
        self.upsample_ratio = 4
        self.img_shape_orig = []
        self.img_shape_new = []
        self.num_keypoints = Pose.num_kpts
        self.delay = 33
        self.all_keypoints = []
        self.frame_vis = []

        self.pose3d_json = {}
        self.mqtt_client = MqttClient()
        self.feature_vectors = {}

    def start(self):
        Thread(target=self.infer, args=()).start()
        return self

    def infer(self):
        while not self.stopped:
            images = self.frames_np.copy()
            # images_orig = images.copy()
            start_time = time.time()
            heatmaps, pafs, scale, pad = self.infer_fast(self.net, images, self.height_size, self.stride,
                                                         self.upsample_ratio, self.cpu)
            fps = int(1/(time.time() - start_time))
            print("fps..", fps, time.time() - start_time)
            frame_vis = []
            self.frame_color = []
            img_all = []
            for k, (device, frames) in enumerate(images.items()):

                img = frames[rs.stream.color]
                depth_frame = frames[rs.stream.depth]
                img_all.append(img.copy())


                orig_img = img.copy()

                total_keypoints_num = 0
                all_keypoints_by_type = []
                for kpt_idx in range(self.num_keypoints):  # 19th for bg
                    total_keypoints_num += extract_keypoints(heatmaps[k, :, :, kpt_idx], all_keypoints_by_type,
                                                             total_keypoints_num)

                pose_entries, self.all_keypoints = group_keypoints(all_keypoints_by_type, pafs[k], demo=True)

                for kpt_id in range(self.all_keypoints.shape[0]):
                    self.all_keypoints[kpt_id, 0] = (self.all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio -
                                                     pad[
                                                         1]) / scale
                    self.all_keypoints[kpt_id, 1] = (self.all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio -
                                                     pad[
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

                    # Get 3d points
                    # convert to json
                    # print(pixel_to_camera_coordinate(pose.keypoints, self.intrinsics_devices[device][rs.stream.depth], depth_frame, self.depth_scale))
                    pose.draw(img)
                    ## To get 3D coordinates from realsense depth
                    # TODO set arg for
                    if self.args.stereo:
                        self.pose3d_json[device] = ToJson(1, pixel_to_camera_coordinate(pose.keypoints, self.intrinsics_devices[device], depth_frame, self.depth_scale)).toJson()
                    else:
                        # TODO implement module to get the feature vector for each person detected

                        # Step 1: Using pose guided cropping strategy get the persons in the frame
                        print(pose.bbox)
                        # Step 2: Extract features for each person  using extract_features() function, takes input height 256, width 128
                        self.pose3d_json[device] = ToJson(1, pixel_to_camera_coordinate(pose.keypoints, self.intrinsics_devices[device], depth_frame, self.depth_scale)).toJson()
                    
                    self.mqtt_publish(device, self.pose3d_json[device])
                
                # Calculates the weighted sum of two arrays
                img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
                for pose in current_poses:
                    cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                                  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                # img_all.vstack(img)

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (20, 20)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2

                cv2.putText(img, str(fps), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                self.all_keypoints.tolist()
                frame_vis.append(img)
                # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
                # cv2.waitKey(1)
            self.frame_vis = frame_vis

            # TODO 
            # compute the distance matrix between the querry frame and other frame
            # Step 1: Iterate over self.feature_vectors
            # Step 2: First frame's feature vectors are taken as querry frame
            # Step 3: Compute the distance matrix with other frames using compute_distance_matrix(qf, feature_vectors) 
            # Step 4: get the min arg(np.argmin(distance_matrix, dim=1)) and value(np.amin(distance_matrix, dim=1)) from distance matrix 
            ######################################################################
            self.frame_color = img_all.copy()

    def stop(self):
        self.stopped = True

    def infer_fast(self, net, images, net_input_height_size, stride, upsample_ratio, cpu,
                   pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):

        tensor_img_all = []
        for device, frames in images.items():
            img = frames[rs.stream.color]
            height, width, _ = img.shape
            self.img_shape_orig = img.shape
            scale = net_input_height_size / height

            scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            scaled_img = normalize(scaled_img, img_mean, img_scale)
            min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
            padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
            self.img_shape_new = scaled_img.shape

            tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
            tensor_img_all.append(tensor_img)


        tensor_img_all = torch.cat(tensor_img_all, dim=0)
        if not cpu:
            tensor_img_all = tensor_img_all.cuda()

        # print("input shape..", tensor_img_all.shape)
        start_time_infer = time.time()
        stages_output = net(tensor_img_all)
        print("infer fps..", 1/(time.time() - start_time_infer), (time.time() - start_time_infer))
        # TODO 
        ## Inference time reduces by half due to post processing
        ## To be optimised
        stage2_heatmaps = stages_output[-2]
        # heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = np.transpose(stage2_heatmaps.cpu().data.numpy(), (0, 2, 3, 1))
        heatmaps_list = []
        for k in range(0, heatmaps.shape[0]):
            heatmaps_list.append(torch.from_numpy(cv2.resize(heatmaps[k], (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)).unsqueeze(dim=0))
        heatmaps = torch.cat((heatmaps_list), dim=0).data.numpy()

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.cpu().data.numpy(), (0, 2, 3, 1))
        pafs_list = []
        for k in range(0, stage2_pafs.shape[0]):
            pafs_list.append(torch.from_numpy(cv2.resize(pafs[k], (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)).unsqueeze(dim=0))

        pafs = torch.cat((pafs_list), dim=0).data.numpy()
        return heatmaps, pafs, scale, pad

    def publish(self):
        print("pub fun..")
        while not self.stopped:
            print(".......publisher......")
            pose3d_json = self.pose3d_json.copy()
            for device, pose3d_json in pose3d_json.items():
                print(pose3d_json)
                self.mqtt_publish(device, pose3d_json)

    def mqtt_publish(self, device, pose3d_json):
        topic = "/pose_cam/"+device+"/pose_3d"
        print("topic...", topic)
        self.mqtt_client.publish(topic, pose3d_json)


class PosePublisher:
    def __init__(self):
        self.pose3d_json = {}
        self.mqtt_client = MqttClient()
        self.stopped = False
        print("publisher thread..")

    def start(self):
        Thread(target=self.publish, args=()).start()
        return self

    def publish(self):
        print("pub fun..")
        while not self.stopped:
            print(".......publisher......")
            pose3d_json = self.pose3d_json.copy()
            for device, pose3d_json in pose3d_json.items():
                print(pose3d_json)
                self.mqtt_publish(device, pose3d_json)

    def stop(self):
        self.stopped = True

    def mqtt_publish(self, device, pose3d_json):
        topic = "/pose_cam/"+device+"/pose_3d"
        print("topic...", topic)
        self.mqtt_client.publish(topic, pose3d_json)


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
    parser.add_argument('--stereo', action='store_true', help='Depth from stereo')
    parser.add_argument('--reid_model', type=str, required=False, default='pretrained_model/resnet50_person_reid.pth', help='reid model path')

    args = parser.parse_args()


    ####
    ####
    video_getter = VideoGet(args).start()
    video_infer = VideoInfer(args, video_getter.frames_np).start()
    video_infer.intrinsics_devices = video_getter.device_manager.intrinsics_devices
    video_infer.depth_scale = video_getter.device_manager.depth_scale

    # pose_pub = PosePublisher().start()

    while True:
        video_infer.frames_np = video_getter.frames_np
        infer_img = []
        if video_infer.frame_vis:
            frame_vis = video_infer.frame_vis.copy()
            # pose_pub.pose3d_json = video_infer.pose3d_json

            cv2.imshow("vis", np.hstack(tuple(frame_vis)))
        cv2.waitKey(1)
