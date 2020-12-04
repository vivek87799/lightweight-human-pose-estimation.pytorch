import argparse
import copy
import queue as Queue
import cv2
import numpy as np
import pyrealsense2 as rs
import time

from realsense_device_manager_package.realsense_device_manager import DeviceManager
from realsense_device_manager_package.helper_functions import pixel_to_camera_coordinate

from helper_functions_pose_estimation import processing_loop, depth_from_triangulation

from person_reid.person_reid import PersonReid
from scipy.optimize import linear_sum_assignment

import torch

from threading import Thread
from mqtt_manager.tojson import SkeletonPoseToJson, SkeletonIDTrackerToJson
from mqtt_manager.mqtt_client import MqttClient


from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

from tracking.SkeletonsTracker import SkeletonsTracker


class VideoGet:
    def __init__(self, args=None):
        self.args = args
        self.device_manager = DeviceManager()
        self.device_manager.enable_emitter(True)
        self.device_manager.enable_all_devices()
        self.frames_np = self.device_manager.poll_frames()
        self.device_manager.calibrate_homography()
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


class VideoGetCMU:
    def __init__(self, streams=None):
        self.caps = []
        self.frames = {}
        self.frames_queue = Queue.Queue(maxsize=105)
        self.stopped = False
        self.streams = streams
        self.frame_vis_raw = None
        self.frame_count = 0
        for stream in streams:
            self.caps.append(cv2.VideoCapture(stream))
        # Start from 12400 fro visualization
        # for i in range(0, 13400):
        for i in range(0, 1800):
            for cap in self.caps:
                ret, frame = cap.read()
            self.frame_count = self.frame_count+1

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            self.frames = self.poll_frames()
    def wait_and_get_frames(self):
        return self.frames_queue.get()
    
    def wait_and_put_frames(self, frames):
        self.frames_queue.put(frames)
    def poll_frames(self):
        # self.frames = {}
        frames = {}
        stream_id = 0
        frame_vis_raw = []
        for cap in self.caps:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                frames[self.streams[stream_id]] = frame 
                stream_id = stream_id+1
                frame_vis_raw.append(frame)

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (40, 40)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2

                cv2.putText(frame, str(self.frame_count), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                # self.frames.append(frame)
            else:
                self.stopped = True
        self.frame_count = self.frame_count+1
        print("stream len ", len(self.streams), len(frames.keys()), self.frame_count)
        self.frames = copy.deepcopy(frames)
        self.wait_and_put_frames(self.frames)
        self.frame_vis_raw = frame_vis_raw
    def stop(self):
        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()
        self.stopped = True


class VideoInfer:
    def __init__(self, args, frames_np=None, calibration_info_devices=None, skeletonsTracker=None):
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
        self.skeletons_tracker = skeletonsTracker

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
        self.skeleton_ID_tracker_json = []
        self.mqtt_client = MqttClient()
        self.feature_vectors = {}
        self.calibration_info_devices = None
        self.reid_images_all = {}

    def start(self):
        Thread(target=self.infer, args=()).start()
        return self

    def infer(self):
        # while not self.stopped:
        
        images = copy.deepcopy(self.frames_np)
        
        if len(images.keys()) < 1:
            # print("no of images for inferene -->", len(images.keys()))
            return

        # images_orig = images.copy()
        start_time = time.time()
        heatmaps, pafs, scale, pad = self.infer_fast(self.net, images, self.height_size, self.stride,
                                                        self.upsample_ratio, self.cpu)
        fps = int(1/(time.time() - start_time))
        print("fps..", fps, time.time() - start_time)
        frame_vis = []
        self.frame_color = []
        img_all = []
        poses_all = {}
        camera_pose = []
        empty_frame_flag = False
        estimate_3d_pose_flag = False
        reid_images_all = {}
        for k, (device, frames) in enumerate(images.items()):
            reid_images = []
            img = frames
            # img = frames[rs.stream.color]
            # depth_frame = frames[rs.stream.depth]
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
            print("no of pose entries", len(pose_entries))
            if not len(pose_entries):
                print("no of pose entries")
                continue
            for n in range(len(pose_entries)):
                pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(self.num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(self.all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(self.all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                # find the median of left hip and right hip and add the center position
                pose.keypoints = pose.keypoints[:14]
                # Assigning an initial pose id
                pose.id = n
                current_poses.append(pose)

            feature_vectors = None
            for pose in current_poses:

                # Get 3d points
                # convert to json
                # print(pixel_to_camera_coordinate(pose.keypoints, self.intrinsics_devices[device][rs.stream.depth], depth_frame, self.depth_scale))
                pose.draw_14keypoints(img)
                ## To get 3D coordinates from realsense depth
                # TODO set arg for using stereo or triangulation
                if self.args.stereo:
                    pass
                    # self.pose3d_json[device] = SkeletonPoseToJson(1, pixel_to_camera_coordinate(pose.keypoints, self.intrinsics_devices[device], depth_frame, self.depth_scale)).toJson()
                    # self.mqtt_publish(device, self.pose3d_json[device])
                else:
                    # TODO implement module to get the feature vector for each person detected
                    # Step 1: Using pose guided cropping strategy get the persons in the frame
                    # bbox gives x, y, w, h
                    
                    img_= orig_img[pose.bbox[1]:pose.bbox[1]+pose.bbox[3], pose.bbox[0]:pose.bbox[0]+pose.bbox[2], :]
                    # cv2.imshow("reid"+device, img_)
                    img_cropped = torch.from_numpy(img_).cuda()
                    img_cropped = img_cropped.permute(2, 1, 0).unsqueeze(dim=0).type(torch.FloatTensor)
                    # img_cropped = torch.nn.functional.interpolate(img_cropped, size=(256, 128)) # , mode='nearest')
                    img_cropped = torch.nn.functional.interpolate(img_cropped, size=(128, 256))
                    # print(img[pose.bbox[0]:pose.bbox[0]+pose.bbox[2], pose.bbox[0]:pose.bbox[0]+pose.bbox[3], :].shape)
                    # convert the image to tensor and move to cuda 
                    # Step 2: Extract features for each person  using extract_features() function, takes input height 256, width 128
                    # self.feature_vectors[device] = self.person_reid.extract_features(img_cropped)
                    # self.pose3d_json[device] = SkeletonPoseToJson(1, pixel_to_camera_coordinate(pose.keypoints, self.intrinsics_devices[device], depth_frame, self.depth_scale)).toJson()
                    # Save the cropped images for visualization 
                    img_cropped_numpy = img_cropped.type(torch.cuda.ByteTensor).cpu().numpy().squeeze()
                    img_cropped_numpy = np.transpose(img_cropped_numpy, (2, 1, 0))
                    
                    reid_images.append(img_cropped_numpy)
                    if feature_vectors is not None:
                        feature_vectors = torch.cat((feature_vectors, self.person_reid.extract_features(img_cropped)))
                    else:
                        feature_vectors = self.person_reid.extract_features(img_cropped)
            cv2.imshow("reid_stacked"+device, np.hstack(tuple(reid_images)))
            reid_images_all[device] = reid_images
            
            # feature_vectors is assigned only when not of self.args.stereo
            if feature_vectors is not None:
                # Get the feature vector of all the detected persons
                self.feature_vectors[device] = feature_vectors
            
            
            # Calculates the weighted sum of two arrays
            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            _current_poses = []
            for pose in current_poses:
                _current_poses.append(np.array(pose.keypoints))
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                                (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            # img_all.vstack(img)
            poses_all[device] = np.array(_current_poses)
            # #######camera_pose.extend(self.calibration_info_devices[device][3])

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (20, 20)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2

            cv2.putText(img, str(fps), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            self.all_keypoints.tolist()
            frame_vis.append(img)

            # TODO Cheching if a person is detected else continue 
            if not current_poses:
                empty_frame_flag = True
                return
            # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
            # cv2.waitKey(1)
        self.reid_images_all = copy.deepcopy(reid_images_all)
        self.frame_vis = frame_vis

        # TODO 
        print("len of poses-->", len(poses_all.keys()))
        len_poses = len(poses_all.keys())
        if (len(poses_all.keys()) < 2):
            print("len of poses in if-->", len(poses_all.keys()))
            empty_frame_flag = True
            return
        if not empty_frame_flag:
            # compute the distance matrix between the querry frame and other frame
            # Step 1: Iterate over self.feature_vectors
            qf_set = False
            for device, feature_vectors in self.feature_vectors.items():
            # Step 2: First frame's feature vectors are taken as querry frame
                if not qf_set:
                    qf = feature_vectors
                    qf_keypoints_np = poses_all[device]
                    print("setting qf", torch.sum(qf))
                    qf_set = True
                    continue
                else:
                    gf_keypoints_np = poses_all[device]
            # Step 3: Compute the distance matrix with other frames using compute_distance_matrix(qf, feature_vectors)
                print("input shapes-->", qf.shape, feature_vectors.shape)
                distmat = self.person_reid.compute_distance_matrix(qf, feature_vectors) 
                print("distmat", distmat)
            # Step 4: Use Hungarian min algorithm to find the optimal solution from distance matrix 
                row_ind, col_ind = linear_sum_assignment(distmat)
                print("row_ind-->", row_ind)
                print("col_ind-->", col_ind)
                # argmin returns indices of min value along an axis 
                if qf.shape[0] >= feature_vectors.shape[0]:
                    # min along the rows for dim=1
                    
                    gf_ids = distmat.argmin(dim=1) 
                    gf_ids = col_ind 
                ## Step 5: Reorder the poses in the gallery frame
                    print(type(gf_keypoints_np), len(gf_keypoints_np), gf_keypoints_np.shape, gf_ids)
                    #### gf_keypoints_np = gf_keypoints_np[gf_ids]
                    print(type(gf_keypoints_np), len(gf_keypoints_np), gf_keypoints_np.shape)
                else:
                    # min along the cols for dim=0
                    # print(type(gf_keypoints_np), len(gf_keypoints_np))
                    gf_ids = distmat.argmin(dim=0) 
                    gf_ids = row_ind
                    # print(type(gf_keypoints_np), len(gf_keypoints_np), gf_keypoints_np.shape, gf_ids)
            ## Step 6: Reorder the poses in the query frame
                    ### qf_keypoints_np = qf_keypoints_np[gf_ids]
                
                # TODO correct implementation
                """
                gf_ids = col_ind
                qf_ids = row_ind
                qf_keypoints_np = qf_keypoints_np[[k for k in row_ind]]
                gf_keypoints_np = gf_keypoints_np[[k for k in col_ind]]
                """
                print("min values-->",gf_ids.tolist(), type(self.reid_images_all[device]))
                print(len(self.reid_images_all[device]))
                try:
                    gf_keypoints_np = gf_keypoints_np[gf_ids]
                    self.reid_images_all[device] = np.asarray(self.reid_images_all[device])[gf_ids] # .tolist()
                except Exception as error:
                    print("Erro in reordering", error)
                # print(self.reid_images_all[device].shape)
            # Step 7: Find the 3D pose using triangulation 
            qf_keypoints_np = qf_keypoints_np[[k for k in range(gf_keypoints_np.shape[0])]]
            pose_3d_json = SkeletonPoseToJson()
            skeleton_ID_tracker_json = SkeletonIDTrackerToJson()
            detections = []
            skeletons = []
            for k in range(0, qf_keypoints_np.shape[0]):
                skeleton = depth_from_triangulation(camera_pose_all_np, qf_keypoints_np[k].astype(np.float32), gf_keypoints_np[k].astype(np.float32))
                # detections.append(skeleton.joints_centre)
                skeletons.append(skeleton)
                # skeleton.joints is None if not assigned
                if type(skeleton.joints) is np.ndarray: 
                    pose_3d_json.add_pose(k, np.transpose(skeleton.joints))
            
            pose_3d_json = SkeletonPoseToJson()
            # Update the Kalman Filter
            self.skeletons_tracker.Update(skeletons)
            for i, track in enumerate(self.skeletons_tracker.tracks3d):
                skeleton_ID_tracker_json.add_skeleton_position(track.track_id, track.prediction.squeeze().tolist(), track.detection.squeeze().tolist())
                pose_3d_json.add_pose(track.track_id, np.transpose(track.joints))

            self.pose3d_json = pose_3d_json.toJson()
            self.skeleton_ID_tracker_json = skeleton_ID_tracker_json.toJson()
            self.publish()
        self.frame_color = img_all.copy()

    def stop(self):
        self.stopped = True

    def infer_fast(self, net, images, net_input_height_size, stride, upsample_ratio, cpu,
                   pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):

        tensor_img_all = []
        for device, frames in images.items():
            # img = frames[rs.stream.color]
            img = frames
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
        # print("infer fps..", 1/(time.time() - start_time_infer), (time.time() - start_time_infer))
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
        self.mqtt_publish("/pose_cam/triangulate/pose_3d_measured", self.pose3d_json)
        self.mqtt_publish("/pose_cam/triangulate/skeletonIDTracker", self.skeleton_ID_tracker_json)
            
    def mqtt_publish(self, topic, pose3d_json):
        print("topic...", topic)
        self.mqtt_client.publish(topic, pose3d_json)


class PosePublisher:
    def __init__(self):
        self.pose3d_json = {}
        self.mqtt_client = MqttClient()
        self.stopped = False

    def start(self):
        Thread(target=self.publish, args=()).start()
        return self

    def publish(self):
        while not self.stopped:
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
    parser.add_argument('--reid_model', type=str, required=False, default='pretrained_model/osnet_x1_0_market_256x128_person_reid.pth', help='reid model path')

    args = parser.parse_args()


    ####
    ####
    # video_getter = VideoGet(args).start()

    # For running inference on video from cmu panoptic dataset iterate over color frames of frames_np and replace it by the frames from the CMU stream
    ## streams = ["/home/metratec/Development/vivek_thesis/pose_estimation/cmu_haggle/hd_00_17.mp4", "/home/metratec/Development/vivek_thesis/pose_estimation/cmu_haggle/hd_00_18.mp4"]
    streams = ["/home/metratec/Development/vivek_thesis/pose_estimation/170228_haggling_b3/hd_00_10.mp4", "/home/metratec/Development/vivek_thesis/pose_estimation/170228_haggling_b3/hd_00_11.mp4"]
    # Step1: Register the video stream 
    
    #########
    # Get pose matrix
    pose_matrix = {}
    camera_pose = []
    _camera_pose = []
    camera_pose_all_np = None
    
    _camera_pose = []
    for name in ["00_10", "00_11"]:
        camera_pose = processing_loop(calib_file_name="../cmu_haggle/calibration_170228_haggling_b1.json", name=name)
        print("camera_pose-->", camera_pose)
        _camera_pose.append(np.expand_dims(camera_pose, axis=0))
    
    print("camera pose-->", np.vstack(tuple(_camera_pose)).shape)
    camera_pose_all_np= np.vstack(tuple(_camera_pose))

    video_getter_cmu = VideoGetCMU(streams)

    #########
    # video_infer = VideoInfer(args, video_getter_cmu.frames, video_getter.device_manager.calibration_info_devices).start()

    video_infer = VideoInfer(args, video_getter_cmu.frames, skeletonsTracker=SkeletonsTracker(5, 10, 5, 1))
    # video_infer = VideoInfer(args, video_getter_cmu.frames, skeletonIDTracker=Tracker(100, 50, 30, 1))
    # video_infer.intrinsics_devices = video_getter.device_manager.intrinsics_devices
    # video_infer.depth_scale = video_getter.device_manager.depth_scale

    # pose_pub = PosePublisher().start()
    

    while True:
        # Step2: Read the streams
        video_getter_cmu.poll_frames()
        if video_getter_cmu.frames_queue.empty():
            print("Empty queue")
            continue
        # print("len of frames received", len(video_getter_cmu.frames.keys()))
        video_infer.frames_np = video_getter_cmu.wait_and_get_frames()
        video_infer.infer()
        cv2.imshow("cmu_frames", np.hstack(tuple(video_getter_cmu.frame_vis_raw)))
        cv2.waitKey(1)
        # For running inference on video from cmu panoptic dataset iterate over color frames of frames_np and replace it by the frames from the CMU stream
        
        infer_img = []
        if video_infer.frame_vis:
            frame_vis = video_infer.frame_vis.copy()
            # pose_pub.pose3d_json = video_infer.pose3d_json
            cv2.imshow("vis", np.hstack(tuple(frame_vis)))
        
        if video_infer.reid_images_all:
            for device, reid_images in video_infer.reid_images_all.items():
                print(np.hstack(tuple(reid_images)).shape)
                cv2.imshow("reid_reordered"+device, np.hstack(tuple(reid_images)))
            video_infer.reid_images_all = {}
        
        cv2.waitKey(1)
