##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.		                      ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files 					  ####
##################################################################################################

import pyrealsense2 as rs
import numpy as np
import cv2

from collections import defaultdict
from realsense_device_manager_package.config import scale_factor, rsense_fps, dispose_frames_for_stablisation, chessboard_width, chessboard_height, \
    square_size, rsense_x, rsense_y
from realsense_device_manager_package.calibration_kabsch import PoseEstimation, Transformation, \
    calculate_transformation_quaternion
from realsense_device_manager_package.bundle_adjustment.bundle_adjustment import PyBundleAdjustment_cam_coord

"""
  _   _        _                      _____                     _    _                    
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
				 _|                                                                      
"""


class Device:
    def __init__(self, pipeline, pipeline_profile):
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile


def enumerate_connected_devices(context):
    """
    Enumerate the connected Intel RealSense devices

    Parameters:
    -----------
    context 	 	  : rs.context()
                         The context created for using the realsense library

    Return:
    -----------
    connect_device : array
                       Array of enumerated devices which are connected to the PC

    """
    connect_device = []
    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))
    return connect_device


def post_process_depth_frame(depth_frame, decimation_magnitude=1.0, spatial_magnitude=2.0, spatial_smooth_alpha=0.5,
                             spatial_smooth_delta=20, temporal_smooth_alpha=0.4, temporal_smooth_delta=20):
    """
    Filter the depth frame acquired using the Intel RealSense device

    Parameters:
    -----------
    depth_frame 	 	 	 : rs.frame()
                               The depth frame to be post-processed
    decimation_magnitude : double
                              The magnitude of the decimation filter
    spatial_magnitude 	 : double
                            The magnitude of the spatial filter
    spatial_smooth_alpha	 : double
                            The alpha value for spatial filter based smoothening
    spatial_smooth_delta	 : double
                            The delta value for spatial filter based smoothening
    temporal_smooth_alpha : double
                            The alpha value for temporal filter based smoothening
    temporal_smooth_delta : double
                            The delta value for temporal filter based smoothening


    Return:
    ----------
    filtered_frame : rs.frame()
                       The post-processed depth frame
    """

    # Post processing possible only on the depth_frame
    # assert (depth_frame.is_depth_frame())

    # Available filters and control options for the filters
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    filter_magnitude = rs.option.filter_magnitude
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta

    # Apply the control parameters for the filter
    decimation_filter.set_option(filter_magnitude, decimation_magnitude)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)

    return filtered_frame


"""
  __  __         _           ____               _                _   
 |  \/  |  __ _ (_) _ __    / ___| ___   _ __  | |_  ___  _ __  | |_ 
 | |\/| | / _` || || '_ \  | |    / _ \ | '_ \ | __|/ _ \| '_ \ | __|
 | |  | || (_| || || | | | | |___| (_) || | | || |_|  __/| | | || |_ 
 |_|  |_| \__,_||_||_| |_|  \____|\___/ |_| |_| \__|\___||_| |_| \__|

"""


class DeviceManager:
    def __init__(self):
        # device_manager = DeviceManager(rs.context(), rs_config)

        """
        Class to manage the Intel RealSense devices

        """

        self.scale_factor = scale_factor
        self.master_serial_number = 0
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # Define some constants
        resolution_width = rsense_x  # pixels
        resolution_height = rsense_y  # pixels
        frame_rate = rsense_fps  # fps
        # dispose_frames_for_stablisation = 15  # frames

        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        # Create colorizer object
        self.fps = int(frame_rate)
        self.frameSize = (resolution_width, resolution_height)
        self.fourcc = "MPEG"
        self.colorizer = rs.colorizer()
        self._context = rs.context()
        self._available_devices = enumerate_connected_devices(rs.context())
        self.available_devices = self._available_devices
        self.no_devices = len(self._available_devices)
        self._enabled_devices = {}
        self._config = rs_config
        self._frame_counter = 0
        self.intrinsics_devices = []
        self.depth_scale = 0.0010000000474974513  # for meters
        self.calibration_info_devices = defaultdict(list)
        self.bundle_params = []
        self.frame_time_stamp = 0.0
        self.delta_time = 0.0

    def enable_device(self, device_serial, enable_ir_emitter):
        """
        Enable an Intel RealSense Device

        Parameters:
        -----------
        device_serial 	 : string
                             Serial number of the realsense device
        enable_ir_emitter : bool
                            Enable/Disable the IR-Emitter of the device

        """
        self.pipeline = rs.pipeline()
        # Enable the device

        try:
            self._config.enable_device(device_serial)
            pipeline_profile = self.pipeline.start(self._config)
        except Exception as error:
            print("Error in enable device ", error)

        # 0.0010000000474974513 for meters
        # depth_scale = pipeline_profile.get_device().first_depth_sensor().get_depth_scale()
        # self.depth_scale[device_serial] = depth_scale
        # Set the acquisition parameters
        self.intrinsics_ir = pipeline_profile.get_stream(rs.stream.infrared,
                                                         1).as_video_stream_profile().get_intrinsics()

        sensor = pipeline_profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)
        self._enabled_devices[device_serial] = (Device(self.pipeline, pipeline_profile))

    def enable_all_devices(self, enable_ir_emitter=False):
        """
        Enable all the Intel RealSense Devices which are connected to the PC
        """
        print(str(self._available_devices) + " ---available devices")

        for serial in self._available_devices:
            self.enable_device(serial, enable_ir_emitter)

    def set_master_and_slave(self):
        """
        Set the master and slave in the list of devices detected
        :return:
        """
        master_set = False
        for (device_serial, device) in self._enabled_devices.items():
            sensor = device.pipeline_profile.get_device().first_depth_sensor()
            if master_set:
                sensor.set_option(rs.option.inter_cam_sync_mode, 2)
            else:
                sensor.set_option(rs.option.inter_cam_sync_mode, 1)
                self.master_serial_number = device_serial
                master_set = True

    def enable_emitter(self, enable_ir_emitter=True):
        """
        Enable/Disable the emitter of the intel realsense device

        """

        for (device_serial, device) in self._enabled_devices.items():
            # Get the active profile and enable the emitter for all the connected devices
            sensor = device.pipeline_profile.get_device().first_depth_sensor()
            if self.master_serial_number == device_serial:
                sensor.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)
                continue
            sensor.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)

            if enable_ir_emitter:
                sensor.set_option(rs.option.laser_power, 330)

    def load_settings_json(self, path_to_settings_file):
        """
        Load the settings stored in the JSON file

        """

        file = open(path_to_settings_file, 'r')
        json_text = file.read().strip()
        file.close()

        for (device_serial, device) in self._enabled_devices.items():
            # Get the active profile and load the json file which contains settings readable by the realsense
            device = device.pipeline_profile.get_device()
            advanced_mode = rs.rs400_advanced_mode(device)
            advanced_mode.load_json(json_text)

    def calibrate_homography(self, evaluate=False):
        """

        calibrate the homography for all the detected cameras and get the transformation_devices, intrinsics_devices,
        extrinsics_devices
        :return:
        calibration_info_devices : dict
            keys: str
                Serial number of the device
            values: [transformation_devices, intrinsics_devices, extrinsics_devices]
            transformation_devices: Transformation object
                    The transformation object containing the transformation information between the device and the world coordinate systems
            intrinsics_devices: rs.intrinsics
                    The intrinsics of the depth_frame of the realsense device
            extrinsics_devices: rs.extrinsics
                    The extrinsics between the depth imager 1 and the color imager of the realsense device

        """

        # Allow some frames for the auto-exposure controller to stablise
        for frame in range(dispose_frames_for_stablisation):
            _frames = self.poll_frames(return_type='rs')

        assert (len(self._available_devices) > 0)
        """
        1: Calibration
        Calibrate all the available devices to the world co-ordinates.
        For this purpose, a chessboard printout for use with opencv based calibration process is needed.

        """
        # Get the intrinsics of the realsense device
        intrinsics_devices = self.get_device_intrinsics(_frames)

        # Set the chessboard parameters for calibration
        chessboard_params = [chessboard_height, chessboard_width, square_size]

        # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
        extern_calib_points = {}
        for device in self._available_devices:
            extern_calib_points[device] = [[], [], []]
        no_of_images = 1   # int(input("no. of images for calibration"))

        ################
        count = 0
        calibrated_device_count = 0
        transformation_result_kabsch = {}
        while count < no_of_images:
            try:
                while calibrated_device_count < len(self._available_devices):
                    # 1) Create a PoseEstimation object
                    frames = self.poll_frames()
                    pose_estimator = PoseEstimation(frames, intrinsics_devices,
                                                    chessboard_params)
                    # 2) Call function perform_pose_estimation of PoseEstimation
                    transformation_result_kabsch, chessboard_images = pose_estimator.perform_pose_estimation()
                    for device in self._available_devices:
                        if not transformation_result_kabsch[device][0]:
                            calibrated_device_count = 0
                            print("Place the chessboard on the plane where the object needs to be detected..")
                        else:
                            calibrated_device_count += 1
            except Exception as error:
                print("[ERROR] Error in getting the chessboard corners, ", error)

            for device, val in transformation_result_kabsch.items():
                extern_calib_points[device][0].extend(val[5].transpose())  # cum the ref points or object points
                extern_calib_points[device][1].extend(val[8].transpose())  # cum the cam points or observed points
                extern_calib_points[device][2].extend(val[2].transpose())  # cum the 2D points or observed points

            calibrated_device_count = 0
            count = count + 1
        ################

        try:
            for device in self._available_devices:
                valid_object_points = np.array(extern_calib_points[device][0]).transpose()
                valid_observed_object_points = np.array(extern_calib_points[device][1]).transpose()

                [rotation_matrix, translation_vector, rmsd_value] = calculate_transformation_quaternion(
                    valid_object_points, valid_observed_object_points)
                rvecs = cv2.Rodrigues(rotation_matrix)
                param_ba = np.hstack((np.array(rvecs[0]).flatten(), translation_vector,
                                      intrinsics_devices[device][(rs.stream.infrared, 1)].fx, 0, 0,
                                      intrinsics_devices[device][(rs.stream.infrared, 1)].ppx,
                                      intrinsics_devices[device][(rs.stream.infrared, 1)].ppy))
                transformation_result_kabsch[device][1] = Transformation(rotation_matrix, translation_vector)
                transformation_result_kabsch[device][8] = valid_observed_object_points
                transformation_result_kabsch[device][2] = np.array(extern_calib_points[device][2])
                transformation_result_kabsch[device][5] = valid_object_points
                transformation_result_kabsch[device][6] = param_ba
        except Exception as error:
            print("[ERROR] Error in pre bundle adjustment", error)


        # cv2.destroyWindow("source_calibration")
        # Save the transformation object for all devices in an array to use for measurements

        """
        Perform bundle adjustment to reduce rerojection error from 2d to 3d
        """

        camera_params = []
        camera_array = []
        optical_center = []
        points_2d = []
        points_cam_3d = []

        camera_indices = []
        point_indices = []

        for k, (device, val) in enumerate(transformation_result_kabsch.items()):
            if k == 0:
                points_3d = np.transpose(val[5])
            _params = val[6]
            self.bundle_params.append(val[6][:9])
            _points_2d = val[2].transpose(1, 0, 2).squeeze()
            camera_params.append(_params[0:6])
            optical_center.append(_params[9:11])
            points_2d.extend(_points_2d)
            points_cam_3d.extend(np.transpose(val[8]))
            camera_indices.extend(np.repeat(k, val[8].shape[1]))
            point_indices.extend(np.arange(0, val[8].shape[1], 1))

        camera_params = np.array(camera_params)
        optical_center = np.array(optical_center)
        points_2d = np.array(points_2d)
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)
        points_cam_3d = np.array(points_cam_3d)

        # def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices):
        # bundle_adj = PyBundleAdjustment(camera_params, points_3d, points_2d, camera_indices, point_indices)
        # params_ = bundle_adj.bundleAdjust()

        bundle_adjustment = PyBundleAdjustment_cam_coord(camera_params, points_3d, points_2d, camera_indices, point_indices,
                                                         optical_center, points_cam_3d)
        params_all, points_proj = bundle_adjustment.bundleAdjust()
        params = params_all[0]
        transformation_devices = {}
        for k, device in enumerate(self._available_devices):
            r_mat = cv2.Rodrigues(params[k, :3])[0]
            t_vec = params[k, 3:6]
            transformation_devices[device] = Transformation(r_mat, t_vec)  # .inverse()
            print(transformation_devices[device])
            # transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
        # To set the uncalibrated posemat

        img_points = []
        intrinsics_matrix = []
        for k, (device, val) in enumerate(transformation_result_kabsch.items()):
            if k == 0:
                obj_points = val[5].transpose().astype("float32")
            img_points.append(np.array(val[2].transpose(2, 1, 0).astype(int)))
            _intrinsics = intrinsics_devices[device][rs.stream.depth]
            intrinsics_matrix.append(np.array(
                [[_intrinsics.fx, 0, _intrinsics.ppx], [0, _intrinsics.fy, _intrinsics.ppy], [0, 0, 1]]).astype(
                dtype=float))

        # Uncomment below to override bundle adjustment
        """
        transformation_devices = {}
        for device in self._available_devices:
            transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
        """
        # cv2.destroyWindow("source_ir")
        print("Calibration completed... ")

        # Enable the emitter of the devices
        # self.enable_emitter(True)
        # self.set_master_and_slave()

        # Load the JSON settings file in order to enable High Accuracy preset for the realsense
        # self.pointload_settings_json("./HighResHighAccuracyPreset.json")

        # Get the extrinsics of the device to be used later
        # extrinsics_devices = self.get_depth_to_color_extrinsics(frames)
        extrinsics_devices = self.get_depth_to_ir_extrinsics(self.poll_frames(return_type="rs"))

        if not evaluate:
            for j, calibration_info in enumerate((transformation_devices, intrinsics_devices, extrinsics_devices)):
                for k, (key, value) in enumerate(calibration_info.items()):

                    if j == 1:
                        """
                        value1 = value[(rs.stream.infrared, 1)]
                        value1.fx = params[k, 6]
                        value1.fy = params[k, 6]
                        value1.ppx = params[k, 7]
                        value1.ppy = params[k, 8]

                        value2 = value[rs.stream.depth]
                        value2.fx = params[k, 6]
                        value2.fy = params[k, 6]
                        value2.ppx = params[k, 7]
                        value2.ppy = params[k, 8]
                        """

                    self.calibration_info_devices[key].append(value)
        if evaluate:
            transformed_to_ref_frame_points = {}
            pixel_coordinates = {}
            evaluate_transforamtion = {}
            self.calibration_info_devices = defaultdict(list)

            for j, calibration_info in enumerate((transformation_devices, intrinsics_devices, extrinsics_devices)):
                for k, (key, value) in enumerate(calibration_info.items()):
                    if j == 1:
                        """
                        value1 = value[(rs.stream.infrared, 1)]
                        value1.fx = params[k, 6]
                        value1.fy = params[k, 6]
                        value1.ppx = params[k, 7]
                        value1.ppy = params[k, 8]

                        value2 = value[rs.stream.depth]
                        value2.fx = params[k, 6]
                        value2.fy = params[k, 6]
                        value2.ppx = params[k, 7]
                        value2.ppy = params[k, 8]
                        """

                    self.calibration_info_devices[key].append(value)

            for device in self._available_devices:
                points2D = transformation_result_kabsch[device][2]
                pixel_coordinate = []
                transformed_to_ref_frame_points[device] = transformation_devices[device].apply_transformation(
                    transformation_result_kabsch[device][4])
                points3D_ref_frame = transformed_to_ref_frame_points[device]
                camera_coordinate = np.transpose(transformed_to_ref_frame_points[device])
                for _camera_coordinate in camera_coordinate:
                    _pixel_coordinate = rs.rs2_project_point_to_pixel(
                        intrinsics_devices[device][rs.stream.depth],
                        [_camera_coordinate[0], _camera_coordinate[1],
                         _camera_coordinate[2]])
                    try:
                        pixel_coordinate.append([round(_pixel_coordinate[1]), round(_pixel_coordinate[0])])
                    except Exception as error:
                        print(error)
                pixel_coordinates[device] = pixel_coordinate
                points2D_ref_frame = pixel_coordinate
                evaluate_transforamtion[device] = [points2D, points2D_ref_frame,
                                                   transformation_result_kabsch[device][4], points3D_ref_frame,
                                                   transformation_result_kabsch[device][5]]
            return evaluate_transforamtion, frames


        """
        # Store the calibrated intrinsics and calibration_info_devices
        intrinsics_file = open('calibration_device_info/intrinsics_devices.pkl', 'rb')
        pickle.dump(intrinsics_devices, intrinsics_file)

        calibration_info_devices_file = open('calibration_device_info/calibration_info_devices.pkl', 'rb')
        pickle.dump(self.calibration_info_devices, calibration_info_devices_file)

        """

    def poll_frames(self, return_type="np"):
        """
        Poll for frames from the enabled Intel RealSense devices.
        If temporal post processing is enabled, the depth stream is averaged over a certain amount of frames

        Parameters:
        -----------

        """
        frames = {}
        source = []
        for (serial, device) in self._enabled_devices.items():
            streams = device.pipeline_profile.get_streams()
            # device.pipeline.poll_for_frames(frameset)
            try:
                frameset = device.pipeline.wait_for_frames()
            except Exception as error:
                print("[ERROR] Error while getting the frame from relasense camera, ", error)
                # If frame fails continue to the next device
                continue
            if frameset.size() == len(streams):
                frames[serial] = {}
                for stream in streams:
                    if rs.stream.infrared == stream.stream_type():
                        frame = frameset.get_infrared_frame(stream.stream_index())
                        key_ = (stream.stream_type(), stream.stream_index())
                        source.append(np.array(frame.get_data()))
                    elif rs.stream.color == stream.stream_type():
                        frame = frameset.get_color_frame()
                        key_ = stream.stream_type()
                        # color_frame = np.asanyarray(frame.get_data())
                        # print(color_frame.shape)
                        # cv2.imshow("color frame"+serial, color_frame)
                        # cv2.waitKey(1)
                    else:
                        frame = frameset.first_or_default(stream.stream_type())
                        key_ = stream.stream_type()
                    # print(key_)
                    frames[serial][key_] = frame
        self.intrinsics_devices = self.get_device_intrinsics(frames)
        if return_type == "np":
            frames_np = {}
            for (serial, frames_rs) in frames.items():
                frames_np[serial] = {}
                for (key_, frame_rs) in frames_rs.items():
                    if type(key_) == type(rs.stream.depth):
                        if rs.stream.depth == key_:
                            frame_np = post_process_depth_frame(frame_rs)
                    frame_np = np.asanyarray(frame_rs.get_data())
                    frames_np[serial][key_] = frame_np
                    # cv2.resize(frame_np, None, fx=self.scale_factor, fy=self.scale_factor)

            source = np.hstack(tuple(source))
            # cv2.imshow("source_ir", source)
            # cv2.waitKey(1)
            return frames_np
        else:
            source = np.hstack(tuple(source))
            if __debug__:
                pass
                # cv2.imshow("source_calibration", source)
                # cv2.waitKey(1)
        return frames

    def get_device_intrinsics(self, frames):
        """
        Get the intrinsics of the imager using its frame delivered by the realsense device

        Parameters:
        -----------
        frames : rs::frame
                  The frame grabbed from the imager inside the Intel RealSense for which the intrinsic is needed

        Return:
        -----------
        device_intrinsics : dict
        keys  : serial
                Serial number of the device
        values: [key]
                Intrinsics of the corresponding device
        """
        device_intrinsics = {}
        for (serial, frameset) in frames.items():
            device_intrinsics[serial] = {}
            for key, value in frameset.items():
                device_intrinsics[serial][key] = value.get_profile().as_video_stream_profile().get_intrinsics()
        return device_intrinsics

    def get_depth_to_color_extrinsics(self, frames):
        """
        Get the extrinsics between the depth imager 1 and the color imager using its frame delivered by the realsense device

        Parameters:
        -----------
        frames : rs::frame
                  The frame grabbed from the imager inside the Intel RealSense for which the intrinsic is needed

        Return:
        -----------
        device_intrinsics : dict
        keys  : serial
                Serial number of the device
        values: [key]
                Extrinsics of the corresponding device
        """
        device_extrinsics = {}
        for (serial, frameset) in frames.items():
            device_extrinsics[serial] = frameset[
                rs.stream.depth].get_profile().as_video_stream_profile().get_extrinsics_to(
                frameset[rs.stream.color].get_profile())
            temp_variable = frameset[rs.stream.depth].get_profile().as_video_stream_profile()
        return device_extrinsics

    def get_depth_to_ir_extrinsics(self, frames):
        """
        Get the extrinsics between the depth imager 1 and the ir imager using its frame delivered by the realsense device

        Parameters:
        -----------
        frames : rs::frame
                  The frame grabbed from the imager inside the Intel RealSense for which the intrinsic is needed

        Return:
        -----------
        device_intrinsics : dict
        keys  : serial
                Serial number of the device
        values: [key]
                Extrinsics of the corresponding device
                 self.intrinsics_color = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics_ir = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
        """
        device_extrinsics = {}
        for (serial, frameset) in frames.items():
            print(frameset[
                rs.stream.depth].get_profile().as_video_stream_profile().get_extrinsics_to(
                frameset[(rs.stream.infrared, 1)].get_profile()))
            device_extrinsics[serial] = frameset[
                rs.stream.depth].get_profile().as_video_stream_profile().get_extrinsics_to(
                frameset[(rs.stream.infrared, 1)].get_profile())
        return device_extrinsics

    def disable_streams(self):
        self._config.disable_all_streams()

    def get_device_intrinsics(self, frames):
        """
        Get the intrinsics of the imager using its frame delivered by the realsense device

        Parameters:
        -----------
        frames : rs::frame
                  The frame grabbed from the imager inside the Intel RealSense for which the intrinsic is needed

        Return:
        -----------
        device_intrinsics : dict
        keys  : serial
                Serial number of the device
        values: [key]
                Intrinsics of the corresponding device
        """
        device_intrinsics = {}
        for (serial, frameset) in frames.items():
            device_intrinsics[serial] = {}
            for key, value in frameset.items():
                device_intrinsics[serial][key] = value.get_profile().as_video_stream_profile().get_intrinsics()
        return device_intrinsics


"""
  _____           _    _               
 |_   _|___  ___ | |_ (_) _ __    __ _ 
   | | / _ \/ __|| __|| || '_ \  / _` |
   | ||  __/\__ \| |_ | || | | || (_| |
   |_| \___||___/ \__||_||_| |_| \__, |
									   |___/ 

"""

if __name__ == "__main__":
    try:
        c = rs.config()
        c.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        c.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 6)
        c.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 6)
        c.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
        device_manager = DeviceManager()
        device_manager.enable_all_devices()
        for k in range(150):
            frames = device_manager.poll_frames()
        device_manager.enable_emitter(True)
        # device_extrinsics = device_manager.get_depth_to_color_extrinsics(frames)
    finally:
        device_manager.disable_streams()