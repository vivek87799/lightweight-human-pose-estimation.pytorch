import logging


class DetectorParamValues:

    def __init__(self):
        self.threshold_value = 242
        self.blob_radius_thresh_min = 5
        self.blob_radius_thresh_max = 12
        self.marker_count = 2
        self.scale_factor = 1
        self.morph_kernel_size = 5
        self.morph_kernel_type = "cv2.MORPH_RECT"
        self.marker_distance_threshold = 1.0

    def update_config(self, cfg_value):
        # TODO call from the call back
        self.threshold_value = int(cfg_value["threshold_value"])
        self.blob_radius_thresh_min = float(cfg_value["blob_radius_thresh_min"])
        self.blob_radius_thresh_max = float(cfg_value["blob_radius_thresh_max"])
        self.marker_distance_threshold = float(cfg_value["marker_distance_threshold"])
        self.scale_factor = float(cfg_value["scale_factor"])
        self.morph_kernel_size = int(cfg_value["morph_kernel_size"])
        self.morph_kernel_type = str("cv2."+cfg_value["morph_kernel_type"])

class TrackerParamValues:

    # threshold_value = 242
    # blob_radius_thresh_min = 5
    # blob_radius_thresh_max = 12
    # marker_distance_threshold = 1.5
    # morph_kernel_size = 5
    # scale_factor = 1
    # morph_kernel_type = "cv2.MORPH_RECT"

    def __init__(self):
        self.marker_count = 2
        self.marker_distance_threshold = 1.5
        self.scale_factor = 1
        self.re_projection_error = 0.03
        self.frame_sync = 0.0001
        self.min_hit_criteria = 50
        self.max_frames_to_skip = 80

    def update_config(self, cfg_value):
        # TODO call from the call back
        self.marker_count = int(cfg_value["marker_count"])
        self.marker_dist_min_error = float(cfg_value["marker_dist_min_error"])
        self.marker_distance_threshold = float(cfg_value["marker_distance_threshold"])
        self.scale_factor = float(cfg_value["scale_factor"])


ConfigValuesObj = DetectorParamValues()

logging.basicConfig(level=logging.INFO)
marker_count = 3
# source = 0
source = "output.avi"
source = "rtsp://admin:HIK12345@192.168.2.164:554/Streaming/channels/1/"
# http://192.168.2.164/doc/page/login.asp?_1543928861357&page=config
source = "ROS"  # "multi_stereo"  # "realsense"
cam_calibration_filename = 'cam_calibration_ouput.npz'
log_filename='poseval3d.txt'
log_filename_rotation = 'rotation.txt'
log_filename_orientation='orientation.txt'

delta_time = 0.033
boundingbox_threshold = 80
scale_factor = 1  # 0 > scale_factor <=1
rsense_fps = 30  # 15, 30
rsense_x = 848  # 640 , 1280
rsense_y = 480   # 480, 720

dispose_frames_for_stablisation = 30
chessboard_width = 8  # 5  # squares
chessboard_height = 6 	# squares
square_size = 0.05 #0.077  # 0.0148  # 0.0253 # meters

depth_scale = 0.0010000000474974513  # for meters
