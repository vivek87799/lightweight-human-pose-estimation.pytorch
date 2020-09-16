import logging

logging.basicConfig(level=logging.INFO)
marker_count = 3
# source = 0
source = "output.avi"
cam_calibration_filename = 'cam_calibration_ouput.npz'
log_filename='poseval3d.txt'
log_filename_rotation = 'rotation.txt'
log_filename_orientation='orientation.txt'

delta_time = 0.033
boundingbox_threshold = 80
scale_factor = 1  # 0 > scale_factor <=1
rsense_fps = 30  # 15, 30
rsense_x = 640  # 848  # 640 , 1280
rsense_y = 480  # 480  # 480, 720

dispose_frames_for_stablisation = 30
chessboard_width = 8  # 5  # squares
chessboard_height = 6 	# squares
square_size = 0.05 #0.077  # 0.0148  # 0.0253 # meters

depth_scale = 0.0010000000474974513  # for meters
