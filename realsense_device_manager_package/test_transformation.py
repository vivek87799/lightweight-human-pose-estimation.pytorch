# Import python libraries
import copy
import logging
import cv2
import pyrealsense2 as rs
import numpy as np
from realsense_device_manager import DeviceManager, post_process_depth_frame

from helper_functions import pixel_to_world_coordinate, pixel_to_camera_coordinate, world_to_pixel_coordinate


# Create a custom logger
logger = logging.getLogger(__name__)
logger1 = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('orientation_eval.log')
# f_handler = logging.FileHandler('logeval.txt')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.WARNING)

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger1.addHandler(f_handler)


rsense = DeviceManager()
# TODO check if calibration param requested
rsense.enable_all_devices()
rsense.set_master_and_slave()
rsense.enable_emitter(False)
rsense.load_settings_json("realsense_device_manager_package/HighResHighAccuracyPreset.json")

while(True):
    evaluate_transforamtion, frames = rsense.calibrate_homography(True)

    infrared_frames = []
    color = 0
    for device in rsense._enabled_devices:
        color = color + 100
        if len(infrared_frames) == 0:
            infrared_ref_frame = frames[device][(rs.stream.infrared, 1)]
            infrared_ref_frame = np.dstack((infrared_ref_frame, infrared_ref_frame, infrared_ref_frame))
            cv2.putText(infrared_ref_frame, "infrared_ref_frame"+device, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        infrared_frame = frames[device][(rs.stream.infrared, 1)]
        infrared_frame = np.dstack((infrared_frame, infrared_frame, infrared_frame))
        cv2.putText(infrared_frame, "infrared_frame" + device, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)

        points2D_ref = evaluate_transforamtion[device][1]  # .reshape(48, 2).astype(int).tolist()  # .reshape(2, 48)
        for _pixel_coordinate in points2D_ref:
            cv2.circle(infrared_ref_frame, tuple([round(_pixel_coordinate[1]), round(_pixel_coordinate[0])]), 5, (color, 0, color))

        print("[INFO] points2D_ref..", points2D_ref)
        print("[INFO] points..", np.squeeze(evaluate_transforamtion[device][0]).astype(int).tolist())
        for _pixel_coordinate in np.squeeze(evaluate_transforamtion[device][0]).astype(int).tolist():
            cv2.circle(infrared_frame, tuple([round(_pixel_coordinate[0]), round(_pixel_coordinate[1])]), 5, (0, color, color))
        infrared_frames.append(infrared_frame)

        # coord_test_ref_world = pixel_to_world_coordinate(points2D, rsense.calibration_info_devices[device], np.asarray(frames[device][rs.stream.depth].get_data()), rsense)
        # coord_test_ref_pixel = world_to_pixel_coordinate(np.transpose(evaluate_transforamtion[device][2]), rsense.calibration_info_devices[device])
    infrared_frames.append(infrared_ref_frame)
    cv2.imshow("re_plotted_marker_points", np.hstack((tuple(infrared_frames))))
