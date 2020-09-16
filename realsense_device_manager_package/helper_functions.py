##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.		                      ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files 					  ####
##################################################################################################

# Opencv helper functions and class
import logging as logger
import inspect
import cv2
import numpy as np
import pyrealsense2 as rs

"""
  _   _        _                      _____                     _    _
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
				 _|
"""


def calculate_rmsd(points1, points2, validPoints=None):
    """
	calculates the root mean square deviation between to point sets

	Parameters:
	-------
	points1, points2: numpy matrix (K, N)
	where K is the dimension of the points and N is the number of points

	validPoints: bool sequence of valid points in the point set.
	If it is left out, all points are considered valid
	"""
    assert (points1.shape == points2.shape)
    N = points1.shape[1]

    if validPoints == None:
        validPoints = [True] * N

    assert (len(validPoints) == N)

    points1 = points1[:, validPoints]
    points2 = points2[:, validPoints]

    N = points1.shape[1]

    dist = points1 - points2
    rmsd = 0
    for col in range(N):
        rmsd += np.matmul(dist[:, col].transpose(), dist[:, col]).flatten()[0]

    return np.sqrt(rmsd / N)


def get_chessboard_points_3D(chessboard_params):
    """
	Returns the 3d coordinates of the chessboard corners
	in the coordinate system of the chessboard itself.

	Returns
	-------
	objp : array
		(3, N) matrix with N being the number of corners
	"""
    assert (len(chessboard_params) == 3)
    width = chessboard_params[0]
    height = chessboard_params[1]
    square_size = chessboard_params[2]
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    return objp.transpose() * square_size


def cv_find_chessboard(depth_frame, infrared_frame, chessboard_params, device=""):
    """
	Searches the chessboard corners using the set infrared image and the
	checkerboard size

	Returns:
	-----------
	chessboard_found : bool
						  Indicates wheather the operation was successful
	corners          : array
						  (2,N) matrix with the image coordinates of the chessboard corners
	"""
    assert (len(chessboard_params) == 3)
    infrared_image = infrared_frame
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_found = False
    chessboard_found, corners = cv2.findChessboardCorners(infrared_image, (
        chessboard_params[1], chessboard_params[0]))

    _infrared_image = infrared_image.copy()
    _infrared_image = np.dstack((_infrared_image, _infrared_image, _infrared_image))
    if chessboard_found:
        corners = cv2.cornerSubPix(infrared_image, corners, (11, 11), (-1, -1), criteria)
        try:
            _patterns = (chessboard_params[1], chessboard_params[0])
            cv2.drawChessboardCorners(_infrared_image, _patterns, corners.reshape(-1, 2), chessboard_found)
            # cv2.imshow("calib" + device, _infrared_image)
            # cv2.waitKey(5)
            corners = np.transpose(corners, (2, 0, 1))

            _corners = corners.squeeze().transpose().astype(int)
            _infrared_image1 = infrared_image.copy()
            _infrared_image1 = np.dstack((_infrared_image1, _infrared_image1, _infrared_image1))
            for k, _corner in enumerate(_corners.tolist()):
                cv2.putText(_infrared_image1, str(k), tuple(_corner), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2,
                            cv2.LINE_AA)

            # cv2.imshow("calib_plotted" + device, _infrared_image1)
            # cv2.waitKey(5)

        except Exception as error:
            print("[ERROR] error in plotting the corners, ", error)
    return chessboard_found, corners, _infrared_image


def get_depth_at_pixel(depth_frame, pixel_x, pixel_y):
    """
	Get the depth value at the desired image point

	Parameters:
	-----------
	depth_frame 	 : rs.frame()
						   The depth frame containing the depth information of the image coordinate
	pixel_x 	  	 	 : double
						   The x value of the image coordinate
	pixel_y 	  	 	 : double
							The y value of the image coordinate

	Return:
	----------
	depth value at the desired pixel

	"""
    return depth_frame.as_depth_frame().get_distance(round(pixel_x), round(pixel_y))


def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
    """
	Convert the depth and image point information to metric coordinates

	Parameters:
	-----------
	depth 	 	 	 : double
						   The depth value of the image point
	pixel_x 	  	 	 : double
						   The x value of the image coordinate
	pixel_y 	  	 	 : double
							The y value of the image coordinate
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	X : double
		The x value in meters
	Y : double
		The y value in meters
	Z : double
		The z value in meters

	"""
    X = (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx * depth
    Y = (pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy * depth
    return X, Y, depth


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
    """
	Convert the depthmap to a 3D point cloud

	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters

	"""

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    z = depth_image.flatten() / 1000;
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return x, y, z


def convert_pointcloud_to_depth(pointcloud, camera_intrinsics):
    """
	Convert the world coordinate to a 2D image coordinate

	Parameters:
	-----------
	pointcloud 	 	 : numpy array with shape 3xN

	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	x : array
		The x coordinate in image
	y : array
		The y coordiante in image

	"""

    assert (pointcloud.shape[0] == 3)
    x_ = pointcloud[0, :]
    y_ = pointcloud[1, :]
    z_ = pointcloud[2, :]

    m = x_[np.nonzero(z_)] / z_[np.nonzero(z_)]
    n = y_[np.nonzero(z_)] / z_[np.nonzero(z_)]

    x = m * camera_intrinsics.fx + camera_intrinsics.ppx
    y = n * camera_intrinsics.fy + camera_intrinsics.ppy

    return x, y


def get_boundary_corners_2D(points):
    """
	Get the minimum and maximum point from the array of points

	Parameters:
	-----------
	points 	 	 : array
						   The array of points out of which the min and max X and Y points are needed

	Return:
	----------
	boundary : array
		The values arranged as [minX, maxX, minY, maxY]

	"""
    padding = 0.05
    if points.shape[0] == 3:
        assert (len(points.shape) == 2)
        minPt_3d_x = np.amin(points[0, :])
        maxPt_3d_x = np.amax(points[0, :])
        minPt_3d_y = np.amin(points[1, :])
        maxPt_3d_y = np.amax(points[1, :])

        boudary = [minPt_3d_x - padding, maxPt_3d_x + padding, minPt_3d_y - padding, maxPt_3d_y + padding]

    else:
        raise Exception("wrong dimension of points!")

    return boudary


def get_clipped_pointcloud(pointcloud, boundary):
    """
	Get the clipped pointcloud withing the X and Y bounds specified in the boundary

	Parameters:
	-----------
	pointcloud 	 	 : array
						   The input pointcloud which needs to be clipped
	boundary      : array
										The X and Y bounds

	Return:
	----------
	pointcloud : array
		The clipped pointcloud

	"""
    assert (pointcloud.shape[0] >= 2)
    pointcloud = pointcloud[:, np.logical_and(pointcloud[0, :] < boundary[1], pointcloud[0, :] > boundary[0])]
    pointcloud = pointcloud[:, np.logical_and(pointcloud[1, :] < boundary[3], pointcloud[1, :] > boundary[2])]
    return pointcloud


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
    assert (depth_frame.is_depth_frame())

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


def calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2d, depth_threshold=0.01):
    """
 Calculate the cumulative pointcloud from the multiple devices
	Parameters:
	-----------
	frames_devices : dict
		The frames from the different devices
		keys: str
			Serial number of the device
		values: [frame]
			frame: rs.frame()
				The frameset obtained over the active pipeline from the realsense device

	calibration_info_devices : dict
		keys: str
			Serial number of the device
		values: [transformation_devices, intrinsics_devices]
			transformation_devices: Transformation object
					The transformation object containing the transformation information between the device and the world coordinate systems
			intrinsics_devices: rs.intrinscs
					The intrinsics of the depth_frame of the realsense device

	roi_2d : array
		The region of interest given in the following order [minX, maxX, minY, maxY]

	depth_threshold : double
		The threshold for the depth value (meters) in world-coordinates beyond which the point cloud information will not be used.
		Following the right-hand coordinate system, if the object is placed on the chessboard plane, the height of the object will increase along the negative Z-axis

	Return:
	----------
	point_cloud_cumulative : array
		The cumulative pointcloud from the multiple devices
	"""
    # Use a threshold of 5 centimeters from the chessboard as the area where useful points are found
    point_cloud_cumulative = np.array([-1, -1, -1]).transpose()
    for (device, frame) in frames_devices.items():
        # Filter the depth_frame using the Temporal filter and get the corresponding pointcloud for each frame
        filtered_depth_frame = post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=0.1,
                                                        temporal_smooth_delta=80)
        point_cloud = convert_depth_frame_to_pointcloud(np.asarray(filtered_depth_frame.get_data()),
                                                        calibration_info_devices[device][1][rs.stream.depth])
        point_cloud = np.asanyarray(point_cloud)

        # Get the point cloud in the world-coordinates using the transformation
        point_cloud = calibration_info_devices[device][0].apply_transformation(point_cloud)

        # Filter the point cloud based on the depth of the object
        # The object placed has its height in the negative direction of z-axis due to the right-hand coordinate system
        point_cloud = get_clipped_pointcloud(point_cloud, roi_2d)
        point_cloud = point_cloud[:, point_cloud[2, :] < -depth_threshold]
        point_cloud_cumulative = np.column_stack((point_cloud_cumulative, point_cloud))
    point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)
    return point_cloud_cumulative


def calculate_boundingbox_points(point_cloud, calibration_info_devices, depth_threshold=0.01):
    """
	Calculate the top and bottom bounding box corner points for the point cloud in the image coordinates of the color imager of the realsense device

	Parameters:
	-----------
	point_cloud : ndarray
		The (3 x N) array containing the pointcloud information

	calibration_info_devices : dict
		keys: str
			Serial number of the device
		values: [transformation_devices, intrinsics_devices, extrinsics_devices]
			transformation_devices: Transformation object
					The transformation object containing the transformation information between the device and the world coordinate systems
			intrinsics_devices: rs.intrinscs
					The intrinsics of the depth_frame of the realsense device
			extrinsics_devices: rs.extrinsics
					The extrinsics between the depth imager 1 and the color imager of the realsense device

	depth_threshold : double
		The threshold for the depth value (meters) in world-coordinates beyond which the point cloud information will not be used
		Following the right-hand coordinate system, if the object is placed on the chessboard plane, the height of the object will increase along the negative Z-axis

	Return:
	----------
	bounding_box_points_color_image : dict
		The bounding box corner points in the image coordinate system for the color imager
		keys: str
				Serial number of the device
			values: [points]
				points: list
					The (8x2) list of the upper corner points stacked above the lower corner points

	length : double
		The length of the bounding box calculated in the world coordinates of the pointcloud

	width : double
		The width of the bounding box calculated in the world coordinates of the pointcloud

	height : double
		The height of the bounding box calculated in the world coordinates of the pointcloud
	"""
    # Calculate the dimensions of the filtered and summed up point cloud
    # Some dirty array manipulations are gonna follow
    if point_cloud.shape[1] > 500:
        # Get the bounding box in 2D using the X and Y coordinates
        coord = np.c_[point_cloud[0, :], point_cloud[1, :]].astype('float32')
        min_area_rectangle = cv2.minAreaRect(coord)
        bounding_box_world_2d = cv2.boxPoints(min_area_rectangle)
        # Caculate the height of the pointcloud
        height = max(point_cloud[2, :]) - min(point_cloud[2, :]) + depth_threshold

        # Get the upper and lower bounding box corner points in 3D
        height_array = np.array([[-height], [-height], [-height], [-height], [0], [0], [0], [0]])
        bounding_box_world_3d = np.column_stack(
            (np.row_stack((bounding_box_world_2d, bounding_box_world_2d)), height_array))

        # Get the bounding box points in the image coordinates
        bounding_box_points_color_image = {}
        for (device, calibration_info) in calibration_info_devices.items():
            # Transform the bounding box corner points to the device coordinates
            bounding_box_device_3d = calibration_info[0].inverse().apply_transformation(
                bounding_box_world_3d.transpose())

            # Obtain the image coordinates in the color imager using the bounding box 3D corner points in the device coordinates
            color_pixel = []
            bounding_box_device_3d = bounding_box_device_3d.transpose().tolist()
            for bounding_box_point in bounding_box_device_3d:
                bounding_box_color_image_point = rs.rs2_transform_point_to_point(calibration_info[2],
                                                                                 bounding_box_point)
                color_pixel.append(
                    rs.rs2_project_point_to_pixel(calibration_info[1][rs.stream.color], bounding_box_color_image_point))

            bounding_box_points_color_image[device] = np.row_stack(color_pixel)
        return bounding_box_points_color_image, min_area_rectangle[1][0], min_area_rectangle[1][1], height
    else:
        return {}, 0, 0, 0


def calculate_boundingbox_points_ir(point_cloud, calibration_info_devices, depth_threshold=0.01):
    """
	Calculate the top and bottom bounding box corner points for the point cloud in the image coordinates of the ir imager of the realsense device

	Parameters:
	-----------
	point_cloud : ndarray
		The (3 x N) array containing the pointcloud information

	calibration_info_devices : dict
		keys: str
			Serial number of the device
		values: [transformation_devices, intrinsics_devices, extrinsics_devices]
			transformation_devices: Transformation object
					The transformation object containing the transformation information between the device and the world coordinate systems
			intrinsics_devices: rs.intrinscs
					The intrinsics of the depth_frame of the realsense device
			extrinsics_devices: rs.extrinsics
					The extrinsics between the depth imager 1 and the color imager of the realsense device

	depth_threshold : double
		The threshold for the depth value (meters) in world-coordinates beyond which the point cloud information will not be used
		Following the right-hand coordinate system, if the object is placed on the chessboard plane, the height of the object will increase along the negative Z-axis

	Return:
	----------
	bounding_box_points_color_image : dict
		The bounding box corner points in the image coordinate system for the color imager
		keys: str
				Serial number of the device
			values: [points]
				points: list
					The (8x2) list of the upper corner points stacked above the lower corner points

	length : double
		The length of the bounding box calculated in the world coordinates of the pointcloud

	width : double
		The width of the bounding box calculated in the world coordinates of the pointcloud

	height : double
		The height of the bounding box calculated in the world coordinates of the pointcloud
	"""
    # Calculate the dimensions of the filtered and summed up point cloud
    # Some dirty array manipulations are gonna follow
    if point_cloud.shape[1] > 500:
        # Get the bounding box in 2D using the X and Y coordinates
        coord = np.c_[point_cloud[0, :], point_cloud[1, :]].astype('float32')
        min_area_rectangle = cv2.minAreaRect(coord)
        bounding_box_world_2d = cv2.boxPoints(min_area_rectangle)
        # Caculate the height of the pointcloud
        height = max(point_cloud[2, :]) - min(point_cloud[2, :]) + depth_threshold

        # Get the upper and lower bounding box corner points in 3D
        height_array = np.array([[-height], [-height], [-height], [-height], [0], [0], [0], [0]])
        bounding_box_world_3d = np.column_stack(
            (np.row_stack((bounding_box_world_2d, bounding_box_world_2d)), height_array))

        # Get the bounding box points in the image coordinates
        bounding_box_points_ir_image = {}
        for (device, calibration_info) in calibration_info_devices.items():
            # Transform the bounding box corner points to the device coordinates
            bounding_box_device_3d = calibration_info[0].inverse().apply_transformation(
                bounding_box_world_3d.transpose())

            # Obtain the image coordinates in the color imager using the bounding box 3D corner points in the device coordinates
            ir_pixel = []
            bounding_box_device_3d = bounding_box_device_3d.transpose().tolist()
            for bounding_box_point in bounding_box_device_3d:
                bounding_box_ir_image_point = rs.rs2_transform_point_to_point(calibration_info[2],
                                                                              bounding_box_point)
                ir_pixel.append(
                    rs.rs2_project_point_to_pixel(calibration_info[1][(rs.stream.infrared, 1)],
                                                  bounding_box_ir_image_point))

            bounding_box_points_ir_image[device] = np.row_stack(ir_pixel)
        return bounding_box_points_ir_image, min_area_rectangle[1][0], min_area_rectangle[1][1], height
    else:
        return {}, 0, 0, 0


def pixel_to_world_coordinate(pixel, calibration_info, depth_frame, rsense):
    """
    1) Use function pixel_to_camera_coordinate(pixel, calibration_info) to get camera coordinate
    2) Use extrinsics from calibration_info to get world_coordinate wrt that camera
    3) use pose transformation to get the world coordinate wrt to reference world coordinate
    :param pixel: should be a list of list
    :param calibration_info
    :param depth_frame
    :param rsense
    :return: world_coordinates
    """
    try:
        _module_name = inspect.currentframe().f_code.co_name
        # logger.info(f"Starting {_module_name} module")
        camera_coordinate = pixel_to_camera_coordinate(pixel, calibration_info, depth_frame, rsense)
        world_coorinate = []
        for _camera_coordinate in camera_coordinate:
            _camera_coordinate = np.array(_camera_coordinate).transpose().reshape(3, 1)
            _world_coordinate = calibration_info[0].apply_transformation(_camera_coordinate)
            world_coorinate.append(
                _world_coordinate.reshape(-1).tolist())  # world_coorinate.append(_world_coordinate.transpose())
        return world_coorinate
    except Exception as error:
        print("Error in module,", error)
    finally:
        print("Returning from pix to world module")


def pixel_to_camera_coordinate(pixel, camera_intrinsics, depth_frame, depth_scale):
    """
        1) Get depth value at given pixel
        2) Use intrinsics from calibration_info_devices and get the camera coordinates

            :param pixel:
            :param calibration_info:
            :param depth_frame:
            :param rsense:
            :return: camera_coordinate
            """
    try:
        _module_name = inspect.currentframe().f_code.co_name
        # logger.info(f"Starting {_module_name} module")
        camera_coordinate = []
        print("...", pixel.shape[0])
        for k in range(0, pixel.shape[0]):
            # print(_pixel, type(_pixel))
            _pixel = pixel[k]
            # _pixel = [pixel[1], pixel[0]]
            depth_value = depth_frame[_pixel[1], _pixel[0]]
            # depth_value = depth_frame[_pixel[0], _pixel[1]]

            # print(k, "depth_frame", depth_frame[_pixel[1], _pixel[0]] * depth_scale, depth_frame[_pixel[0], _pixel[1]] * depth_scale)
            _camera_coordinate = rs.rs2_deproject_pixel_to_point(camera_intrinsics[rs.stream.depth], _pixel.tolist(),
                                                                  depth_value * depth_scale)
            # print(_pixel, "reprojected..", rs.rs2_project_point_to_pixel(camera_intrinsics[rs.stream.depth], _camera_coordinate))
            camera_coordinate.append(_camera_coordinate)

        return camera_coordinate
    except Exception as error:
        print("[ERROR] error in pix to cam module", error)
    finally:
        print("Returning from pix to cam module")


def world_to_pixel_coordinate(ref_world_coordinate, calibration_info):
    """
    1) Use function world_to_camera_coordinate(world_coordinate, calibration_info)
    :param ref_world_coordinate: should be N,3 where N is the number of points
    :param calibration_info
    :return: pixel_coordinate
    """
    try:
        _module_name = inspect.currentframe().f_code.co_name
        # logger.info(f"Starting {_module_name} module")
        camera_coordinate = world_to_camera_coordinate(ref_world_coordinate, calibration_info)
        pixel_coordinate = []
        for _camera_coordinate in camera_coordinate:
            _pixel_coordinate = rs.rs2_project_point_to_pixel(calibration_info[1][(rs.stream.infrared, 1)],
                                                              [_camera_coordinate[0], _camera_coordinate[1],
                                                               _camera_coordinate[2]])
            _pixel_coordinate = rs.rs2_project_point_to_pixel(calibration_info[1][rs.stream.depth],
                                                              [_camera_coordinate[0], _camera_coordinate[1],
                                                               _camera_coordinate[2]])
            pixel_coordinate.append([round(_pixel_coordinate[1]), round(_pixel_coordinate[0])])
        return pixel_coordinate
    except Exception as error:
        print("Error in world to pix module ,", error)
    finally:
        print("Returning from world to pix module")


def world_to_camera_coordinate(ref_world_coordinate, calibration_info):
    """
    1) Use pose transformation to transfer from ref_world_coordinate to world coordinate wrt to camera
    2) Use externsics from calibration_info to transform from world coordinate to camera coordinate

    :param ref_world_coordinate: should be N,3 where N is the number of points
    :param calibration_info
    :return: camera_coordinate N,3
    """
    try:
        _module_name = inspect.currentframe().f_code.co_name
        # logger.info(f"Starting {_module_name} module")
        world_coordinate = calibration_info[0].inverse().apply_transformation(
            np.array(ref_world_coordinate.transpose()))
        world_coordinate = world_coordinate.transpose()
        world_coordinate = world_coordinate.tolist()
        camera_coordinate = []
        for _world_coordinate in world_coordinate:
            camera_coordinate.append(rs.rs2_transform_point_to_point(calibration_info[2], _world_coordinate))
        return camera_coordinate
    except Exception as error:
        print("[INFO]Error in world to cam module ,", error)
    finally:
        print("Returning from world to cam")


