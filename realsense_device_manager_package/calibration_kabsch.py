##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.		                      ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files 					  ####
##################################################################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import realsense_device_manager_package.calculate_rmsd_kabsch as rmsd
from realsense_device_manager_package.helper_functions import cv_find_chessboard, get_chessboard_points_3D, \
    get_depth_at_pixel, convert_depth_pixel_to_metric_coordinate, post_process_depth_frame

"""
  _   _        _                      _____                     _    _                    
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___ 
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
				 _|                                                                      
"""


def calculate_transformation_kabsch(src_points, dst_points):
    """
    Calculates the optimal rigid transformation from src_points to
    dst_points
    (regarding the least squares error)

    Parameters:
    -----------
    src_points: array
        (3,N) matrix
    dst_points: array
        (3,N) matrix

    Returns:
    -----------
    rotation_matrix: array
        (3,3) matrix

    translation_vector: array
        (3,1) matrix

    rmsd_value: float

    """
    assert src_points.shape == dst_points.shape
    if src_points.shape[0] != 3:
        raise Exception("The input data matrix had to be transposed in order to compute transformation.")

    src_points = src_points.transpose()
    dst_points = dst_points.transpose()

    src_points_centered = src_points - rmsd.centroid(src_points)
    dst_points_centered = dst_points - rmsd.centroid(dst_points)

    rotation_matrix = rmsd.kabsch(src_points_centered, dst_points_centered)
    rmsd_value = rmsd.kabsch_rmsd(src_points_centered, dst_points_centered)

    translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

    return rotation_matrix.transpose(), translation_vector.transpose(), rmsd_value


def calculate_transformation_quaternion(src_points, dst_points):
    """
    Calculates the optimal rigid transformation from src_points to
    dst_points
    (regarding the least squares error)

    Parameters:
    -----------
    src_points: array
        (3,N) matrix
    dst_points: array
        (3,N) matrix

    Returns:
    -----------
    rotation_matrix: array
        (3,3) matrix

    translation_vector: array
        (3,1) matrix

    rmsd_value: float

    """
    assert src_points.shape == dst_points.shape
    if src_points.shape[0] != 3:
        raise Exception("The input data matrix had to be transposed in order to compute transformation.")

    src_points = src_points.transpose()
    dst_points = dst_points.transpose()

    src_points_centered = src_points - rmsd.centroid(src_points)
    dst_points_centered = dst_points - rmsd.centroid(dst_points)

    rotation_matrix = rmsd.quaternion_rotate(src_points_centered, dst_points_centered)
    translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

    rmsd_value = rmsd.quaternion_rmsd(src_points_centered, dst_points_centered)

    translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

    return rotation_matrix.transpose(), translation_vector.transpose(), rmsd_value


"""
  __  __         _           ____               _                _   
 |  \/  |  __ _ (_) _ __    / ___| ___   _ __  | |_  ___  _ __  | |_ 
 | |\/| | / _` || || '_ \  | |    / _ \ | '_ \ | __|/ _ \| '_ \ | __|
 | |  | || (_| || || | | | | |___| (_) || | | || |_|  __/| | | || |_ 
 |_|  |_| \__,_||_||_| |_|  \____|\___/ |_| |_| \__|\___||_| |_| \__|																	 

"""


class Transformation:
    def __init__(self, rotation_matrix, translation_vector):
        self.pose_mat = np.zeros((4, 4))
        self.pose_mat[:3, :3] = rotation_matrix
        self.pose_mat[:3, 3] = translation_vector.flatten()
        self.pose_mat[3, 3] = 1

    def apply_transformation(self, points):
        """
        Applies the transformation to the pointcloud

        Parameters:
        -----------
        points : array
            (3, N) matrix where N is the number of points

        Returns:
        ----------
        points_transformed : array
            (3, N) transformed matrix
        """
        assert (points.shape[0] == 3)
        n = points.shape[1]
        points_ = np.vstack((points, np.ones((1, n))))
        points_trans_ = np.matmul(self.pose_mat, points_)
        test = self.pose_mat[:3, 3]
        test1 = self.pose_mat[:, 3]
        # points_trans_[:3] = points_trans_[:3] - self.pose_mat[:, 3][:, np.newaxis]
        points_transformed = np.true_divide(points_trans_[:3, :], points_trans_[[-1], :])
        return points_transformed

    def inverse(self):
        """
        Computes the inverse transformation and returns a new Transformation object

        Returns:
        -----------
        inverse: Transformation

        """
        rotation_matrix = self.pose_mat[:3, :3]
        translation_vector = self.pose_mat[:3, 3]

        rot = np.transpose(rotation_matrix)
        trans = - np.matmul(np.transpose(rotation_matrix), translation_vector)
        return Transformation(rot, trans)


class PoseEstimation:

    def __init__(self, frames, intrinsic, chessboard_params):
        assert (len(chessboard_params) == 3)
        self.frames = frames
        self.intrinsic = intrinsic
        self.chessboard_params = chessboard_params

    def get_chessboard_corners_in3d(self):
        """
        Searches the chessboard corners in the infrared images of
        every connected device and uses the information in the
        corresponding depth image to calculate the 3d
        coordinates of the chessboard corners in the coordinate system of
        the camera

        Returns:
        -----------
        corners3D : dict
            keys: str
                Serial number of the device
            values: [success, points3D, validDepths]
                success: bool
                    Indicates wether the operation was successfull
                points3d: array
                    (3,N) matrix with the coordinates of the chessboard corners
                    in the coordinate system of the camera. N is the number of corners
                    in the chessboard. May contain points with invalid depth values
                validDephts: [bool]*
                    Sequence with length N indicating which point in points3D has a valid depth value
        """
        corners3D = {}
        chessboard_images = []
        for (serial, frameset) in self.frames.items():
            # depth_frame = post_process_depth_frame(frameset[rs.stream.depth])
            depth_frame = frameset[rs.stream.depth]
            infrared_frame = frameset[(rs.stream.infrared, 1)]
            depth_intrinsics = self.intrinsic[serial][rs.stream.depth]
            found_corners, points2D, chessboard_image = cv_find_chessboard(depth_frame, infrared_frame, self.chessboard_params, serial)
            corners3D[serial] = [found_corners, None, None, None]
            if found_corners:
                points3D = np.zeros((3, len(points2D[0])))
                validPoints = [False] * len(points2D[0])
                for index in range(len(points2D[0])):
                    corner = points2D[:, index].flatten()
                    depth_value = depth_frame[int(round(corner[1])), int(round(corner[0]))]
                    # depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
                    if depth_value != 0 and depth_value is not None:
                        validPoints[index] = True
                        # depth_frame_np = np.asanyarray(depth_frame.get_data())
                        # depth_value_ = depth_frame_np[int(round(corner[0])), int(round(corner[1]))]

                        [X1, Y1, Z1] = rs.rs2_deproject_pixel_to_point(depth_intrinsics,
                                                                       [int(round(corner[0])), int(round(corner[1]))],
                                                                       depth_value * 0.0010000000474974513)
                        _pixel_coordinate = rs.rs2_project_point_to_pixel(depth_intrinsics, [X1, Y1, Z1])
                        [X, Y, Z] = convert_depth_pixel_to_metric_coordinate(depth_value * 0.0010000000474974513,
                                                                             corner[0], corner[1], depth_intrinsics)
                        _pixel_coordinate_ = rs.rs2_project_point_to_pixel(depth_intrinsics, [X, Y, Z])
                        # Project pixel to point
                        # [X, Y, Z] = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [corner[0], corner[1]], depth * 0.0010000000474974513)

                        points3D[0, index] = X
                        points3D[1, index] = Y
                        points3D[2, index] = Z
                corners3D[serial] = found_corners, points2D, points3D, validPoints
            chessboard_images.append(chessboard_image)
        return corners3D, chessboard_images

    def perform_pose_estimation(self, master_serial="None"):
        """
        Calculates the extrinsic calibration from the coordinate space of the camera to the
        coordinate space spanned by a chessboard by retrieving the 3d coordinates of the
        chessboard with the depth information and subsequently using the kabsch algortihm
        for finding the optimal rigid transformation between the two coordinate spaces

        Returns:
        -----------
        retval : dict
        keys: str
            Serial number of the device
        values: [success, transformation, points2D, rmsd]
            success: bool
            transformation: Transformation
                Rigid transformation from the coordinate system of the camera to
                the coordinate system of the chessboard
            points2D: array
                [2,N] array of the chessboard corners used for pose_estimation
            rmsd:
                Root mean square deviation between the observed chessboard corners and
                the corners in the local coordinate system after transformation
        """
        corners3D, chessboard_images = self.get_chessboard_corners_in3d()
        retval = {}
        ref_set = False
        for (serial, [found_corners, points2D, points3D, validPoints]) in corners3D.items():
            objectpoints = get_chessboard_points_3D(self.chessboard_params)
            retval[serial] = [False, None, None, None]
            points2D_indices = np.where(validPoints)[0]
            if found_corners:
                # initial vectors are just for correct dimension
                valid_object_points = objectpoints[:, validPoints]
                valid_observed_object_points = points3D[:, validPoints]

                # check for sufficient points
                if valid_object_points.shape[1] < 48:
                    print("[INFO] detected points with valid depth.. ", valid_object_points.shape[1])
                    print("Not enough points have a valid depth for calculating the transformation")

                else:

                    if not ref_set:
                        ref_set = True
                        ref_plane_2D = points2D[:, validPoints]
                        valid_object_points2D = ref_plane_2D
                        valid_observed_object_points2D = ref_plane_2D
                        ref_plane_points = points3D[:, validPoints]
                        valid_object_points = points3D[:, validPoints]
                        valid_observed_object_points = points3D[:, validPoints]
                    else:
                        valid_object_points2D = ref_plane_2D[:, validPoints]
                        valid_observed_object_points2D = points2D[:, validPoints]
                        valid_object_points = ref_plane_points[:, validPoints]
                        valid_observed_object_points = points3D[:, validPoints]

                    # [rotation_matrix, translation_vector, rmsd_value] =
                    # calculate_transformation_kabsch(valid_object_points, valid_observed_object_points)
                    """
                    [rotation_matrix, translation_vector, rmsd_value] = calculate_transformation_kabsch(
                        valid_object_points, valid_observed_object_points)
                        
                    [rotation_matrix_, translation_vector_, rmsd_value] = rigid_transform_3D(
                        np.asmatrix(valid_object_points.transpose()),
                        np.asmatrix(valid_observed_object_points.transpose()))
                        """

                    # TODO
                    # Add param to check if ros coordinates is required
                    # Rotate the points to ros coordinates
                    ###############################################################################################
                    rot_optical_coord_to_ros_coord = np.array(([[0.0000000, 0.0000000, 1.0000000],
                                                      [-1.0000000, 0.0000000, 0.0000000],
                                                      [0.0000000, -1.0000000, 0.0000000]]))

                    [rotation_matrix, translation_vector, rmsd_value] = calculate_transformation_quaternion(
                        valid_object_points, valid_observed_object_points)

                    # valid_object_points = np.matmul(rot_optical_coord_to_ros_coord, valid_object_points)
                    # valid_observed_object_points = np.matmul(rot_optical_coord_to_ros_coord, valid_observed_object_points)

                    ###############################################################################################

                    [rotation_matrix_ros, translation_vector_ros, rmsd_value] = calculate_transformation_quaternion(
                        valid_object_points,
                        valid_observed_object_points)

                    # transformation = Transformation(rotation_matrix_ros, translation_vector_ros)
                    # eval_3d = transformation.apply_transformation(valid_object_points)
                    # eval_3d_ref = transformation.inverse().apply_transformation(valid_observed_object_points)

                    rvecs = cv2.Rodrigues(rotation_matrix_ros)
                    param_ba = np.hstack((np.array(rvecs[0]).flatten(), translation_vector_ros,
                                          self.intrinsic[serial][(rs.stream.infrared, 1)].fx, 0, 0,
                                          self.intrinsic[serial][(rs.stream.infrared, 1)].ppx,
                                          self.intrinsic[serial][(rs.stream.infrared, 1)].ppy))
                    # param_ba = np.hstack((rotation_matrix.flatten(), translation_vector))
                    if rmsd_value > 0.5:  # 0.03:  # 0.015:  # 0.0045:  # 0.004:
                        retval[serial] = [False, Transformation(rotation_matrix_ros, translation_vector_ros),
                                          (points2D.squeeze()[:, validPoints][:, :, np.newaxis]),
                                          rmsd_value, points3D[:, validPoints], valid_object_points, param_ba,
                                          points2D_indices, valid_observed_object_points]
                        print("RMS error for calibration with device number", serial, "is :", rmsd_value, "m")
                        print("To be re-calibrated")

                    else:
                        print("RMS error for calibration with device number", serial, "is :", rmsd_value, "m")
                        retval[serial] = [True, Transformation(rotation_matrix_ros, translation_vector_ros),
                                          (points2D.squeeze()[:, validPoints][:, :, np.newaxis]),
                                          rmsd_value, points3D[:, validPoints], valid_object_points, param_ba,
                                          points2D_indices, valid_observed_object_points]

        return retval, chessboard_images
