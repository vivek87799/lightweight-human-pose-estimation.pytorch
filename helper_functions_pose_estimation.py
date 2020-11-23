from os import listdir
from os.path import isfile, join

import sys
import json
import time 

import concurrent.futures

import cv2
import numpy as np

######################################For Transformation######################################
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
#########################################################################################################################

################################Get the projection matrix for the cmu dataset############################################
def read_json_data(filename):
    with open(filename) as json_file:
        json_data = json.load(json_file)
    return json_data

def get_camera_matrix(calibration_data, type='hd', name='00_17'):
    """
    calibration_data: calibration json file data
    type: type of sensor hd of vga
    name: seq of the camera
    """
    for calib in calibration_data:
        # print("calib..", calib)
        if calib['name']==name and calib['type']==type:
            calibration_data = calib
            print("calibration required", calib)
    
    # TODO index [0] to be replaced by camera ind
    # TODO Iterate over available camera calib and find the desired one
    K = np.array(calibration_data['K'])
    R = np.array(calibration_data['R'])
    t = np.array(calibration_data['t'])
    extrinsics = np.c_[R, t]
    camera_matrix = np.matmul(K, extrinsics)
    proj_matrix_transformation = Transformation(camera_matrix[:3, :3], camera_matrix[:3, 3])

    return proj_matrix_transformation.pose_mat[:3, :]

def processing_loop(calib_file_name="cmu_haggle", type="", name=""):
    name=name
    calibration_data = read_json_data(calib_file_name)

    return get_camera_matrix(calibration_data['cameras'], name=name)
########################################################################################################################################

#################################################Get the 3D pose using triangulation####################################################
def depth_from_triangulation(camera_pose, qf_keypoints, gf_keypoints):
    """
    Takes a list of camera_pose and corresponding key_points and returns the 3D coordinates
    camera_pose - n x 3 x 4
    points_2d - 2 x n
    camera_indices
    returns points_3d 3 x n
    """
    try:
        # TODO modifiy to handle more than two cameras
        points_3d = []
        print(qf_keypoints.shape, gf_keypoints.shape)
        for i in range(0, len(camera_pose)-1):
            print(np.array(camera_pose[0]))
            print(camera_pose[0].shape)
            x = cv2.triangulatePoints(np.array(camera_pose[0]), np.array(camera_pose[i + 1]), qf_keypoints.transpose(),
                                      gf_keypoints.transpose())
            x /= x[3]
            x = x[:3]
            points_3d.extend(x)
        print("points 3d -->", np.array(points_3d).shape)
        return points_3d
    except Exception as error:
        print("ERROR in depth_from_triangulation module ", error)
########################################################################################################################################
if __name__ == "__main__":
    processing_loop("../cmu_haggle/calibration_170228_haggling_b1.json")