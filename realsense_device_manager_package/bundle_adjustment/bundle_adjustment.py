"""
MIT License (MIT)
Copyright (c) FALL 2016, Jahdiel Alvarez
Author:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


class PyBundleAdjustment:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices):
        """Intializes all the class attributes and instance variables.
            Write the specifications for each variable:

            cameraArray with shape (n_cameras, 9) contains initial estimates of parameters for all cameras.
                    First 3 components in each row form a rotation vector,
                    next 3 components form a translation vector,
                    then a focal distance and two distortion parameters.

            points_3d with shape (n_points, 3)
                    contains initial estimates of point coordinates in the world frame.

            camera_ind with shape (n_observations,)
                    contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

            point_ind with shape (n_observations,)
                    contatins indices of points (from 0 to n_points - 1) involved in each observation.

            points_2d with shape (n_observations, 2)
                    contains measured 2-D coordinates of points projected on images in each observations.
        """
        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def project(self, points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        f = cameraArray[:, 6]
        k1 = cameraArray[:, 7]
        k2 = cameraArray[:, 8]
        n = np.sum(points_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points_proj *= (r * f)[:, np.newaxis]
        return points_proj

    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """

        try:
            camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
            points_3d = params[n_cameras * 6:].reshape((n_points, 3))
            points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
            return (points_proj - points_2d).ravel()
        except Exception as error:
            print("Error caught in fun", error)

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        n = numCameras * 9 + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(9):
            A[2 * i, cameraIndices * 9 + s] = 1
            A[2 * i + 1, cameraIndices * 9 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 9 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 9 + pointIndices * 3 + s] = 1

        return A

    def optimizedParams(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))

        return camera_params, points_3d

    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        try:
            numCameras = self.cameraArray.shape[0]
            numPoints = self.points3D.shape[0]

            x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
            print("before params -->", self.cameraArray)
            f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D)
            A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)
            res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                                args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))

            params = self.optimizedParams(res.x, numCameras, numPoints)
            print("after params -->", params[0])

            return params
        except Exception as error:
            print("ERROR in bundle adjustment with intrinsics", error)


class PyBundleAdjustment_cam_coord:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, camera_params, points3D, points2D, camera_indices, points_indices, optical_center, points3D_cam, depth_values=None):
        """Intializes all the class attributes and instance variables.
            Write the specifications for each variable:
            camera_params with shape (n_cameras, 9) contains initial estimates of parameters for all cameras.
                    First 3 components in each row form a rotation vector,
                    next 3 components form a translation vector,
                    then a focal distance and two distortion parameters.
            points_3d with shape (n_points, 3)
                    contains initial estimates of point coordinates in the world frame.
            camera_ind with shape (n_observations,)
                    contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
            point_ind with shape (n_observations,)
                    contatins indices of points (from 0 to n_points - 1) involved in each observation.
            points_2d with shape (n_observations, 2)
                    contains measured 2-D coordinates of points projected on images in each observations.
        """
        self.camera_params = camera_params
        self.points_ref_3D = points3D
        self.optical_center = optical_center
        self.points3D = points3D
        self.points2D = points2D

        self.camera_indices = camera_indices
        self.points_indices = points_indices

        self.points3D_cam = points3D_cam
        self.depth_values = depth_values
        self.points_proj = []

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def get_pose_mat(self, rotation_matrix, translation_vector):
        """

        :param rotation_matrix:
        :param translation_vector:
        :return:
        """
        pose_mat = np.zeros((rotation_matrix.shape[0], 4, 4))
        pose_mat[:, :3, :3] = rotation_matrix
        # translation_vector = translation_vector[:, np.newaxis]
        pose_mat[:, :3, 3] = translation_vector
        pose_mat[:, 3, 3] = 1
        return pose_mat

    def inverse_rotation_trans(self, pose_mat):
        """
        Computes the inverse transformation and returns a new Transformation object

        Returns:
        -----------
        inverse: Transformation

        """
        rotation_matrix = pose_mat[:, :3, :3]
        translation_vector = pose_mat[:, :3, 3]

        rot = rotation_matrix.transpose(0, 2, 1)
        trans = - np.matmul(rotation_matrix.transpose(0, 2, 1), translation_vector[:, :, np.newaxis])
        return self.get_pose_mat(rot, trans.squeeze())

    def project_native(self, points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        # points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        return points_proj

    def project(self, points, camera_params, optical_center):
        """
        :param points: N x D
        :param camera_params:  N x no_params
        :param optical_center: N x 2
        :return:
        """
        """Convert 3-D points to 2-D by projecting onto images."""
        r_vec = camera_params[:, :3]
        translation_vector = camera_params[:, 3:6]
        r_vec = r_vec[:, np.newaxis, :]
        rotation_matrix = []
        for i in range(r_vec.shape[0]):
            rmat = cv2.Rodrigues(r_vec[i])
            rotation_matrix.append(rmat[0])
        rotation_matrix = np.array(rotation_matrix)

        pose_mat = self.get_pose_mat(rotation_matrix, translation_vector)
        pose_mat_inv = self.inverse_rotation_trans(pose_mat)

        ###########
        # 1) Extrinsics transformation
        assert (points.shape[1] == 3)
        n = points.shape[0]
        points_ = np.vstack((np.transpose(points), np.ones((1, n))))
        points_trans_ = np.matmul(pose_mat, points_.transpose()[:, :, np.newaxis])
        points_proj = np.true_divide(points_trans_[:, :3, :], points_trans_[:, [-1], :]).squeeze()

        """
        points_proj = rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]

        # points_cam_coord, points_3d_eval = de_project(points_2d, camera_params, optical_center)

        points_proj_eval_man = points_proj
        points_proj = -points_proj[:, :2] / points_proj[:, 2,
                                            np.newaxis]  # -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        """
        points_proj_man = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        points_transformed = points_proj[:, :2] / points_proj[:, 2, np.newaxis]


        # 2) Applying intrinsics
        f = camera_params[:, 6]
        k1 = camera_params[:, 7]
        k2 = camera_params[:, 8]

        n = np.sum(points_proj ** 2, axis=1)
        # r = 1 + k1 * n + k2 * n ** 2
        points_proj_man = points_proj_man * ((1 + k1 * n + k2 * n ** 2) * f)[:, np.newaxis]

        points_proj_man[:, 0] = points_proj_man[:, 0] + optical_center[:, 0]
        points_proj_man[:, 1] = points_proj_man[:, 1] + optical_center[:, 1]

        # from observations the y was - and swaped with x.. look into theory
        # points_proj_man = points_proj_man[:, [1, 0]]

        return points_proj_man

    def de_project(self, points, camera_params, optical_center):
        """

        :param points: Nx2
        :param camera_params: N x no_params
        :param optical_center: Nx2
        :param pose_mat: Nx3x3
        :return:
        """
        """Convert 2-D points to 3-D """
        r_vec = camera_params[:, 0:3]
        translation_vector = camera_params[:, 3:6]
        r_vec = r_vec[:, np.newaxis, :]
        # cv_rodrigues = np.vectorize(cv2.Rodrigues)
        # rotation_matrix = cv_rodrigues(r_vec[::1, :, :])[0]
        rotation_matrix = []
        for i in range(r_vec.shape[0]):
            rmat = cv2.Rodrigues(r_vec[i])
            rotation_matrix.append(rmat[0])
        rotation_matrix = np.array(rotation_matrix)

        pose_mat = self.get_pose_mat(rotation_matrix, translation_vector)

        f = camera_params[:, 6]
        k1 = camera_params[:, 7]
        k2 = camera_params[:, 8]
        points_reshaped = points.reshape(self.camera_params.shape[0], int(len(points) / self.camera_params.shape[0]), 2).astype(int)
        depth_frames = []
        depth_values = self.depth_values
        # 1) Swap axis
        # points = points[:, [1, 0]]

        # 2) optical center and focal w and h
        points[:, 0] = points[:, 0] - optical_center[:, 0]
        points[:, 1] = points[:, 1] - optical_center[:, 1]
        points = points / f[:, np.newaxis]

        # 3) cam dist
        n = np.sum(points ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        points = points / (1 + k1 * n + k2 * n ** 2)[:, np.newaxis]

        # 4) depth factor
        points[:, :] = points[:, :] * depth_values[:, np.newaxis]
        # points[:, , np.newaxis] = depth_values[:, np.newaxis]
        points = np.insert(points, [2], depth_values[:, np.newaxis], axis=1)

        # 5) Apply extrinsics
        pose_mat_inv = self.inverse_rotation_trans(pose_mat)
        assert (points.shape[1] == 3)
        n = points.shape[0]
        points_ = np.vstack((np.transpose(points), np.ones((1, n))))
        points_trans_ = np.matmul(pose_mat_inv, points_.transpose()[:, :, np.newaxis])
        points_transformed = np.true_divide(points_trans_[:, :3, :], points_trans_[:, [-1], :]).squeeze()

        return points, points_transformed

    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points3D_cam):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))

        """focal (h,w) and the two distortions are constant and calibrated"""
        # camera_params[:, 6:9] = self.camera_params[:, 6:9]
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        # the 3d observed reference points are fixed
        # points_proj = self.project(self.points3D[point_indices], camera_params[camera_indices], self.optical_center[camera_indices])
        # points_proj = self.project(points_3d[point_indices], camera_params[camera_indices], self.optical_center[camera_indices])
        points_proj = self.project_native(points_3d[point_indices], camera_params[camera_indices])

        # Use this function to visualize the adjustment
        # visualize_bundle(points_proj)
        error = (points_proj - points3D_cam).ravel()
        return np.nan_to_num(error)

    def bundle_adjustment_sparsity(self, numCameras, numPoints, camera_indices, pointIndices):
        m = camera_indices.size * 3  # m = f0.size
        n = numCameras * 6 + numPoints * 3  # n = x0.size
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 6 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 6 + pointIndices * 3 + s] = 1

        return A

    def optimizedParams(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        return camera_params, points_3d

    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        numCameras = self.camera_params.shape[0]
        numPoints = self.points3D.shape[0]
        print("[INFO] camera params before-->", self.camera_params)

        x0 = np.hstack((self.camera_params.ravel(), self.points3D.ravel()))
        f0 = self.fun(x0, numCameras, numPoints, self.camera_indices, self.points_indices, self.points3D_cam)

        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.camera_indices, self.points_indices)

        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(numCameras, numPoints, self.camera_indices, self.points_indices, self.points3D_cam))

        params = self.optimizedParams(res.x, numCameras, numPoints)

        # Only the camera params
        camera_params = params[0]
        print("[INFO] camera params after-->", camera_params)
        # points_proj = self.project_native(self.points3D[self.points_indices], camera_params[self.camera_indices],
        #                           self.optical_center[self.camera_indices])
        return params, []