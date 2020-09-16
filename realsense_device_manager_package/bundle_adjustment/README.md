1) Initialize PyBundleAdjustment with 
            
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

2) Call to the function bundleAdjust returns optimized camera params and 3d_Points


## For multi camera pose estimation
We take camera coordinates along with the 3d coordinates and optimize the rotation and translation vectors

1) Initialize PyBundleAdjustment_cam_coord
2) Call to function bundleAdjust returns optimised rotation and translation vectors

https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
https://github.com/jahdiel/pySBA.git
