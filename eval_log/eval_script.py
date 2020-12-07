import json
import numpy as np

import scipy.spatial.distance as distance

def eval(GT_3d_pose, measured_3d_pose):
    joint_error = []
    for i in range(1988, 2090):
        # print(distance.cdist(np.asarray(measured_3d_pose[str(i)]), np.asarray(measured_3d_pose[str(i)]), "euclidean"))
        print(np.asarray(measured_3d_pose[str(i)])==-1)
        measured_3d_pose_np = np.asarray(measured_3d_pose[str(i)])
        GT_3d_pose_np = np.asarray(GT_3d_pose[str(i)])
        mask_measured_undetected_joints = (measured_3d_pose_np == -1)
        GT_3d_pose_np[mask_measured_undetected_joints] = -1
        dist = distance.cdist(GT_3d_pose_np, measured_3d_pose_np, "euclidean")
        joint_error.append(dist.diagonal())
        print(len(joint_error))
        print(np.asarray(joint_error).shape)
        print(np.mean(joint_error, axis=0))

if __name__ == "__main__":
    
    with open("GT_pose_log.json") as GT_log:
        GT_3d_pose = json.load(GT_log)

    with open("measured_pose_log.json") as measured_log:
        measured_3d_pose = json.load(measured_log)

    eval(GT_3d_pose, measured_3d_pose)