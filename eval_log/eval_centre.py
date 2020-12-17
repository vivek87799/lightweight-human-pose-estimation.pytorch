import json
import numpy as np

import scipy.spatial.distance as distance

def eval(GT_centre, measured_centre, predicted_centre):
    measured_centre_error = []
    predicted_centre_error = []
    for i in range(1988, 2090):
        # print(distance.cdist(np.asarray(measured_3d_pose[str(i)]), np.asarray(measured_3d_pose[str(i)]), "euclidean"))


        #print(np.asarray(GT_centre[str(i)])==-1)
        GT_centre_np = np.asarray(GT_centre[str(i)])# .reshape(3,1)
        measured_centre_np = np.asarray(measured_centre[str(i)])# .reshape(3,1)
        predicted_centre_np = np.asarray(predicted_centre[str(i)])#.reshape(3,1)
        # print(measured_centre_np)

        
        # dist_gt_measured = distance.cdist(GT_centre_np, measured_centre_np, "euclidean")
        # measured_centre_error.append(dist_gt_measured.diagonal())
        # print(len(measured_centre_error))
        # print("measured diff ", measured_centre_np, GT_centre_np)
        # print("predicted diff ", predicted_centre_np, GT_centre_np)
        measured_centre_error.append(np.linalg.norm(GT_centre_np-measured_centre_np))
        predicted_centre_error.append(np.linalg.norm(GT_centre_np-predicted_centre_np))
        
        # print(np.linalg.norm(GT_centre_np-measured_centre_np))
        print(np.linalg.norm(GT_centre_np-predicted_centre_np))
        ##print("mean measured==>", np.mean(measured_centre_error, axis=0))
        ##print("mean measured==>", np.mean(predicted_centre_error, axis=0))
    
    print("mean error of ", np.mean(np.asarray(measured_centre_error).reshape(1, -1)))
    print("mean error of ", np.mean(np.asarray(predicted_centre_error).reshape(1, -1)))
if __name__ == "__main__":
        
    with open("GT_centre.json") as GT_log:
        GT_centre = json.load(GT_log)

    with open("measured_centre.json") as measured_log:
        measured_centre = json.load(measured_log)
    
    with open("predicted_centre.json") as predicted_centre:
        predicted_centre = json.load(predicted_centre)


    eval(GT_centre, measured_centre, predicted_centre)