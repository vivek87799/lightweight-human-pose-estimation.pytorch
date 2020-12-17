import json
import time


class SkeletonPoseToJson:
    def __init__(self, device="triangulate"):
        self.header = Header(device)
        # self.header = json.dumps(Header().__dict__, separators=(',', ':'))
        # self.id = id
        self.poses = []

    def add_pose(self, id, pose_3d):
        pose = []
        for j in range(0, len(pose_3d)):
            pose.append(MarkerJson(j, pose_3d[j]))
        self.poses.append(Pose(id, pose))

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

class SkeletonIDTrackerToJson:
    def __init__(self, device="triangulate"):
        self.header = Header(device)
        self.skeletonIDPosition = []
    
    def add_skeleton_position(self, id, predicted_position, detected_position, ground_truth=None):
        self.skeletonIDPosition.append(IDPositionsDetectedPredicted(id, predicted_position, detected_position))
    
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

class IDPositionsDetectedPredicted():
    def __init__(self, id, predicted_position, detected_position):
        self.id = id
        print(predicted_position, detected_position)
        self.predicted_position = Position(float(predicted_position[0]), float(predicted_position[1]), float(predicted_position[2]))
        self.detected_position = Position(float(detected_position[0]), float(detected_position[1]), float(detected_position[2]))

class Header():
    def __init__(self, device="triangulate"):
        self.stamp = TimeStamp()
        self.frame_id = device
        self.seq = 0

class TimeStamp():
    def __init__(self):
        self.secs = time.time()
        self.nsecs = float("%.20f"% time.time()) # time.time_ns() # self.secs, self.nsecs = divmod(time.time(), 1)

class Pose:
    def __init__(self, id, pose):
        self.id = id
        self.pose = pose

class Position():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MarkerJson():
    def __init__(self, id, marker_point):
        # self.header = Header()
        self.id = id
        self.position = Position(float(marker_point[0]), float(marker_point[1]), float(marker_point[2]))