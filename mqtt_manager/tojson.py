import json
import time


class ToJson:
    def __init__(self, id=1, pose_3d=None):
        self.header = Header()
        # self.header = json.dumps(Header().__dict__, separators=(',', ':'))
        # self.id = id
        self.poses = []
        for j in range(0, len(pose_3d)):
            self.poses.append(MarkerJson(id, pose_3d[j]))

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class Header(ToJson):
    def __init__(self):
        self.stamp = TimeStamp()
        self.frame_id = "realsense"
        self.seq = 0


class TimeStamp(ToJson):
    def __init__(self):
        self.secs, self.nsecs = divmod(time.time(), 1)


class Pose:
    def __init__(self, marker_point):
        self.position = Position(float(marker_point[0]), float(marker_point[1]), float(marker_point[2]))
        self.orientation = Orientation()


class Position(Pose):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Orientation(Pose):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0


class MarkerJson(ToJson):
    def __init__(self, id, marker_point):
        # self.header = Header()
        self.id = id
        self.position = Position(float(marker_point[0]), float(marker_point[1]), float(marker_point[2]))
        self.orientation = Orientation()