from tracking.tracker1 import Tracker

class SkeletonIDTracker:
    def __init__(self, skeleton_ID=1, dist_thresh=1, max_frames_to_skip=20, max_trace_length=5):
        """Initialize SkeletonIDTracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_length: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.skeletonID = skeleton_ID
        self.skeleton_mean_point = None
        self.skeleton_joints = []
        self.skeleton_joints_tracker_3d = Tracker(2, 50, 30, 100)

    def set_skeleton_ID(self, ID):
        self.skeleton_ID = ID

    def get_skeleton_ID(self):
        return self.skeleton_ID

    def set_skeleton_mean(self, mean):
        self.skeleton_mean_point = mean
    
    def get_skeleton_mean(self):
        return self.skeleton_mean_point


    def update_joints(self, joints):
        """
        Args: 
            joints: should be a list of 3x1 numpy arrays
        """
        self.skeleton_joints_tracker_3d.update(joints)
    
    def get_joints(self):
        joints = []
        for track in self.skeleton_joints_tracker_3d.tracks3d:
            joints.append(track.prediction)
        return joints

    
