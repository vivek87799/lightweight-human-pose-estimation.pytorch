import numpy as np

from scipy.optimize import linear_sum_assignment

from tracking.kalman_filter1 import KalmanFilter
from tracking.tracker1 import Tracker

from tracking.tracker1 import Track as TrackParent

class Track(TrackParent):
    """Track class for every object to be tracked
    """

    def __init__(self, prediction=None, joints=None, trackIdCount=None, measurement_noise=0.1):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        TrackParent.__init__(self, prediction, trackIdCount)
        self.joints = joints
        self.joints_tracker = None
        self.updateJoints(1.5, 20, 5, 10) # Should be a list of tracks
    
    def updateJoints(self, dist_thresh=None, max_frames_to_skip=None, max_trace_length=None, trackIdCount=None):
        if not self.joints_tracker:
            self.joints_tracker = JointsTracker(dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount)
        detections = []
        for detection in self.joints.transpose().tolist():
            # Check if the detection has -1 which indicates that the joint is missing so we 
            # should not update the detection
            if (-1 in detection):
                continue
            # print("joint detections to be updated", np.asarray(detection).reshape(3, 1))
            detections.append(np.asarray(detection).reshape(3, 1))
        self.joints_tracker.Update(detections)
        

class Skeleton(object):
    def __init__(self, joints=None, mask=None):
        """
        joints: 3xN numpy array
        """
        self.joints = joints 
        self.joints_centre = None
    
    def add_joints(self, joints, mask=None):
        self.joints = joints/10
        rot_optical_coord_to_ros_coord = np.array(([[0.0000000, 0.0000000, 1.0000000],
                                                [-1.0000000, 0.0000000, 0.0000000],
                                                [0.0000000, -1.0000000, 0.0000000]]))
        self.joints = np.matmul(rot_optical_coord_to_ros_coord, self.joints)
        self.joints[mask] = np.nan
        
        # Replace nan with 0 in both the joints and joints_centre
        self.joints_centre = np.nan_to_num(np.nanmean(self.joints, axis=1).reshape(3,1))
        self.joints = np.nan_to_num(self.joints)
        
        self.joints[mask] = -1

class JointsTracker(Tracker):
        """
        Tracker class that updates track vectors of object tracked
        """
        def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                    trackIdCount, measurement_noise=0.005):
            """Initialize variable used by Tracker class
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
            Tracker.__init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount, measurement_noise)

        def get_joints(self):
            joints = []
            for track in self.tracks3d:
                joints.extend(track.prediction.transpose())
                print("added==>")
                print("min hits ==>", track.min_hits)
                print("skipped frames ==>", track.skipped_frames)
                print("max age ==>", track.max_age)
                # print(track.prediction.transpose().shape, track.prediction.transpose())
            # print("track from joints", joints)
            
            return np.asarray(joints)

class SkeletonsTracker(Tracker):
    """Tracker class that updates track vectors of object tracked

    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount, measurement_noise=0.10):
        """Initialize variable used by Tracker class
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
        Tracker.__init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount, measurement_noise)

    def Update(self, skeletons, min_hit_criteria=10):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            Skeleton: Create a Skeleton object with joints from SkeletonsTracker.py
        Return:
            None
        """

        detections = []
        for skeleton in skeletons:
            detections.append(skeleton.joints_centre)
        # Create tracks if no tracks vector found
        if len(self.tracks3d) == 0:
            for i in range(len(detections)):
                track = Track(prediction=skeletons[i].joints_centre, joints=skeletons[i].joints, trackIdCount=self.trackIdCount, measurement_noise=self.measurement_noise)
                
                self.trackIdCount += 1
                self.tracks3d.append(track)
                

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks3d)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks3d)):
            for j in range(len(detections)):
                try:
                    diff = self.tracks3d[i].prediction - detections[j]
                    distance = np.sqrt(diff[0][0] * diff[0][0] +
                                       diff[1][0] * diff[1][0] +
                                       diff[2][0] * diff[2][0])
                    cost[i][j] = distance
                except Exception as e:
                    print("[ERROR] error in tracker1.py while calculation the dist", e)

        # Let's average the squared ERROR
        cost = 0.5 * cost
        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            ## To keep the measure value
            # TODO
            ## Read the KF params and please remove the below line as it is already stored some where
            # self.tracks3d[i].detection = np.asarray(detections[assignment[i]])
            self.tracks3d[i].detection = np.asarray(skeletons[assignment[i]].joints_centre)
            # self.tracks3d[i].joints = np.asarray(skeletons[assignment[i]].joints)
            self.tracks3d[i].joints = np.asarray(skeletons[assignment[i]].joints)
            # Update the joints_tracker
            self.tracks3d[i].updateJoints()
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                self.tracks3d[i].min_hits = self.tracks3d[i].min_hits + 1

                if self.tracks3d[i].min_hits > min_hit_criteria:  # int(len())
                    self.tracks3d[i].max_age = 1
            else:
                self.tracks3d[i].skipped_frames += 1
                if self.tracks3d[i].skipped_frames > int(self.tracks3d[i].skipped_frames):
                    self.tracks3d[i].min_hits = 0
                # if self.tracks3d[i].skipped_frames > int(self.tracks3d[i].skipped_frames/2):
                    # self.tracks3d[i].min_hits = 0

        # If tracks are not detected for long time, remove them
        del_tracks = []
        for i in range(len(self.tracks3d)):
            if self.tracks3d[i].skipped_frames > self.max_frames_to_skip:
                del_tracks.append(i)
        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks3d):
                    del self.tracks3d[id]
                    del assignment[id]

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(prediction=skeletons[un_assigned_detects[i]].joints_centre, joints=skeletons[un_assigned_detects[i]].joints, trackIdCount=self.trackIdCount, measurement_noise=self.measurement_noise)
                self.trackIdCount += 1
                self.tracks3d.append(track)

        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            self.tracks3d[i].KF.predict()

            if assignment[i] != -1:
                self.tracks3d[i].skipped_frames = 0
                # TODO_ with the prediction update the 3d points
                self.tracks3d[i].KF.correct(detections[assignment[i]], 1)
                
                _temp = self.tracks3d[i].KF.kf.x[[0, 2, 4]]
                self.tracks3d[i].prediction = self.tracks3d[i].KF.kf.x[[0, 2, 4]]
                """
                _temp = self.tracks3d[i].KF.kf.x[[0, 1, 2]]
                self.tracks3d[i].prediction = self.tracks3d[i].KF.kf.x[[0, 1, 2]]
                """
            else:
                # TODO_ with the prediction update the 3d points
                self.tracks3d[i].KF.correct(self.tracks3d[i].KF.kf.x, 0)
                self.tracks3d[i].prediction = self.tracks3d[i].KF.kf.x[[0, 2, 4]]

            if len(self.tracks3d[i].trace) > self.max_trace_length:
                for j in range(len(self.tracks3d[i].trace) -
                               self.max_trace_length):
                    del self.tracks3d[i].trace[j]

            self.tracks3d[i].trace.append(self.tracks3d[i].prediction)
            self.tracks3d[i].KF.lastResult = self.tracks3d[i].prediction

    