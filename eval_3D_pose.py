# 
# read the predicted pose log 
data = open("predicted_pose.json", "rb")
predicted_pose_log = json.load(data)
print(predicted_pose_log)
# a dict of dicts of "header" and "poses"
# Each poses is an array of "pose" with an "id" 