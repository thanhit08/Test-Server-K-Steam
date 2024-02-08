import numpy as np
import matplotlib.pyplot as plt
import json
angleJointNames = ["Right Shoulder", "Right Elbow", "Left Shoulder", "Left Elbow",
                   "Right Leg", "Right Knee", "Left Leg", "Left Knee",
                   "Right Shoulder & Elbow", "Left Shoulder & Elbow"]
std_bad_angle_data = json.load(open('std_bad_angle.json'))
std_good_angle_data = json.load(open('std_good_angle.json'))
teacher_pose_data = json.load(open('teacher_angle.json'))
std_bad_angle_dict = {}
std_good_angle_dict = {}
teacher_angle_dict = {}
# Sample joint angle time series (replace this with your actual data)
for index, angleJointName in enumerate(angleJointNames):
    for std_bad_pose, std_good_pose, teacher_pose in zip(std_bad_angle_data, std_good_angle_data, teacher_pose_data):
        if len(std_bad_pose) != 10 or len(std_good_pose) != 10 or len(teacher_pose) != 10:
            continue
        if angleJointName not in std_bad_angle_dict:
            std_bad_angle_dict[angleJointName] = []
        std_bad_angle_dict[angleJointName].append(std_bad_pose[index])
        
        if angleJointName not in std_good_angle_dict:
            std_good_angle_dict[angleJointName] = []
        std_good_angle_dict[angleJointName].append(std_good_pose[index])

        if angleJointName not in teacher_angle_dict:
            teacher_angle_dict[angleJointName] = []
        teacher_angle_dict[angleJointName].append(teacher_pose[index])

std_bad_right_shoulder = std_bad_angle_dict["Right Elbow"]
std_good_right_shoulder = std_good_angle_dict["Right Elbow"]
teacher_right_shoulder = teacher_angle_dict["Right Elbow"]

# convert to np array
std_bad_right_shoulder = np.array(std_bad_right_shoulder)
std_good_right_shoulder = np.array(std_good_right_shoulder)
teacher_right_shoulder = np.array(teacher_right_shoulder)

time = np.linspace(0, 1200, num=1200)  # Assuming 100 frames in 20 seconds
# Replace this with your joint angle data
std_bad_joint_angles = std_bad_right_shoulder
std_good_joint_angles = std_good_right_shoulder
teacher_joint_angles = teacher_right_shoulder

# Calculate speed (first derivative)
std_bad_speed = np.gradient(std_bad_joint_angles, time)
std_good_speed = np.gradient(std_good_joint_angles, time)
teacher_speed = np.gradient(teacher_joint_angles, time)

# Calculate velocity (second derivative)
std_bad_acceleration = np.gradient(std_bad_speed, time)
std_good_acceleration = np.gradient(std_good_speed, time)
teacher_acceleration = np.gradient(teacher_speed, time)

std_bad_velocity = np.gradient(std_bad_acceleration, time)
std_good_velocity = np.gradient(std_good_acceleration, time)
teacher_velocity = np.gradient(teacher_acceleration, time)

# Plot the results
plt.figure(figsize=(18, 9))

plt.subplot(3, 1, 1)
plt.plot(time, std_bad_joint_angles, label='Bad Joint Angles')
plt.plot(time, std_good_joint_angles, label='Good Joint Angles')
plt.plot(time, teacher_joint_angles, label='Teacher Joint Angles')
plt.title('Joint Angles over Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle')
plt.legend()


plt.subplot(3, 1, 2)
plt.plot(time, std_bad_speed, label='Bad Speed')
plt.plot(time, std_good_speed, label='Good Speed')
plt.plot(time, teacher_speed, label='Teacher Speed')
plt.title('Joint Speed over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed')
plt.legend()


plt.subplot(3, 1, 3)
plt.plot(time, std_bad_velocity, label='Bad Velocity')
plt.plot(time, std_good_velocity, label='Good Velocity')
plt.plot(time, teacher_velocity, label='Teacher Velocity')
plt.title('Joint Velocity over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()


