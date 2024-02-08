#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 6.2
#  in conjunction with Tcl version 8.6
#    Nov 22, 2021 05:51:45 PM JST  platform: Windows NT

import multiprocessing
import sys
import os
from datetime import datetime
from threading import currentThread
import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import threading
import shutil
import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
from matplotlib import pyplot as plt


def vp_start_gui():
    root = tk.Tk()
    window = MyWindow(root)
    init(root, window)
    root.mainloop()


class MyWindow:
    def __init__(self, my_root=None):
        my_root.geometry("600x215+650+150")
        my_root.minsize(600, 215)
        my_root.maxsize(1200, 430)
        my_root.resizable(0,  0)
        my_root.title("Test K-STEAM")
        my_root.configure(background="wheat")
        my_root.configure(highlightbackground="wheat")
        my_root.configure(highlightcolor="black")

        self.Label1 = tk.Label(my_root)
        self.Label1.place(x=20, y=20, height=25, width=48)
        self.Label1.configure(background="wheat")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {DejaVu Sans Mono} -size 14")
        self.Label1.configure(foreground="#000000")
        self.Label1.configure(text='''Name''')

        self.Text1 = tk.Text(my_root)
        self.Text1.place(x=80, y=20, height=30, width=204)
        self.Text1.configure(background="white")
        self.Text1.configure(font="-family {DejaVu Sans} -size 14")
        self.Text1.configure(foreground="black")
        self.Text1.configure(highlightbackground="wheat")
        self.Text1.configure(highlightcolor="black")
        self.Text1.configure(insertbackground="black")
        self.Text1.configure(selectbackground="blue")
        self.Text1.configure(selectforeground="white")
        self.Text1.configure(wrap="word")
        # Set the initial text of the textbox
        self.Text1.insert(tk.END, "Welcome to K-STEAM!\n")
        # Set the text to be uneditable
        self.Text1.configure(state=tk.DISABLED)
        # Set the padding of the text
        self.Text1.configure(padx=2, pady=4)

        self.Button2 = tk.Button(my_root)
        self.Button2.place(x=300, y=20, height=30, width=207)
        self.Button2.configure(activebackground="#f4bcb2")
        self.Button2.configure(activeforeground="#000000")
        self.Button2.configure(background="wheat")
        self.Button2.configure(command=test_pose_estimation_real_time)
        self.Button2.configure(disabledforeground="#a3a3a3")
        self.Button2.configure(font="-family {DejaVu Sans Mono} -size 14")
        self.Button2.configure(foreground="#000000")
        self.Button2.configure(highlightbackground="wheat")
        self.Button2.configure(highlightcolor="black")
        self.Button2.configure(pady="0")
        self.Button2.configure(text='''FPS Test''')

        self.Button3 = tk.Button(my_root)
        self.Button3.place(x=80, y=60, height=30, width=427)
        self.Button3.configure(activebackground="#f4bcb2")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="wheat")
        self.Button3.configure(command=test_pose_estimation)
        self.Button3.configure(disabledforeground="#a3a3a3")
        self.Button3.configure(font="-family {DejaVu Sans Mono} -size 14")
        self.Button3.configure(foreground="#000000")
        self.Button3.configure(highlightbackground="wheat")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(pady="0")
        self.Button3.configure(text='''Accuracy of Motion Recognition Test''')

        self.Button3 = tk.Button(my_root)
        self.Button3.place(x=80, y=100, height=30, width=427)
        self.Button3.configure(activebackground="#f4bcb2")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="wheat")
        self.Button3.configure(command=test_speed_pose_estimation)
        self.Button3.configure(disabledforeground="#a3a3a3")
        self.Button3.configure(font="-family {DejaVu Sans Mono} -size 14")
        self.Button3.configure(foreground="#000000")
        self.Button3.configure(highlightbackground="wheat")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(pady="0")
        self.Button3.configure(text='''Speed of Pose Estimation Test''')

        self.Button3 = tk.Button(my_root)
        self.Button3.place(x=80, y=140, height=30, width=427)
        self.Button3.configure(activebackground="#f4bcb2")
        self.Button3.configure(activeforeground="#000000")
        self.Button3.configure(background="wheat")
        self.Button3.configure(command=test_distance)
        self.Button3.configure(disabledforeground="#a3a3a3")
        self.Button3.configure(font="-family {DejaVu Sans Mono} -size 14")
        self.Button3.configure(foreground="#000000")
        self.Button3.configure(highlightbackground="wheat")
        self.Button3.configure(highlightcolor="black")
        self.Button3.configure(pady="0")
        self.Button3.configure(text='''Distance Test''')

        self.Button1 = tk.Button(my_root)
        self.Button1.place(x=223, y=180, height=30, width=120)
        self.Button1.configure(activebackground="#f4bcb2")
        self.Button1.configure(activeforeground="black")
        self.Button1.configure(background="wheat")
        self.Button1.configure(command=quit)
        self.Button1.configure(compound='top')
        self.Button1.configure(disabledforeground="#b8a786")
        self.Button1.configure(font="-family {DejaVu Sans} -size 14")
        self.Button1.configure(foreground="#000000")
        self.Button1.configure(highlightbackground="wheat")
        self.Button1.configure(highlightcolor="black")
        self.Button1.configure(pady="0")
        self.Button1.configure(text='''Quit''')


def init(root, window, *args, **kwargs):
    global my_window, my_root
    my_root = root
    my_window = window

    global my_mp_drawing
    my_mp_drawing = mp.solutions.drawing_utils
    global my_mp_drawing_styles
    my_mp_drawing_styles = mp.solutions.drawing_styles
    global my_mp_pose
    my_mp_pose = mp.solutions.pose
    global my_pose
    my_pose = my_mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1)

    global test_folder, IMAGE_FILES
    test_folder = 'Test_Data/'
    IMAGE_FILES = os.listdir(test_folder)

    global ref_image_folder, REF_IMAGES, REF_POSES, REF_LABELS
    ref_image_folder = 'Test_Data/Accuracy_Test/ref/'
    REF_IMAGES = os.listdir(ref_image_folder)
    REF_POSES = []
    REF_LABELS = []
    for ref_image_name in REF_IMAGES:
        label_name = ref_image_name[ref_image_name.index(
            '_') + 1:ref_image_name.index('.')]
        REF_LABELS.append(label_name)

    global test_image_folder, TEST_IMAGES, result_image_folder, BASE_DIR
    test_image_folder = 'Test_Data/Accuracy_Test/test'
    TEST_IMAGES = os.listdir(test_image_folder)
    result_image_folder = 'Test_Data/Accuracy_Test/Result/'
    if not os.path.exists(result_image_folder):
        os.makedirs(result_image_folder)
    else:
        shutil.rmtree(result_image_folder)
        os.makedirs(result_image_folder)

    BASE_DIR = os.getcwd()
    load_ref_pose()

# Function to calculate FPS


def calculate_fps(prev_time, current_time):
    fps = 1 / (current_time - prev_time)
    return fps


def test_pose_estimation_real_time_thread():
    global my_pose, my_mp_drawing, my_mp_pose
    change_text("Init Camera...\n")
    cap = cv2.VideoCapture(0)
    pTime = 0.0
    paTime = 0.0
    # Variables for FPS calculation
    prev_time = 0
    current_time = 0

    text_changed = False
    while cap.isOpened():
        prev_time = time.time()
        success, img = cap.read()
        if not text_changed:
            change_text("Testing FPS...\n")
            text_changed = True

        if not success:
            print("Ignoring empty camera frame.")
            break

        bgr_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = my_pose.process(bgr_image)

        # Calculate and display FPS
        current_time = time.time()
        fps = calculate_fps(prev_time, current_time)
        prev_time = current_time

        cTime = time.time()

        if results.pose_landmarks:
            my_mp_drawing.draw_landmarks(
                img, results.pose_landmarks, my_mp_pose.POSE_CONNECTIONS)

        dif_Time = (cTime - pTime) * 1000

        if paTime == 0:
            paTime = dif_Time

        cur_a_time = (paTime + dif_Time) / 2

        paTime = cur_a_time

        # fps = 1 / (cTime - pTime)

        pTime = cTime

        cv2.putText(img, str(int(fps)) + '   ' + str(int(cur_a_time)),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("FPS Testing...", img)
        if cv2.waitKey(5) & 0xFF == 27:
            change_text("Finished FPS Testing\n")
            break


def change_text(text):
    global my_window
    textbox = my_window.Text1
    # Set the text to editable
    textbox.configure(state=tk.NORMAL)
    # Clear the text
    textbox.delete('1.0', tk.END)
    # Set the text to "Testing FPS..."
    textbox.insert(tk.END, text)
    # Set the text to uneditable
    textbox.configure(state=tk.DISABLED)


def test_pose_estimation_real_time():
    change_text("Wait FPS Testing...\n")

    # start a thread to run the function
    thread = threading.Thread(target=test_pose_estimation_real_time_thread)
    thread.start()


angleJoints = [[11, 12, 14], [12, 14, 16],  # Right Shoulder, Right Elbow
               [12, 11, 13], [11, 13, 15],  # Left Shoulder, Left Elbow
               [23, 24, 26], [24, 26, 28],  # Right Leg, Right Knee
               [24, 23, 25], [23, 25, 27]]  # Left Leg, Left Knee


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def process_2D_pose(jointDatas):
    processed_angle_data = []
    temp_angle = []
    counter = 0
    for angleJoint in angleJoints:
        angle_value = calculate_angle([jointDatas[angleJoint[0]][0],
                                       jointDatas[angleJoint[0]][1]],
                                      [jointDatas[angleJoint[1]][0],
                                       jointDatas[angleJoint[1]][1]],
                                      [jointDatas[angleJoint[2]][0],
                                       jointDatas[angleJoint[2]][1]])
        processed_angle_data.append(angle_value)
    #     temp_angle.append(angle_value)
    # left_shoulder_elbow_angle = temp_angle[0] + temp_angle[1]
    # right_shoulder_elbow_angle = temp_angle[2] + temp_angle[3]

    # processed_angle_data.append(left_shoulder_elbow_angle)
    # processed_angle_data.append(right_shoulder_elbow_angle)

    return processed_angle_data


def process_2D_OP_pose(op_keypoints):
    processed_angle_data = []

    for angleJoint in OP_AngleJoints:
        angle_value = calculate_angle([op_keypoints[angleJoint[0]][0], op_keypoints[angleJoint[0]][1], op_keypoints[angleJoint[0]][2]],
                                      [op_keypoints[angleJoint[1]][0], op_keypoints[angleJoint[1]]
                                       [1], op_keypoints[angleJoint[1]][2]],
                                      [op_keypoints[angleJoint[2]][0], op_keypoints[angleJoint[2]][1], op_keypoints[angleJoint[2]][2]])
        processed_angle_data.append(angle_value)

    return processed_angle_data


def mp_to_op(mp_keypoints):
    op_keypoints = []
    op_keypoints.append(mp_keypoints[0])
    op_keypoints.append([(mp_keypoints[11][0] + mp_keypoints[12][0]) / 2.0, (mp_keypoints[11]
                        [1] + mp_keypoints[12][1])/2.0, (mp_keypoints[11][2] + mp_keypoints[12][2])/2.0])
    op_keypoints.append(mp_keypoints[12])
    op_keypoints.append(mp_keypoints[14])
    op_keypoints.append(mp_keypoints[16])
    op_keypoints.append(mp_keypoints[11])
    op_keypoints.append(mp_keypoints[13])
    op_keypoints.append(mp_keypoints[15])
    op_keypoints.append([(mp_keypoints[23][0] + mp_keypoints[24][0]) / 2.0, (mp_keypoints[23]
                        [1] + mp_keypoints[24][1])/2.0, (mp_keypoints[23][2] + mp_keypoints[24][2])/2.0])
    op_keypoints.append(mp_keypoints[24])
    op_keypoints.append(mp_keypoints[26])
    op_keypoints.append(mp_keypoints[28])
    op_keypoints.append(mp_keypoints[23])
    op_keypoints.append(mp_keypoints[25])
    op_keypoints.append(mp_keypoints[27])
    op_keypoints.append(mp_keypoints[5])
    op_keypoints.append(mp_keypoints[2])
    op_keypoints.append(mp_keypoints[8])
    op_keypoints.append(mp_keypoints[7])
    op_keypoints.append(mp_keypoints[31])
    op_keypoints.append(mp_keypoints[31])
    op_keypoints.append(mp_keypoints[29])
    op_keypoints.append(mp_keypoints[32])
    op_keypoints.append(mp_keypoints[32])
    op_keypoints.append(mp_keypoints[30])
    return op_keypoints


OP_AngleJoints = [  # [0, 12, 11, 12],  # Neck
    [1, 2, 3], [2, 3, 4],  # Right Shoulder, Right Elbow
    [1, 5, 6], [5, 6, 7],  # Left Shoulder, Left Elbow
    # [12, 24, 26],  # Hip
    [8, 9, 10], [9, 10, 11],  # Right Leg, Right Knee
    [8, 12, 13], [13, 14, 15]]  # Left Leg, Left Knee


def calculate_op_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def load_ref_pose():
    global ref_image_folder, REF_IMAGES, REF_POSES
    global my_pose
    for img_file_name in REF_IMAGES:
        image_url = os.path.join(ref_image_folder, img_file_name)
        image = cv2.imread(image_url)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = my_pose.process(imageRGB)

        pose_2D_landmarks = results.pose_landmarks
        landmark2D = []
        key_points = []
        height, width, _ = image.shape
        if results.pose_landmarks:
            for landmark in pose_2D_landmarks.landmark:
                landmark2D.append([landmark.x * width, landmark.y * height])
                key_point = [landmark.x * width,
                             landmark.y * height, landmark.z * width]
                key_points.append(key_point)
        op_keypoints = mp_to_op(key_points)
        op_keypoints = nom_op_to_Hips(op_keypoints)
        # processed_angle_data = process_2D_OP_pose(op_keypoints)
        processed_angle_data = process_2D_pose(landmark2D)
        REF_POSES.append(processed_angle_data)


def nom_op_to_Hips(op_keypoints):
    nom_op_keypoints = []
    hips_index_in_op_pose = 8
    hips_keypoint = op_keypoints[hips_index_in_op_pose]
    for i in range(len(op_keypoints)):
        op_keypoint = op_keypoints[i]
        if i == hips_index_in_op_pose:
            nom_op_keypoints.append([0, 0, op_keypoint[2]])
            # nom_op_keypoints.append(op_keypoint)
        else:
            nom_op_keypoint = [op_keypoint[0] - hips_keypoint[0],
                               op_keypoint[1] - hips_keypoint[1], op_keypoint[2]]
            nom_op_keypoints.append(nom_op_keypoint)
    return nom_op_keypoints


labeled_value = [
    "Pose20",
    "Pose18",
    "Pose17",
    "Pose01",
    "Pose20",
    "Pose19",
    "Pose18",
    "Pose17",
    "Pose05",
    "Pose19",
    "Pose17",
    "Pose16",
    "Pose13",
    "Pose19",
    "Pose18",
    "Pose16",
    "Pose20",
    "Pose19",
    "Pose18",
    "Pose17",
    "Pose01",
    "Pose16",
    "Pose19",
    "Pose18",
    "Pose17",
    "Pose05",
    "Pose20",
    "Pose19",
    "Pose18",
    "Pose17",
    "Pose16",
    "Pose19",
    "Pose18",
    "Pose16",
    "Pose20",
    "Pose15",
    "Pose14",
    "Pose11",
    "Pose10",
    "Pose09",
    "Pose08",
    "Pose07",
    "Pose13",
    "Pose12",
    "Pose04",
    "Pose06",
    "Pose03",
    "Pose02",
    "Pose01",
    "Pose15",
    "Pose15",
    "Pose09",
    "Pose08",
    "Pose07",
    "Pose14",
    "Pose06",
    "Pose12",
    "Pose05",
    "Pose11",
    "Pose10",
    "Pose06",
    "Pose03",
    "Pose02",
    "Pose01",
    "Pose15",
    "Pose09",
    "Pose08",
    "Pose07",
    "Pose14",
    "Pose13",
    "Pose06",
    "Pose12",
    "Pose05",
    "Pose11",
    "Pose04",
    "Pose10",
    "Pose06",
    "Pose03",
    "Pose02",
    "Pose01",
    "Pose15",
    "Pose09",
    "Pose08",
    "Pose07",
    "Pose14",
    "Pose13",
    "Pose12",
    "Pose05",
    "Pose11",
    "Pose04",
    "Pose10",
    "Pose03",
    "Pose02",
    "Pose15",
    "Pose09",
    "Pose08",
    "Pose07",
    "Pose12",
    "Pose04",
    "Pose10"
]


def draw_keypoints(frame, keypoints, confidence, circle_size):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence:
            cv2.circle(frame, (int(kx), int(ky)), circle_size, (0, 255, 0), -1)


def pose_extraction_move_net(interpreter, test_input_folder, test_output_folder, img_file_name, circle_size):
    image_url = os.path.join(test_input_folder, img_file_name)
    image = cv2.imread(image_url)
    img = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    start_time = time.time() * 1000
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time() * 1000
    process_time = end_time - start_time
    draw_keypoints(image, keypoints_with_scores, 0.3, circle_size)
    cv2.imwrite(os.path.join(test_output_folder,
                f'result_{img_file_name}'), image)
    return process_time, keypoints_with_scores


def test_distance():
    interpreter = tf.lite.Interpreter(model_path='3.tflite')
    interpreter.allocate_tensors()
    test_root_folder = "Test_Data"
    test_speed_folder = "Distance_Test"
    test_input_folder = "Input"
    test_output_folder = "Output"

    # Path to test input folder
    test_input_folder_path = os.path.join(
        test_root_folder, test_speed_folder, test_input_folder)
    # Path to test output folder
    test_output_folder_path = os.path.join(
        test_root_folder, test_speed_folder, test_output_folder)

    # Check if test output folder exists
    if os.path.exists(test_output_folder_path):
        # Delete test output folder
        shutil.rmtree(test_output_folder_path)
        # Create test output folder
        os.makedirs(test_output_folder_path)
    else:
        # Create test output folder
        os.makedirs(test_output_folder_path)

    # Find all image files in test input folder
    image_files = os.listdir(test_input_folder_path)
    # Find all image with .jpg extension
    image_files = [file_name for file_name in image_files if file_name.lower().endswith(
        '.jpg') or file_name.endswith('.png')]

    with open(os.path.join(test_output_folder_path, 'Poses.csv'), 'a') as f:
        for image_file_name in image_files:
            process_time, keypoints_with_scores = pose_extraction_move_net(
                interpreter, test_input_folder_path, test_output_folder_path, image_file_name, 30)
            keypoints_with_scores_string = ' '.join(
                str(keypoints_with_scores).split())
            f.write(
                f'{image_file_name},{round(process_time, 2)}, {keypoints_with_scores_string}\n')
    change_text("Finished Testing\n")


def test_speed_pose_estimation():
    interpreter = tf.lite.Interpreter(model_path='3.tflite')
    interpreter.allocate_tensors()
    test_root_folder = "Test_Data"
    test_speed_folder = "Speed_Test"
    test_input_folder = "Input"
    test_output_folder = "Output"

    # Path to test input folder
    test_input_folder_path = os.path.join(
        test_root_folder, test_speed_folder, test_input_folder)
    # Path to test output folder
    test_output_folder_path = os.path.join(
        test_root_folder, test_speed_folder, test_output_folder)

    # Check if test output folder exists
    if os.path.exists(test_output_folder_path):
        # Delete test output folder
        shutil.rmtree(test_output_folder_path)
        # Create test output folder
        os.makedirs(test_output_folder_path)
    else:
        # Create test output folder
        os.makedirs(test_output_folder_path)

    # Find all image files in test input folder
    image_files = os.listdir(test_input_folder_path)
    # Find all image with .jpg extension
    image_files = [file_name for file_name in image_files if file_name.lower().endswith(
        '.jpg') or file_name.endswith('.png')]

    # Extract pose from each image file using pose_extraction_move_net function
    # Save processing_time to csv file
    total_time = 0
    with open(os.path.join(test_output_folder_path, 'processing_time.csv'), 'a') as f:
        for image_file_name in image_files:
            process_time, keypoints_with_scores = pose_extraction_move_net(
                interpreter, test_input_folder_path, test_output_folder_path, image_file_name, 20)
            total_time += process_time
            keypoints_with_scores_string = ' '.join(
                str(keypoints_with_scores).split())
            f.write(
                f'{image_file_name},{round(process_time, 5)}, {keypoints_with_scores_string}\n')
            # f.write(f'{image_file_name},{process_time}\n')

    # Calculate average processing time
    average_time = total_time / len(image_files)
    output_text = f"Avg. time: {round(average_time, 2)} ms"
    change_text(output_text)


def test_pose_estimation():
    # total_time = 0
    predict_labels = []
    global test_folder, IMAGE_FILES, result_image_folder, REF_LABELS, REF_POSES, TEST_IMAGES, test_image_folder, BASE_DIR
    global my_pose
    # list_mins = []
    list_labels = []
    # list_subs = []
    for img_file_name in TEST_IMAGES:
        # Path to test image
        image_url = os.path.join(test_image_folder, img_file_name)

        # Read image
        image = cv2.imread(image_url)
        # Convert image from BGR to RGB
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # start_time = time.time() * 1000
        # Process image using pose estimation model
        results = my_pose.process(imageRGB)
        # end_time = time.time() * 1000

        # process_time = end_time - start_time
        # total_time += process_time
        # Get 2D pose landmarks from results
        pose_2D_landmarks = results.pose_landmarks
        landmark2D = []
        # key_points = []
        height, width, _ = image.shape
        # Convert 2D pose landmarks to 2D keypoints
        if results.pose_landmarks:
            for landmark in pose_2D_landmarks.landmark:
                landmark2D.append([landmark.x * width, landmark.y * height])
                # key_point = [landmark.x * width,
                #              landmark.y * height, landmark.z * width]
                # key_points.append(key_point)

        # op_keypoints = mp_to_op(key_points)
        # op_keypoints = nom_op_to_Hips(op_keypoints)
        # processed_angle_data = process_2D_OP_pose(op_keypoints)
        processed_angle_data = process_2D_pose(landmark2D)
        # upper_body_angle = processed_angle_data[:4]
        # lower_body_angle = processed_angle_data[4:]

        list_distance = []
        for ref_pose_data in REF_POSES:
            # Check distance between two poses using Euclidean distance
            # upper_ref_pose_data = ref_pose_data[:4]
            # lower_ref_pose_data = ref_pose_data[4:]
            # upper_dist = np.linalg.norm(
            #     np.array(upper_body_angle)-np.array(upper_ref_pose_data))
            # lower_dist = np.linalg.norm(
            #     np.array(lower_body_angle)-np.array(lower_ref_pose_data))
            # dist = upper_dist + lower_dist
            dist_overall = np.linalg.norm(np.array(processed_angle_data) -
                                          np.array(ref_pose_data))

            list_distance.append(round(dist_overall, 2))

        list_distance_backup = list_distance.copy()
        # Get 2 smallest number in list list_distances
        # list_min = []
        list_label = []
        # list_sub = []

        # Get the smallest number in list list_distances
        first_min = min(list_distance)
        # Get the label of the smallest number in list list_distances
        label1_dance = REF_LABELS[list_distance.index(first_min)]
        # Add the smallest number and its label to list list_mins and list_labels
        # list_min.append(first_min)
        # list_sub.append(0)
        list_label.append(label1_dance)
        list_distance.remove(first_min)

        # Get the other smaller 16 than first_min number in list list_distances
        while (True):
            min_value = min(list_distance)
            if abs(min_value-first_min) > 16:
                break
            # list_sub.append(abs(round((min_value-first_min)/8, 2)))
            # list_min.append(min_value)
            list_label.append(
                REF_LABELS[list_distance_backup.index(min_value)])
            list_distance.remove(min_value)

        dance_label_folder = os.path.join(
            result_image_folder, 'Ref_' + label1_dance)
        if not os.path.exists(dance_label_folder):
            os.makedirs(dance_label_folder)

        dance_label_url = os.path.join(dance_label_folder, img_file_name)
        cv2.imwrite(dance_label_url, image)
        predict_labels.append(label1_dance)

        # s = ' -- '.join([str(elem) for elem in list_sub])
        # list_subs.append(s)
        # s = ' -- '.join([str(elem) for elem in list_min])
        # list_mins.append(s)
        s = ' or '.join([str(elem) for elem in list_label])
        list_labels.append(s)

    col1 = np.array(TEST_IMAGES)
    col2 = np.array(labeled_value)
    col3 = np.array(predict_labels)
    col4 = np.array(list_labels)
    # col5 = np.array(list_mins)
    # col6 = np.array(list_subs)

    # label_df = pd.DataFrame({"Image": col1, "Ref Label": col2, "Predicted 1 Label": col3,
    #                         "Possible Pose": col4, "Pose Value": col5, "Diff per Joint": col6})

    label_df = pd.DataFrame({"Image": col1, "Ref Label": col2,
                            "Predict Pose": col3, "Possible Pose": col4})

    label_df.to_csv(os.path.join(
        result_image_folder, "Prediction_Result.csv"), index=False)

    # average_time = total_time / len(TEST_IMAGES)
    # change_text(f"Avg time: {round(average_time, 2)}\n")
    change_text("Finished Testing\n")


def quit():
    if messagebox.askokcancel("Quit", "Do you really want to quit?"):
        # Function which closes the window.
        global my_root
        my_root.destroy()
        my_root = None
        print('destroy root')


if __name__ == '__main__':
    vp_start_gui()