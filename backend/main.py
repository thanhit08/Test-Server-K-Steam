import subprocess
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import sys
import os
import time
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi.middleware.cors import CORSMiddleware
import argparse
from layers.similarity import manhattan_distance_w_max
import os
import sys
from datetime import datetime
import torch
import torch.nn as nn
from common.dataset import CustomPosePoseDataset2Dxy_numpy_with_keypoint
from tqdm import tqdm
import json
import numpy as np
import shutil
import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import mediapipe as mp
import cv2
import csv
from pathlib import Path
from dtw import *
from databasemanager import DatabaseManager
from ultralytics import YOLO
from contextlib import asynccontextmanager


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://192.168.56.1:3000", "http://192.168.56.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models['yolo8pose'] = YOLO('yolov8n-pose.pt')
    yield


class KSteamPoseModel(nn.Module):
    """ 
    Base Drop 

    """

    def __init__(self, num_joints_in, in_features, filter_width, filter_width2,  out_channels=1024, dropout=0.25, dense=False) -> None:
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.filter_width = filter_width
        self.filter_width2 = filter_width2
        self.out_channels = out_channels
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.pad = [filter_width // 2]

        #! input channels :num_joints_in * in_features -> feature dimension을 1로 보겠다는 의미

        # 33 , 3 channel
        self.model = nn.Sequential(
            conv_bn(in_features, 64, 2),
            conv_bn(64, 128, 1),
            conv_bn(128, 256, 2),
            conv_bn(256, 256, 1),
            conv_bn(256, 512, 2),
            conv_bn(512, 512, 1),
            conv_bn(512, 1024, 2),
            conv_bn(1024, 1024, 1),
            conv_bn(1024, 1024, 1),
            conv_bn(1024, 1024, 1),
            conv_bn(1024, 1024, 1),
            conv_bn(1024, 1024, 1),
            conv_bn(1024, 2048, 2),
            conv_bn(2048, 2048, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.regression1 = nn.Linear(2048, 1024)
        self.regression2 = nn.Linear(1024, 512)

    def forward(self, x):
        print(x.size())
        x = x.permute(0, 3, 2, 1)
        x = self.model(x)
        x = x.view(-1, 2048)
        x = self.regression1(x)
        x = self.regression2(x)

        return x


# Path to the folder containing images
images_folder = Path("images")
velocity_time_interval = 10

angleJoints = [  # [0, 12, 11, 12],  # Neck
    [9, 2, 3], [2, 3, 4],  # Right Shoulder, Right Elbow
    [12, 5, 6], [5, 6, 7],  # Left Shoulder, Left Elbow
    # [12, 24, 26],  # Hip
    [8, 9, 10], [9, 10, 11],  # Right Leg, Right Knee
    [8, 12, 13], [12, 13, 14]]  # Left Leg, Left Knee

angleJointNames = ["Right Shoulder", "Right Elbow", "Left Shoulder", "Left Elbow",
                   "Right Leg", "Right Knee", "Left Leg", "Left Knee",
                   "Right Shoulder & Elbow", "Left Shoulder & Elbow"]

angleJointWeight = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5]


# distance between two velocity values
five_velocity_score_boundary = 0.1
four_velocity_score_boundary = 0.15
three_velocity_score_boundary = 0.25
two_velocity_score_boundary = 0.40

# distance between two angles
five_acc_score_boundary = 0.15
four_acc_score_boundary = 0.25
three_acc_score_boundary = 0.40
two_acc_score_boundary = 0.70

acc_score_boundary = {
    'Elbow': [0.15, 0.17, 0.19, 0.25],
    'Knee': [0.03, 0.04, 0.05, 0.06],
    'Leg': [0.03, 0.04, 0.05, 0.06],
    'Shoulder & Elbow': [0.1, 0.13, 0.15, 0.2],
    'Shoulder': [0.1, 0.15, 0.17, 0.2],    
}

velocity_score_boundary = {
    'Elbow': [0.05, 0.07, 0.1, 0.2],
    'Knee': [0.03, 0.04, 0.05, 0.06],
    'Leg': [0.01, 0.02, 0.05, 0.06],
    'Shoulder & Elbow': [0.02, 0.03, 0.05, 0.07],
    'Shoulder': [0.02, 0.03, 0.04, 0.05],    
}

# distance between two frames' indexes
five_timing_score_boundary = 6.0
four_timing_score_boundary = 10.0
three_timing_score_boundary = 20.0
two_timing_score_boundary = 36.0

# distance between two frames' indexes
new_five_timing_score_boundary = 1.0
new_four_timing_score_boundary = 2.0
new_three_timing_score_boundary = 4.0
new_two_timing_score_boundary = 7.0

dict_joint_index_count = {0: 0, 1: 0,
                          2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,
                          8: 0, 9: 0, 10: 0, 11: 0}
dict_joint_index_score = {0: 0, 1: 0,
                          2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,
                          8: 0, 9: 0, 10: 0, 11: 0}
dict_joint_difference_score = {0: 0, 1: 0,
                               2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0,
                               8: 0, 9: 0, 10: 0, 11: 0}


def model_init(model_path):
    num_joints_in = 33  # 0 ~4
    in_features = 2  # x, y, z
    out_channels = 768
    filter_width = 3
    filter_width2 = 1200

    # Model 준비
    model_pos = KSteamPoseModel(num_joints_in, in_features, filter_width, filter_width2,
                                dropout=0.25, out_channels=out_channels,
                                dense=False)

    model_pos.load_state_dict(torch.load(model_path), strict=False)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)
    model_pos.to(device)
    model_pos.eval()
    return model_pos


def model_test(
    model_path: str,
    student_json: str,
    teacher_json: str
) -> None:

    # Prepair Dataset
    print("Loading 2D detections...")
    start_time = datetime.now()

    # * model initialization
    model_pos = model_init(model_path)

    # * data preperation
    section_length = 600
    pos_min = -3193.8388671875
    pos_max = 2831.092529296875

    student_pose_data = json.load(open(student_json))
    # * only x,y channel
    numpy_3d_arrays_student = [[joint[:2]
                                for joint in pose]for pose in student_pose_data]
    np_student_pose = np.array(numpy_3d_arrays_student)
    np_student_pose = np_student_pose[:section_length]

    teacher_pose_data = json.load(open(teacher_json))
    # *  only x,y channel
    #! if techer data fps same as student data - you can you following line
    # numpy_3d_arrays_teacher = [[joint[:2] for joint in pose]for pose in teacher_pose_data]
    #! but it techer data's fps are 60 - you should use following
    numpy_3d_arrays_teacher = [[joint[:2] for joint in pose]
                               for pose in teacher_pose_data[::2]]
    np_teacher_pose = np.array(numpy_3d_arrays_teacher)
    np_teacher_pose = np_teacher_pose[:section_length]

    # * To Tensor
    tensor_student_pose = torch.tensor(np_student_pose, dtype=torch.float32)
    tensor_student_pose = torch.unsqueeze(tensor_student_pose, dim=0)
    tensor_teacher_pose = torch.tensor(np_teacher_pose, dtype=torch.float32)
    tensor_teacher_pose = torch.unsqueeze(tensor_teacher_pose, dim=0)

    # * normalization - normalize twice
    tensor_student_pose = (tensor_student_pose - pos_min) / (pos_max - pos_min)
    tensor_teacher_pose = (tensor_teacher_pose - pos_min) / (pos_max - pos_min)
    tensor_student_pose = (tensor_student_pose - pos_min) / (pos_max - pos_min)
    tensor_teacher_pose = (tensor_teacher_pose - pos_min) / (pos_max - pos_min)
    # label = label /5.

    predicted_score = single_request(
        model_pos, tensor_student_pose, tensor_teacher_pose)

    end_time = datetime.now()
    diff_time = end_time - start_time

    print(f"predicted_score : {predicted_score}")
    print(f"diff_time : {diff_time}")

    return predicted_score


def read_data(pose_test_data_path, pose_teacher_data_path, bar_length, normalize=True, pos_min=-3193.8388671875, pos_max=2831.092529296875):
    pose_data = json.load(open(pose_test_data_path))
    print(len(pose_data))
    numpy_3d_arrays_student = []
    for pose in pose_data:
        temp = []
        for idx, joint in enumerate(pose):
            if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                temp.append(joint[:2])
            else:
                temp.append([0., 0.])
        numpy_3d_arrays_student.append(temp)

    np_pose_student_ret = np.array(numpy_3d_arrays_student)
    np_pose_student_ret = np_pose_student_ret[:bar_length]

    pose_teacher_data = json.load(open(pose_teacher_data_path))
    numpy_3d_arrays_teacher = []
    for pose in pose_teacher_data[::2]:
        temp = []
        for idx, joint in enumerate(pose):
            if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                temp.append(joint[:2])
            else:
                temp.append([0., 0.])
        numpy_3d_arrays_teacher.append(temp)
    np_pose_teacher_ret = np.array(numpy_3d_arrays_teacher)
    np_pose_teacher_ret = np_pose_teacher_ret[:bar_length]

    image_student = torch.tensor(np_pose_student_ret, dtype=torch.float32)
    image_teacher = torch.tensor(np_pose_teacher_ret, dtype=torch.float32)
    if normalize:
        image_student = (image_student - pos_min) / (pos_max - pos_min)
        image_teacher = (image_teacher - pos_min) / (pos_max - pos_min)
    return image_student, image_teacher,


def draw_angle_section(student_angle_data, teacher_angle_data, angle_name):
    # init plot and set size of the figure
    fig, axs = plt.subplots(2)
    # setup x, y axis names
    axs[0].set(xlabel='frames')
    axs[1].set(xlabel='frames')
    # make distance between axs[0] and axs[1]
    fig.subplots_adjust(hspace=0.5)
    # make title of the plot
    fig.suptitle(f'Angle {angle_name} Visualization')
    # make title of axs[0] and axs[1]
    axs[0].set_title('Student')
    axs[1].set_title('Teacher')
    # align title to left side
    axs[0].title.set_position([0, 1.05])
    axs[1].title.set_position([0, 1.05])

    # draw line plot for student_angle_data
    axs[0].plot(student_angle_data)
    # draw line plot for teacher_angle_data
    axs[1].plot(teacher_angle_data)

    # add grid to the plot
    axs[0].grid()
    axs[1].grid()

    # add x, y axis names to the plot
    axs[0].set(xlabel='frames', ylabel='angle')
    axs[1].set(xlabel='frames', ylabel='angle')


def draw_section(numpy_array_std, numpy_array_ref):
    def process_joint(vals):
        return (vals[0] + vals[1]) / 255.

    # init plot and set size of the figure
    fig, axs = plt.subplots(2)
    # setup x, y axis names
    axs[0].set(xlabel='frames')
    axs[1].set(xlabel='frames')
    # make distance between axs[0] and axs[1]
    fig.subplots_adjust(hspace=0.5)
    # make title of the plot
    fig.suptitle('Pose Data Visualization')
    # make title of axs[0] and axs[1]
    axs[0].set_title('Student')
    axs[1].set_title('Teacher')
    # align title to left side
    axs[0].title.set_position([0, 1.05])
    axs[1].title.set_position([0, 1.05])

    # using linspace to create x, y values for contour plot
    x_vals = np.linspace(
        0, numpy_array_std.shape[0]-1, numpy_array_std.shape[0])
    y_vals = np.linspace(
        0, numpy_array_std.shape[1]-1, numpy_array_std.shape[1])
    z_vals = []
    for this_y in y_vals:
        this_z = []
        for this_x in x_vals:
            # calculate the value for each x,y point using process_joint function
            this_val = process_joint(numpy_array_std[int(this_x), int(this_y)])
            this_z.append(this_val)
        z_vals.append(this_z)

    # calculate contour plot using x, y, z values and set number of contour lines
    cs0 = axs[0].contourf(x_vals, y_vals, z_vals, 8)
    # add contour lines to the plot
    fig.colorbar(cs0, ax=axs[0])
    # add contour labels to the plot
    contour_labels = axs[0].contour(x_vals, y_vals,
                                    z_vals, 8, colors='black', linewidths=1)    #
    im0 = axs[0].clabel(contour_labels, inline=1, linewidths=10)

    # using linspace to create x, y values for contour plot
    x_vals = np.linspace(
        0, numpy_array_ref.shape[0]-1, numpy_array_ref.shape[0])
    y_vals = np.linspace(
        0, numpy_array_ref.shape[1]-1, numpy_array_ref.shape[1])
    z_vals = []
    for this_y in y_vals:
        this_z = []
        for this_x in x_vals:
            # calculate the value for each x,y point using process_joint function
            this_val = process_joint(numpy_array_ref[int(this_x), int(this_y)])
            this_z.append(this_val)
        z_vals.append(this_z)

    # calculate contour plot using x, y, z values and set number of contour lines
    cs1 = axs[1].contourf(x_vals, y_vals, z_vals, 8)
    # add contour lines to the plot
    fig.colorbar(cs1, ax=axs[1])
    # add contour labels to the plot
    contour_labels = axs[1].contour(x_vals, y_vals,
                                    z_vals, 8, colors='black', linewidths=1)
    im1 = axs[1].clabel(contour_labels, inline=1, linewidths=10)


def single_request(model_pos, student_pose, teacher_pose):
    # normalize inputs
    # check device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    student_pose = student_pose.to(device)
    teacher_pose = teacher_pose.to(device)

    student_outputs = model_pos(student_pose)
    teacher_outputs = model_pos(teacher_pose)

    label_predict = manhattan_distance_w_max(
        student_outputs, teacher_outputs, 5.0)

    return label_predict


def get_codec_info(video_path: str) -> str:
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name',
                            '-of', 'default=noprint_wrappers=1:nokey=1', video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.decode().strip()


def check_profile(video_path: str) -> str:
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=profile',
                            '-of', 'default=noprint_wrappers=1:nokey=1', video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.decode().strip()


def generate_new_name(file_type: str) -> str:
    current_time = time.strftime("%Y%m%d-%H%M%S")
    return f"{current_time}_{file_type}"


def music_synchronize(output_folder: str, student_music_filename: str, teacher_music_filename: str) -> None:
    # use dynamic time warping to align music using librosa library
    # load student music and teacher music using librosa library
    x_student, sr_student = librosa.load(
        os.path.join(output_folder, student_music_filename))
    x_teacher, sr_teacher = librosa.load(
        os.path.join(output_folder, teacher_music_filename))
    # extract chroma features from music using librosa library
    n_fft = 2048
    hop_length = 512
    chroma_student = librosa.feature.chroma_stft(
        y=x_student, sr=sr_student, tuning=0, norm=2, hop_length=hop_length, n_fft=n_fft)
    chroma_teacher = librosa.feature.chroma_stft(
        y=x_teacher, sr=sr_teacher, tuning=0, norm=2, hop_length=hop_length, n_fft=n_fft)
    # calculate dynamic time warping distance using librosa library
    d, wp = librosa.sequence.dtw(
        X=chroma_student, Y=chroma_teacher, subseq=True)
    # visualize dynamic time warping path using librosa library

    plt.imshow(d, origin='lower', cmap='gray', interpolation='nearest')
    plt.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='red')
    plt.legend()
    # save the plot as an image
    plt.savefig(os.path.join(output_folder, "dtw.png"))
    # clean and close plt
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(16, 8))

    # Plot x_1
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(x_student, sr=sr_student)
    plt.title('Slower Version $X_1$')
    ax1 = plt.gca()

    # Plot x_2
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(x_teacher, sr=sr_teacher)
    plt.title('Slower Version $X_2$')
    ax2 = plt.gca()

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 60
    points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))

    # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    # Draw matching line
    teacher_start_times = []
    student_start_times = []
    for tp1, tp2 in wp[points_idx] * hop_length / sr_student:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

        # draw a line
        line = mpl.lines.Line2D((coord1[0], coord2[0]),
                                (coord1[1], coord2[1]),
                                transform=fig.transFigure,
                                color='r')
        lines.append(line)
        teacher_start_times.append(tp1)
        student_start_times.append(tp2)
    fig.lines = lines
    plt.tight_layout()
    # save the plot as an image
    plt.savefig(os.path.join(output_folder, "dtw2.png"))

    return student_start_times, teacher_start_times


def append_keypoints(op_kp, yolo_kp):
    op_kp.append(*yolo_kp[:2], 0.0)


def yolo_to_op(yolo_keypoints):
    # convert yolo keypoints to open pose keypoints
    op_keypoints = []
    op_keypoints.append([*yolo_keypoints[0][:2], 0.0])
    op_keypoints.append([(yolo_keypoints[5][0] + yolo_keypoints[6][0]) /
                        2.0, (yolo_keypoints[5][1] + yolo_keypoints[6][1])/2.0, 0.0])
    op_keypoints.append([*yolo_keypoints[6][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[8][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[10][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[5][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[7][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[9][:2], 0.0])
    op_keypoints.append([(yolo_keypoints[11][0] + yolo_keypoints[12][0]) /
                        2.0, (yolo_keypoints[11][1] + yolo_keypoints[12][1])/2.0, 0.0])
    op_keypoints.append([*yolo_keypoints[12][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[14][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[16][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[11][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[13][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[15][:2], 0.0])
    
    op_keypoints.append([*yolo_keypoints[2][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[1][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[4][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[3][:2], 0.0])
    
    op_keypoints.append([*yolo_keypoints[15][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[15][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[15][:2], 0.0])
    
    op_keypoints.append([*yolo_keypoints[16][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[16][:2], 0.0])
    op_keypoints.append([*yolo_keypoints[16][:2], 0.0])
    return op_keypoints


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


def write_dance_processed_data_to_csv(processed_data_csv, data_landmarks):
    processed_data = []
    with open(processed_data_csv, 'w', newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        # csv_writer.writerow(title_name)
        for i_landmark in data_landmarks:
            # i_processed_data = pose_evaluation.process_2D_data(i_landmark)
            processed_data.append(i_landmark)
            csv_writer.writerow(i_landmark)

    return processed_data


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    # normalize angle to 0~1
    angle = angle / 180.0
    return angle


def extract_pose_from_video_v2(output_folder, video_filename):
    previous_yolo_result_keypoints = None
    video_capture = cv2.VideoCapture(video_filename)
    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))    
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    yolo_video = cv2.VideoWriter(video_filename.replace('.mp4', '_yolo.mp4'), cv2.VideoWriter_fourcc(
        *'mp4v'), frame_rate, (frame_width, frame_height))
    angle_data = []
    raw_yolo_pose_data = []
    raw_op_pose_data = []
    normalized_op_pose_data = []
    ml_models['yolo8pose'] = YOLO('yolov8n-pose.pt')
    frame_index = 0
    while True:
        # extract the frame from the video
        success, image = video_capture.read()
        # if the frame was not successfully extracted, break out of the loop
        if not success:
            break
        # clone frame 
        frame_clone = image.copy()
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        frame_array = np.array(frame_rgb)
        yolo_results = ml_models['yolo8pose'].predict(frame_array)
        if len(yolo_results) == 0:
            break
        yolo_result_keypoints = yolo_results[0].keypoints.xyn.cpu().numpy()[0]
        # Draw the yolo keypoints on the frame
        if frame_index == 273:
            print('yolo_result_keypoints: ', yolo_result_keypoints)
        for i in range(len(yolo_result_keypoints)):
            x = int(yolo_result_keypoints[i][0] * frame_width)
            y = int(yolo_result_keypoints[i][1] * frame_height)
            if x == 0 and y == 0 and previous_yolo_result_keypoints is not None:
                x = int(previous_yolo_result_keypoints[i][0] * frame_width)
                y = int(previous_yolo_result_keypoints[i][1] * frame_height)
                yolo_result_keypoints[i][0] = previous_yolo_result_keypoints[i][0]
                yolo_result_keypoints[i][1] = previous_yolo_result_keypoints[i][1]
            cv2.circle(frame_clone, (x, y), 5, (0, 0, 255), -1)   
        # Draw the line from the index 5 to 6
        cv2.line(frame_clone, (int(yolo_result_keypoints[5][0] * frame_width), int(yolo_result_keypoints[5][1] * frame_height)), (int(yolo_result_keypoints[6][0] * frame_width), int(yolo_result_keypoints[6][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 5 to 7
        cv2.line(frame_clone, (int(yolo_result_keypoints[5][0] * frame_width), int(yolo_result_keypoints[5][1] * frame_height)), (int(yolo_result_keypoints[7][0] * frame_width), int(yolo_result_keypoints[7][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 7 to 9
        cv2.line(frame_clone, (int(yolo_result_keypoints[7][0] * frame_width), int(yolo_result_keypoints[7][1] * frame_height)), (int(yolo_result_keypoints[9][0] * frame_width), int(yolo_result_keypoints[9][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 6 to 8
        cv2.line(frame_clone, (int(yolo_result_keypoints[6][0] * frame_width), int(yolo_result_keypoints[6][1] * frame_height)), (int(yolo_result_keypoints[8][0] * frame_width), int(yolo_result_keypoints[8][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 8 to 10
        cv2.line(frame_clone, (int(yolo_result_keypoints[8][0] * frame_width), int(yolo_result_keypoints[8][1] * frame_height)), (int(yolo_result_keypoints[10][0] * frame_width), int(yolo_result_keypoints[10][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 6 to 12
        cv2.line(frame_clone, (int(yolo_result_keypoints[6][0] * frame_width), int(yolo_result_keypoints[6][1] * frame_height)), (int(yolo_result_keypoints[12][0] * frame_width), int(yolo_result_keypoints[12][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 5 to 11
        cv2.line(frame_clone, (int(yolo_result_keypoints[5][0] * frame_width), int(yolo_result_keypoints[5][1] * frame_height)), (int(yolo_result_keypoints[11][0] * frame_width), int(yolo_result_keypoints[11][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 11 to 12
        cv2.line(frame_clone, (int(yolo_result_keypoints[11][0] * frame_width), int(yolo_result_keypoints[11][1] * frame_height)), (int(yolo_result_keypoints[12][0] * frame_width), int(yolo_result_keypoints[12][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 11 to 13
        cv2.line(frame_clone, (int(yolo_result_keypoints[11][0] * frame_width), int(yolo_result_keypoints[11][1] * frame_height)), (int(yolo_result_keypoints[13][0] * frame_width), int(yolo_result_keypoints[13][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 13 to 15
        cv2.line(frame_clone, (int(yolo_result_keypoints[13][0] * frame_width), int(yolo_result_keypoints[13][1] * frame_height)), (int(yolo_result_keypoints[15][0] * frame_width), int(yolo_result_keypoints[15][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 12 to 14
        cv2.line(frame_clone, (int(yolo_result_keypoints[12][0] * frame_width), int(yolo_result_keypoints[12][1] * frame_height)), (int(yolo_result_keypoints[14][0] * frame_width), int(yolo_result_keypoints[14][1] * frame_height)), (0, 255, 0), 2)
        # Draw the line from the index 14 to 16
        cv2.line(frame_clone, (int(yolo_result_keypoints[14][0] * frame_width), int(yolo_result_keypoints[14][1] * frame_height)), (int(yolo_result_keypoints[16][0] * frame_width), int(yolo_result_keypoints[16][1] * frame_height)), (0, 255, 0), 2)
        yolo_video.write(frame_clone)
        
        previous_yolo_result_keypoints = yolo_result_keypoints
        yolo_to_op_keypoints = yolo_to_op(yolo_result_keypoints)
        normalized_op_data = nom_op_to_Hips(yolo_to_op_keypoints)        
        # calculate angle between two op_keypoints
        angles = []
        for angleJoint in angleJoints:
            try:
                angle_value = calculate_angle([*yolo_to_op_keypoints[angleJoint[0]]],
                                              [*yolo_to_op_keypoints[angleJoint[1]]],
                                              [*yolo_to_op_keypoints[angleJoint[2]]])
            except Exception as e:
                angle_value = 360
            angles.append(angle_value)
        # if angles[0] and angles[1] is not 360, then create a new angle = angles[0] + angles[1]
        if angles[0] != 360 and angles[1] != 360:
            angles.append((angles[0] + angles[1])/2)
        else:
            angles.append(360)
        # if angles[2] and angles[3] is not 360, then create a new angle = angles[2] + angles[3]
        if angles[2] != 360 and angles[3] != 360:
            angles.append((angles[2] + angles[3])/2)
        else:
            angles.append(360)
        angle_data.append(angles)        
        normalized_op_pose_data.append(normalized_op_data)
        raw_yolo_pose_data.append(yolo_result_keypoints)
        raw_op_pose_data.append(yolo_to_op_keypoints)
        frame_index += 1
    # release the video object
    video_capture.release()
    yolo_video.release()
    
    # convert video to mp4 format
    fixed_video_yolo_filename = video_filename.replace('.mp4', '_yolo_fixed.mp4')
    cmd = f"ffmpeg -hwaccel cuvid -i {video_filename.replace('.mp4', '_yolo.mp4')} -c:v h264_nvenc -preset fast {fixed_video_yolo_filename} -y"
    try:
        os.system(cmd)
    except Exception as e:
        print(e)
    
    cv2.destroyAllWindows()
    # save the pose data as a json file
    json_filename = video_filename.replace('.mp4', '_mp_pose.json')
    json_angle_filename = video_filename.replace('.mp4', '_angle.json')
    json_raw_yolo_filename = video_filename.replace('.mp4', '_raw_yolo.json')
    json_raw_op_filename = video_filename.replace('.mp4', '_raw_op.json')
    
    with open(json_filename, 'w') as f:
        print('writing normalized_op_pose_data json file')
        json.dump(normalized_op_pose_data, f, cls=FloatEncoder, indent=4)
    with open(json_angle_filename, 'w') as f:
        print('writing angle_data json file')
        json.dump(angle_data, f, cls=FloatEncoder, indent=4)
    with open(json_raw_yolo_filename, 'w') as f:
        print('writing raw_op_pose_data json file')
        json.dump(raw_yolo_pose_data, f, cls=NumpyEncoder, indent=4)
    with open(json_raw_op_filename, 'w') as f:
        print('writing raw_op_pose_data json file')
        json.dump(raw_op_pose_data, f, cls=FloatEncoder, indent=4)

    return json_filename, json_angle_filename

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class FloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def extract_pose_from_video(output_folder, video_filename):
    # extract pose from student section video using mediapipe
    # initialize the MediaPipe Pose solution drawing utils and style
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, enable_segmentation=True, model_complexity=0, min_detection_confidence=0.5)
    # read the input image
    video_capture = cv2.VideoCapture(video_filename)
    # initialize an array to store the pose data
    mp_pose_data = []
    op_pose_data = []
    normalized_op_pose_data = []
    angle_data = []
    # loop over the frames of the video
    while True:
        # extract the frame from the video
        success, image = video_capture.read()
        # if the frame was not successfully extracted, break out of the loop
        if not success:
            break
        img_height = image.shape[0]
        img_width = image.shape[1]
        # convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect poses in the image
        results = mp_pose.process(image)
        # if there are no poses detected in the image, continue with the next frame
        angles = []
        if not results.pose_landmarks:
            for angleJoint in angleJoints:
                angle_value = 360
                angles.append(angle_value)
            angle_data.append(angles)
            continue
        key_points = []
        op_keypoints = []
        for landmark in results.pose_landmarks.landmark:
            key_point = [landmark.x * img_width,
                         landmark.y * img_height, landmark.z * img_width]
            key_points.append(key_point)
        # extract the pose landmarks
        pose_landmarks = results.pose_landmarks.landmark
        # append the pose landmarks to the pose data array
        mp_pose_data.append(pose_landmarks)
        # convert key_points to open pose key_points using mp_to_op function
        op_keypoints = mp_to_op(key_points)
        # append the open pose key_points to the open pose data array
        op_pose_data.append(op_keypoints)
        normalized_op_data = nom_op_to_Hips(op_keypoints)
        # calculate angle between two op_keypoints
        for angleJoint in angleJoints:
            try:
                angle_value = calculate_angle([op_keypoints[angleJoint[0]][0], op_keypoints[angleJoint[0]][1], op_keypoints[angleJoint[0]][2]],
                                              [op_keypoints[angleJoint[1]][0], op_keypoints[angleJoint[1]]
                                                  [1], op_keypoints[angleJoint[1]][2]],
                                              [op_keypoints[angleJoint[2]][0], op_keypoints[angleJoint[2]][1], op_keypoints[angleJoint[2]][2]])
            except:
                angle_value = 360
            angles.append(angle_value)
        # if angles[0] and angles[1] is not 360, then create a new angle = angles[0] + angles[1]
        if angles[0] != 360 and angles[1] != 360:
            angles.append((angles[0] + angles[1])/2)
        else:
            angles.append(360)
        # if angles[2] and angles[3] is not 360, then create a new angle = angles[2] + angles[3]
        if angles[2] != 360 and angles[3] != 360:
            angles.append((angles[2] + angles[3])/2)
        else:
            angles.append(360)
        # if angles[4] and angles[5] is not 360, then create a new angle = angles[4] + angles[5]
        if angles[4] != 360 and angles[5] != 360:
            angles.append((angles[4] + angles[5])/2)
        else:
            angles.append(360)
        # if angles[6] and angles[7] is not 360, then create a new angle = angles[6] + angles[7]
        if angles[6] != 360 and angles[7] != 360:
            angles.append((angles[6] + angles[7])/2)
        else:
            angles.append(360)
        angle_data.append(angles)
        normalized_op_pose_data.append(normalized_op_data)
    # release the video object
    video_capture.release()
    # close the MediaPipe Pose solution
    mp_pose.close()
    # save the pose data as a json file
    json_filename = video_filename.replace('.mp4', '_mp_pose.json')
    json_angle_filename = video_filename.replace('.mp4', '_angle.json')
    with open(json_filename, 'w') as f:
        json.dump(normalized_op_pose_data, f, indent=4)
    with open(json_angle_filename, 'w') as f:
        json.dump(angle_data, f, indent=4)

    return json_filename, json_angle_filename


class ai_data(BaseModel):
    model: str
    good_or_bad: str


def crop_video(video_filename: str, start_time: float, end_time: float, output_filename: str) -> bool:
    # Define crop info based on start and end time
    crop_info = ""
    if not math.isclose(end_time, 0):
        crop_info = f" -ss {start_time} -to {end_time} "

    # Define the FFMPEG command to process the video file
    # cmd = f'ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {video_filename} {crop_info} -c:a copy -r 60 {output_filename} -y'
    cmd = f'ffmpeg -hwaccel cuvid -i {video_filename} {crop_info} -c:v h264_nvenc -preset fast -c:a copy -r 60 {output_filename}  -y'

    # Execute the FFMPEG command using the os.system function
    try:
        os.system(cmd)
        return True
    except:
        print("Failed to pre-process video!")
        return False


def preprocess_video(video_filename: str, output_filename: str) -> bool:
    # convert video to 60 fps using ffmpeg library (GPU accelerated) on windows
    # cmd = f'ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {video_filename}  -c:a copy -r 60 {output_filename} -y'
    cmd = f'ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {video_filename}  -c:a copy -r 60 {output_filename} -y'

    try:
        os.system(cmd)
        print("Pre-processing video completed!")
        return True
    except:
        print("Failed to pre-process video!")
        return False


def calculate_instantaneous_velocity(angle_data):
    new_motion_data_array = []
    half_velocity_time_interval = int(velocity_time_interval/2)
    for index_data in range(len(angle_data))[half_velocity_time_interval:-1-half_velocity_time_interval]:
        new_motion_data = (angle_data[index_data - half_velocity_time_interval] -
                           angle_data[index_data + half_velocity_time_interval])/velocity_time_interval
        new_motion_data_array.append(new_motion_data)
    return np.ravel(new_motion_data_array)

def calculate_instantaneous_velocity_v2(angle_data):
    time = np.linspace(0, 1200, num=1200)  # Assuming 100 frames in 20 seconds
    angle_data = np.array(angle_data)
    angle_data_speed = np.gradient(angle_data, time)
    angle_data_acceleration = np.gradient(angle_data_speed, time)
    angle_data_velocity = np.gradient(angle_data_acceleration, time)
    return angle_data_velocity


def check_level_velocity(score, index_name):
    index_name = index_name.replace('Left ', '').replace('Right ', '')
    boundary_score = acc_score_boundary[index_name]
    level = check_level(score, *boundary_score)
    # level = check_level(score,
    #                     five_velocity_score_boundary,
    #                     four_velocity_score_boundary,
    #                     three_velocity_score_boundary,
    #                     two_velocity_score_boundary)

    # return level
    return level


def check_level_accuracy(score, index_name):
    index_name = index_name.replace('Left ', '').replace('Right ', '')
    boundary_score = acc_score_boundary[index_name]
    level = check_level(score, *boundary_score)

    # return level
    return level

def check_level_timing(score):
    
    level = check_level(score,
                        new_five_timing_score_boundary,
                        new_four_timing_score_boundary,
                        new_three_timing_score_boundary,
                        new_two_timing_score_boundary)

    # return level
    return level


def check_level(score,
                five_boundary,
                four_boundary,
                three_boundary,
                two_boundary):
    level_boundaries = {
        five_boundary: 5,
        four_boundary: 4,
        three_boundary: 3,
        two_boundary: 0
    }
    level = 0
    for boundary, level_value in level_boundaries.items():
        if score < boundary:
            level = level_value
            break
    return level


def start_3D_data_evaluation_fastdtw(type_data, ref_data, test_data, sub_data_processing_path, index_name):
    # if index_name has Leg or Knee
    # then check average value of test_data
    standing_detected = False
    if type_data == "accuracy":
        test_average = np.mean(np.abs(np.array(test_data)))
        ref_average = np.mean(np.abs(np.array(ref_data)))
        diff_average = np.abs(test_average - ref_average)
        if "Shoulder" in index_name:
            if diff_average > 0.06:
                standing_detected = True
        if "Elbow" in index_name:
            if diff_average > 0.07:
                standing_detected = True
        
        if "Leg" in index_name or "Knee" in index_name:
            if test_average > 0.97:
                standing_detected = True
            
    # Convert input data to numpy arrays
    x = np.array(ref_data)
    y = np.array(test_data)

    # Perform dynamic time warping
    alignment = dtw(x, y, keep_internals=True,
                    step_pattern=rabinerJuangStepPattern(6, "c"), window_type= "sakoechiba", window_args= {"window_size": 7})
    # Get the alignment indices
    index1 = alignment.index1[1:-1]
    index2 = alignment.index2[1:-1]

    # # index1 = index[i+1] - index[i]
    # index1 = [index1[i+1] - index1[i] for i in range(len(index1)-1)]
    # # index2 = index[i+1] - index[i]
    # index2 = [index2[i+1] - index2[i] for i in range(len(index2)-1)]

    # Calculate the DTW distance
    dtw_distance = np.abs(np.subtract(
        [x[i] for i in index1], [y[i] for i in index2]))
    
    if type_data == "velocity":
        dtw_distance = np.abs(np.subtract(
            [i for i in x], [j for j in y]))

    # Save index_a, index_b and dtw_distance_timing_list to csv file
    # with open(os.path.join(sub_data_processing_path, f'{index_name}_{type_data}.csv'), 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["index_a", "index_b", f"{index_name}_{type_data}"])
    #     writer.writerows(zip(index1, index2, dtw_distance))

    # Check duplicate index in index2
    # If index2[i] is duplicate, then remove index2[i] and index1[i]
    # index2_no_dupe = []
    # index1_no_dupe = []
    # for i in range(len(index2)):
    #     if index2[i] not in index2_no_dupe:
    #         index2_no_dupe.append(index2[i])
    #         index1_no_dupe.append(index1[i])
    # index2 = index2_no_dupe
    # index1 = index1_no_dupe

    # index2_no_dupe = []
    # index1_no_dupe = []

    # for i in range(len(index1)):
    #     if index1[i] not in index1_no_dupe:
    #         index1_no_dupe.append(index1[i])
    #         index2_no_dupe.append(index2[i])
    # index1 = index1_no_dupe
    # index2 = index2_no_dupe

    # Convert the DTW distance list to a dictionary
    # non_dup_index_dtw = {}
    # for i in index2:
    #     if index2[i] in non_dup_index_dtw:
    #         non_dup_index_dtw[index2[i]] = (
    #             non_dup_index_dtw[index2[i]] + dtw_distance[i])/2.0
    #     else:
    #         non_dup_index_dtw[index2[i]] = dtw_distance[i]
    # # Convert the dictionary values to a list
    # non_dup_index_dtw = non_dup_index_dtw.values()
    # non_dup_index_dtw = np.array(
    #     list(non_dup_index_dtw))
    # non_duplicate_index_dtw_distance = list(non_duplicate_index_dtw_distance.values())
    # Initialize the DTW distance level list
    level_from_dtw_distance = []

    # Check the level of velocity or accuracy
    if type_data == "velocity":
        level_from_dtw_distance = [check_level_velocity(
            abs(x), index_name) for x in dtw_distance]
    else:
        level_from_dtw_distance = [check_level_accuracy(
            abs(x), index_name) for x in dtw_distance]

    total_level = sum(abs(number) for number in level_from_dtw_distance)
    # plot_title = f"DTW Distance  of {index_name}: {total_level/len(level_from_dtw_distance)}"
    # fig_path = os.path.join(sub_data_processing_path,
    #                         f'{index_name}_{type_data}_dtw_distance_pic.jpg')
    # save_fig(x, index1, y, index2, plot_title, fig_path)
    # Return the results
    
    # # Initialize another plot
    # fig, ax = plt.subplots(figsize=(24, 10))
    # # Draw lines between data points
    # for [map_x, map_y] in zip(index1, index2):
    #     ax.plot([map_x, map_y], [x[map_x], y[map_y]], '--k', linewidth=1)
    # # Plot the data points
    # ax.plot(x, '-ro', label='x', linewidth=1, markersize=2,
    #         markerfacecolor='forestgreen', markeredgecolor='forestgreen')
    # ax.plot(y, '-bo', label='y', linewidth=1, markersize=2,
    #         markerfacecolor='skyblue', markeredgecolor='skyblue')
    # # Set the title of the plot
    # ax.set_title(
    #     f"DTW Distance  of {index_name}: {total_level/len(level_from_dtw_distance)}", fontsize=28, fontweight="bold")
    # # Save the plot as a jpg image
    # plt.savefig(os.path.join(sub_data_processing_path,
    #                          f'{index_name}_{type_data}_dtw_distance_pic.jpg'))
    # # Clear the current figure
    # plt.clf()
    # # Close the figure
    # plt.close(fig)
    
    if standing_detected:
        total_level = 0
    return zip(index1, index2), total_level, dtw_distance, level_from_dtw_distance


def save_fig(data1, index1, data2, index2, plot_title, fig_path):
    # Initialize another plot
    fig, ax = plt.subplots(figsize=(54, 10))
    # Draw lines between data points
    for [map_1, map_2] in zip(index1, index2):
        ax.plot([map_1, map_2], [data1[map_1], data2[map_2]], '--k', linewidth=1)
    # Plot the data points
    ax.plot(data1, '-ro', label='x', linewidth=1, markersize=3,
            markerfacecolor='forestgreen', markeredgecolor='forestgreen')
    ax.plot(data2, '-bo', label='y', linewidth=1, markersize=3,
            markerfacecolor='skyblue', markeredgecolor='skyblue')
    # Set the title of the plot
    ax.set_title(plot_title, fontsize=28, fontweight="bold")
    plt.savefig(fig_path)
    # Clear the current figure
    plt.clf()
    # Close the figure
    plt.close(fig)


@app.get("/filter_songs")
def filter_songs():
    songNameDictionary = {
        'Power Up': ['12', '100', '135'],
        'Monster': ['18', '102', '137'],
        'Touch': ['42', '104', '139'],
        'Red Flavor': ['3', '61', '98', '106', '133', '141'],
        'Break Dance Pre': ['7', '63', '108', '116', '143', '151'],
        'Break Dance Unit1': ['13', '110', '145'],
        'Break Dance Unit2': ['20', '112', '147'],
        'Break Dance Unit3': ['43', '114', '149']
    }
    songIDDictionary = {
        'Power Up': [],
        'Monster': [],
        'Touch': [],
        'Red Flavor': [],
        'Break Dance Pre': [],
        'Break Dance Unit1': [],
        'Break Dance Unit2': [],
        'Break Dance Unit3': []
    }
    # Find the folder path that end is in the songNameDictionary
    root_folder_name = "../../data/woosong/"
    # List all the folders in the root folder
    folder_list = os.listdir(os.path.realpath(root_folder_name))
    # List all the folder in the folder_list
    for folder in folder_list:
        list_subjects = os.listdir(os.path.join(root_folder_name, folder))
        for subject in list_subjects:
            if subject == "2" or subject == "7" or subject == "11":
                # List all the folder in the subject 1 folder
                list_song_folders = os.listdir(os.path.join(
                    root_folder_name, folder, subject))
                # Find the name if the song in the songNameDictionary
                for song_folder in list_song_folders:
                    for song_name in list(songNameDictionary.keys()):
                        if song_folder in songNameDictionary[song_name]:
                            song_folder_path = os.path.join(
                                root_folder_name, folder, subject, song_folder)
                            song_dictionary_path = os.path.join(
                                root_folder_name, song_name)
                            if not os.path.exists(song_dictionary_path):
                                os.mkdir(song_dictionary_path)
                            songIDDictionary[song_name].append(
                                song_folder_path)
                            # Rename the song folder name to a random string
                            random_string = "song_" + \
                                str(np.random.randint(1000000))
                            os.rename(song_folder_path, os.path.join(
                                root_folder_name, folder, subject, random_string))
                            # Move the song_folder_path to the song_dictionary_path
                            shutil.move(os.path.join(
                                root_folder_name, folder, subject, random_string), song_dictionary_path)

    print(songIDDictionary)
    return songIDDictionary

# Get all predict id from database


@app.get("/get_predict_ids")
def get_predict_ids():
    predict_ids = []
    # Connect to the database
    db = DatabaseManager('students.db')
    # Get all predict id from database
    for predict in db.get_predicts():
        predict_ids.append(predict[0])
    # Close the database
    db.close()
    return predict_ids

# Service to get score of a predict id from database


@app.get("/get_score")
def get_score(predict_id: str):
    # Connect to the database
    db = DatabaseManager('students.db')
    # Get score from database
    score = db.get_scores_by_predict_id(predict_id)
    # Get section scores from database
    section_scores = []
    if len(score) > 0:
        section_scores = db.get_section_scores_by_score_id(score[-1][0])
    # Close the database
    db.close()
    return score, section_scores

# Service to delete a predict id from database


@app.post("/delete_predicts_by_id")
def delete_predicts_by_id(predict_id: str):
    try:
        # Connect to the database
        db = DatabaseManager('students.db')
        # Delete predict id from database
        db.delete_predicts_by_id(predict_id)
        # Close the database
        db.close()
        return "success"
    except:
        return "fail"


@app.post("/visualize_data")
def visualize_data(student_json_path: str,
                   teacher_json_path: str,
                   data_length: int):
    (batch_image, batch_image_ref) = read_data(
        student_json_path, teacher_json_path, data_length)

    draw_section(batch_image, batch_image_ref)
    # save plot as an image
    plt.savefig(generate_new_name("section.png"))
    plt.clf()
    return {"batch_image": batch_image.shape, "batch_image_ref": batch_image_ref.shape}


@app.post("/visualize_angle_data")
def visualize_angle_data(student_angle_json_path: str,
                         teacher_angle_json_path: str):
    student_angle_data = json.load(open(student_angle_json_path))
    student_major_angle_data = []
    top_left = []
    top_right = []
    bottom_left = []
    bottom_right = []
    for angle_data in student_angle_data:
        if len(angle_data) != 10:
            continue
        top_left.append(angle_data[8])
        top_right.append(angle_data[9])
        bottom_left.append(angle_data[10])
        bottom_right.append(angle_data[11])
    student_major_angle_data.append(top_left)
    student_major_angle_data.append(top_right)
    student_major_angle_data.append(bottom_left)
    student_major_angle_data.append(bottom_right)

    teacher_pose_data = json.load(open(teacher_angle_json_path))
    teacher_major_angle_data = []
    top_left = []
    top_right = []
    bottom_left = []
    bottom_right = []
    for angle_data in teacher_pose_data:
        if len(angle_data) != 10:
            continue
        top_left.append(angle_data[8])
        top_right.append(angle_data[9])
        bottom_left.append(angle_data[10])
        bottom_right.append(angle_data[11])
    teacher_major_angle_data.append(top_left)
    teacher_major_angle_data.append(top_right)
    teacher_major_angle_data.append(bottom_left)
    teacher_major_angle_data.append(bottom_right)
    angle_names = ["Shoulder & Elbow LEFT", "Shoulder & Elbow RIGHT",
                   "Hip & Knee LEFT", "Hip & Knee RIGHT"]
    for i in range(len(student_major_angle_data)):
        draw_angle_section(
            student_major_angle_data[i], teacher_major_angle_data[i], angle_names[i])
        # save plot as an image
        plt.savefig(generate_new_name(f"angle_{i}_section.png"))
        plt.clf()
    return {}


@app.post("/predict_skip_upload")
def predict_skip_upload(data: ai_data):
    # check good_or_bad value
    if data.good_or_bad not in ["good", "bad"]:
        return {"predictions": []}

    print("model", data.model)
    model_path = data.model + '.pt'
    logs = []
    images = []
    # add current time to logs
    logs.append("start")
    logs.append(time.strftime("%Y%m%d-%H%M%S"))

    final_score_text = []
    final_score = 0
    number_of_sections = 0
    # load sections_same_length.json file and read "sections" data as an array of integers
    with open("sections_same_length.json", "r") as f:
        sections = json.load(f)["sections"]
        # add current time to logs with caption "sections loaded"
        logs.append("sections loaded")
        logs.append(time.strftime("%Y%m%d-%H%M%S"))
        # split the video into sections using ffmpeg
        for i, section in enumerate(sections):
            if i == 0:
                continue
            # extract pose from student section video using extract_pose_from_video function
            student_json_filename = f"{data.good_or_bad}_student_{sections[i-1]}_{section}_mp_pose.json"
            if not os.path.exists(student_json_filename):
                break
            # extract pose from teacher section video using extract_pose_from_video function
            teacher_json_filename = f"{data.good_or_bad}_teacher_{sections[i-1]}_{section}_mp_pose.json"
            if not os.path.exists(teacher_json_filename):
                break

            # * model test
            predicted_score = model_test(
                model_path, student_json_filename, teacher_json_filename)
            # add current time to logs with caption "model tested"
            logs.append(f"model tested {sections[i-1]}-{section}")
            logs.append(time.strftime("%Y%m%d-%H%M%S"))

            score = predicted_score.item()
            final_score += score
            number_of_sections += 1
            result = f"Section {i} from {sections[i-1]} to {section} score: " + \
                f"{round(predicted_score.item(), 2)}/5"
            final_score_text.append(result)
            (batch_image, batch_image_ref) = read_data(
                student_json_filename, teacher_json_filename, 600)

            draw_section(batch_image, batch_image_ref)
            # save plot as an image
            image_name = f"{data.good_or_bad}_student_{sections[i-1]}_{section}_data_visualization.png"
            plt.savefig(os.path.join('images', image_name))
            plt.clf()
            images.append(image_name)
            logs.append(f"data visualize {sections[i-1]}-{section}")
            logs.append(time.strftime("%Y%m%d-%H%M%S"))

    average_score = final_score / number_of_sections
    final_score_text.append(
        {"Average score: " + f"{round(average_score, 2)}/5"})

    return {"predictions": final_score_text, "logs": logs, "images": images}


@app.get("/image/{image_name}")
async def get_image(image_name: str):
    image_path = images_folder / image_name

    # Check if the image file exists
    if not image_path.is_file():
        return {"error": "Image not found"}

    # Return the image as a FileResponse
    return FileResponse(path=image_path, media_type="image/jpeg")


# Play video from a url
@app.get("/video/{type}/{predict_id}/{section_name}")
async def get_video(type: str, predict_id: str, section_name: str):
    video_path = os.path.join(
        "Output", f"{predict_id}", f"{type}_{section_name}_yolo_fixed.mp4")

    # Return the video as a FileResponse
    return FileResponse(path=video_path, media_type="video/mp4")


@app.post("/predict_again/")
def predict_again(predict_id: str):
    try:
        # query database to get the predict id
        db = DatabaseManager('students.db')
        predict_data = db.get_predicts_by_id(predict_id)
        # query score from database by predict id
        score_data = db.get_scores_by_predict_id(predict_id)
        # query sections score from database by score id
        section_scores_data = db.get_section_scores_by_score_id(
            score_data[0][0])
        section_scores_names = []
        section_scores = []
        final_score_timing = 0
        final_score_velocity = 0
        final_score_accuracy = 0
        number_of_sections = 0
        for section_score in section_scores_data:
            section_scores_names.append(section_score[2])
            student_json_angle_filename = os.path.join(
                "Output", f"predict_{predict_id}", f"student_{section_score[2]}_angle.json")
            teacher_json_angle_filename = os.path.join(
                "Output", f"predict_{predict_id}", f"teacher_{section_score[2]}_angle.json")
            timing_score, velocity_score, accuracy_score, = calculate_dance_evaluation(
                student_json_angle_filename, teacher_json_angle_filename)
            final_score_timing += timing_score
            final_score_velocity += velocity_score
            final_score_accuracy += accuracy_score
            section_scores.append(
                [timing_score, velocity_score, accuracy_score])
            number_of_sections += 1
            db.update_section_score(
                section_score[0], section_score[3], timing_score, velocity_score, accuracy_score)

        average_score_timing = final_score_timing / number_of_sections
        average_score_velocity = final_score_velocity / number_of_sections
        average_score_accuracy = final_score_accuracy / number_of_sections
        db.update_score(score_data[0][0], score_data[0][3], average_score_timing,
                        average_score_velocity, average_score_accuracy)

        db.close()
        return "success"
    except:
        return "fail"


@app.post("/predict/")
def predict(student: UploadFile = File(...), teacher: UploadFile = File(...), model: Optional[str] = Form(None),):
    # query database to get the predict id
    db = DatabaseManager('students.db')
    db.create_table()

    # model path
    model_path = model + '.pt'
    logs = []
    # add current time to logs
    logs.append("start")
    logs.append(time.strftime("%Y%m%d-%H%M%S"))
    # generate new name for video file
    student_video_filename = "student.mp4"
    teacher_video_filename = "teacher.mp4"

    # insert a new predict to the database
    predict_id = db.insert_predict(
        student_video_filename, teacher_video_filename, model)
    # make a new folder using predict_id
    predict_folder = Path(os.path.join("Output", f"predict_{predict_id}"))
    # if the folder does not exist, create a new folder
    if not predict_folder.exists():
        predict_folder.mkdir()
    else:
        # if the folder exists, delete all files in the folder
        shutil.rmtree(predict_folder)
        predict_folder.mkdir()
    output_id = db.insert_output(predict_id, predict_folder.name)

    student_video_path = os.path.join(predict_folder, student_video_filename)
    teacher_video_path = os.path.join(predict_folder, teacher_video_filename)

    # save file
    with open(student_video_path, "wb") as buffer:
        shutil.copyfileobj(student.file, buffer)
    with open(teacher_video_path, "wb") as buffer:
        shutil.copyfileobj(teacher.file, buffer)
    # add current time to logs with caption "video saved"
    logs.append("video saved")
    logs.append(time.strftime("%Y%m%d-%H%M%S"))

    # student_video_60fps_filename = "student_60fps.mp4"
    # teacher_video_60fps_filename = "teacher_60fps.mp4"
    # student_video_60fps_filename = student_video_filename
    # teacher_video_60fps_filename = teacher_video_filename

    # convert video to 60 fps
    # if not preprocess_video(student_video_filename, student_video_60fps_filename):
    #     return {"predictions": []}
    # if not preprocess_video(teacher_video_filename, teacher_video_60fps_filename):

    # return {"predictions": []}
    # add current time to logs with caption "video preprocessed"
    # logs.append("video preprocessed")
    # logs.append(time.strftime("%Y%m%d-%H%M%S"))

    # extract music from input video using ffmpeg
    # student_music_filename = "student_music.mp3"
    # teacher_music_filename = "teacher_music.mp3"
    # os.system(
    #     f"ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {os.path.join(predict_folder, student_video_60fps_filename)} -f mp3 -ab 192000 -vn {os.path.join(predict_folder,student_music_filename)}")
    # os.system(
    #     f"ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {os.path.join(predict_folder,teacher_video_60fps_filename)} -f mp3 -ab 192000 -vn {os.path.join(predict_folder,teacher_music_filename)}")
    # # add current time to logs with caption "music extracted"
    # logs.append("music extracted")
    # logs.append(time.strftime("%Y%m%d-%H%M%S"))

    # synchronize music using dynamic time warping
    # student_matching_frames, teacher_matching_frames = music_synchronize(predict_folder,
    #                                                                      student_music_filename, teacher_music_filename)

    # add current time to logs with caption "music synchronized"
    # logs.append("music synchronized")
    # logs.append(time.strftime("%Y%m%d-%H%M%S"))

    teacher_start_time = 0
    # Get teacher video duration by using opencv
    cap = cv2.VideoCapture(teacher_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    teacher_end_time = frame_count / fps
    cap.release()
    
    # teacher_end_time = teacher_matching_frames[1]
    student_start_time = 0
    # Get student video duration by using opencv
    cap = cv2.VideoCapture(student_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    student_end_time = frame_count / fps
    cap.release()
    
    # student_end_time = student_matching_frames[1]

    # student_crop_filename = os.path.join(predict_folder, "student_crop.mp4")
    # teacher_crop_filename = os.path.join(predict_folder, "teacher_crop.mp4")
    
    student_crop_filename = os.path.join(predict_folder, "student.mp4")
    teacher_crop_filename = os.path.join(predict_folder, "teacher.mp4")

    # # crop video using ffmpeg based on start and end time
    # crop_video(os.path.join(predict_folder, student_video_60fps_filename), student_start_time,
    #            student_end_time, student_crop_filename)
    # crop_video(os.path.join(predict_folder, teacher_video_60fps_filename), teacher_start_time,
    #            teacher_end_time, teacher_crop_filename)
    # # add current time to logs with caption "video cropped"
    # logs.append("video cropped")
    # logs.append(time.strftime("%Y%m%d-%H%M%S"))

    final_score_text = []
    calculation_final_score_text = []
    final_score = 0
    final_score_timing = 0
    final_score_velocity = 0
    final_score_accuracy = 0
    number_of_sections = 0
    try:
        # load sections_same_length.json file and read "sections" data as an array of integers
        with open("sections_same_length.json", "r") as f:
            sections = json.load(f)["sections"]
            # add current time to logs with caption "sections loaded"
            logs.append("sections loaded")
            logs.append(time.strftime("%Y%m%d-%H%M%S"))

            # create score into scores database
            score_id = db.insert_score(predict_id, output_id, 0, 0, 0, 0)

            # split the video into sections using ffmpeg
            for index, section in enumerate(sections):
                if index == 0:
                    continue

                student_section_time = section - student_start_time
                student_previous_section_time = sections[index - 1] - student_start_time
                if student_previous_section_time < 0:
                    continue
                if student_section_time > student_end_time:
                    break

                # generate new name for student section file
                student_section_filename = os.path.join(
                    predict_folder, f"student_{sections[index-1]}_{section}.mp4")
                # extract section from input video using ffmpeg
                crop_video(student_crop_filename, student_previous_section_time,
                           student_section_time, student_section_filename)
                # add current time to logs with caption "section extracted"
                logs.append(f"section {sections[index-1]}-{section} extracted")
                logs.append(time.strftime("%Y%m%d-%H%M%S"))

                # extract pose from student section video using extract_pose_from_video function
                try:
                    # student_json_filename, student_json_angle_filename = extract_pose_from_video(predict_folder,
                    #                                                                              student_section_filename)

                    student_json_filename, student_json_angle_filename = extract_pose_from_video_v2(
                        predict_folder,  student_section_filename)

                    student_angle_data = json.load(
                        open(student_json_angle_filename))
                    student_major_angle_data = []
                    top_left = []
                    top_right = []
                    for angle_data in student_angle_data:
                        if len(angle_data) != 10:
                            continue
                        top_left.append(angle_data[8])
                        top_right.append(angle_data[9])
                    student_major_angle_data.append(top_left)
                    student_major_angle_data.append(top_right)

                    # read data in student_json_angle_filename and visualize data using draw_section_angle

                    # add current time to logs with caption "pose extracted"
                    logs.append(
                        f"student pose {sections[index-1]}-{section} extracted")
                    logs.append(time.strftime("%Y%m%d-%H%M%S"))
                except Exception as e:
                    print(f"Error extracting pose from student section: {e}")

                teacher_section_time = section - teacher_start_time
                teacher_previous_section_time = sections[index -
                                                         1] - teacher_start_time
                if teacher_previous_section_time < 0:
                    print("teacher_previous_section_time < 0")
                    continue
                if teacher_section_time > teacher_end_time:
                    print("teacher_section_time > teacher_end_time")
                    break
                # generate new name for teacher section file
                teacher_section_filename = os.path.join(
                    predict_folder, f"teacher_{sections[index-1]}_{section}.mp4")
                # extract section from input video using ffmpeg
                crop_video(teacher_crop_filename, teacher_previous_section_time,
                           teacher_section_time, teacher_section_filename)
                # extract pose from teacher section video using extract_pose_from_video function
                try:
                    teacher_json_filename, teacher_json_angle_filename = extract_pose_from_video_v2(predict_folder,
                                                                                                 teacher_section_filename)

                    teacher_pose_data = json.load(
                        open(teacher_json_angle_filename))
                    teacher_major_angle_data = []
                    top_left = []
                    top_right = []
                    for angle_data in teacher_pose_data:
                        if len(angle_data) != 10:
                            continue
                        top_left.append(angle_data[8])
                        top_right.append(angle_data[9])
                    teacher_major_angle_data.append(top_left)
                    teacher_major_angle_data.append(top_right)

                    logs.append(
                        f"teacher pose {sections[index-1]}-{section} extracted")
                    logs.append(time.strftime("%Y%m%d-%H%M%S"))
                except Exception as e:
                    print(f"Error extracting pose from teacher section: {e}")

                # * model test
                try:
                    # predicted_score = model_test(
                    #     model_path, student_json_filename, teacher_json_filename)
                    # # add current time to logs with caption "model tested"
                    logs.append(f"model tested {sections[index-1]}-{section}")
                    logs.append(time.strftime("%Y%m%d-%H%M%S"))

                    # ai_score = predicted_score.item()
                    ai_score = 0
                    final_score += ai_score
                    number_of_sections += 1
                    section_name = f"{round(student_previous_section_time, 2)} - {round(student_section_time, 2)}"
                    result = f"{section_name} : " + \
                        f"{round(ai_score, 2)}/5"
                    final_score_text.append(result)

                    timing_score, velocity_score, accuracy_score, = calculate_dance_evaluation(
                        student_json_angle_filename, teacher_json_angle_filename)
                    logs.append(
                        f"calculated tested {sections[index-1]}-{section}")
                    logs.append(time.strftime("%Y%m%d-%H%M%S"))
                    final_score_timing += timing_score
                    final_score_velocity += velocity_score
                    final_score_accuracy += accuracy_score

                    result = f"{ai_score}, timing score: " + \
                        f"{round(timing_score, 2)}/5, " + "velocity score: " + \
                        f"{round(velocity_score, 2)}/5, " + "accuracy score: " + \
                        f"{round(accuracy_score, 2)}/5"
                    calculation_final_score_text.append(result)
                    # insert section_score to database
                    
                    section_score_id = db.insert_section_score(
                        score_id, f"{sections[index-1]}_{section}", ai_score, timing_score, velocity_score, accuracy_score)
                    
                    print(score_id, f"{sections[index-1]}_{section}", ai_score, timing_score, velocity_score, accuracy_score)
                except Exception as e:
                    print(f"Error in model testing: {e}") 
                
                # break              
            
        average_score = final_score / number_of_sections
        average_score_timing = final_score_timing / number_of_sections
        average_score_velocity = final_score_velocity / number_of_sections
        average_score_accuracy = final_score_accuracy / number_of_sections

        final_score_text.append(
            {"Average score: " + f"{round(average_score, 2)}/5"})

        calculation_final_score_text.append(
            {"Average timing score: " + f"{round(average_score_timing, 2)}/5" + ", average velocity score: " + f"{round(average_score_velocity, 2)}/5" + ", average accuracy score: " + f"{round(average_score_accuracy, 2)}/5"})

        for index, log in enumerate(logs):
            db.insert_log(predict_id, output_id, index, log)

        db.update_score(score_id, average_score, average_score_timing,
                        average_score_velocity, average_score_accuracy)

        db.close()
        return {"predictions": final_score_text, "logs": logs, "calculation_predictions": calculation_final_score_text}

    except Exception as e:
        print(f"Error: {e}")
        return {"predictions": final_score_text, "logs": logs, "calculation_predictions": calculation_final_score_text}
        # Handle the error here
        # You can log the error, display an error message, or take any other appropriate action
        # Remember to handle the error gracefully and provide meaningful feedback to the user


@app.post("/single_pose_estimation")
def single_pose_estimation(image_file: UploadFile = File(...)):
    # Save uploaded image to images folder
    image_path = "pose.jpg"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
    # Read image with OpenCV
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True, enable_segmentation=True, model_complexity=0, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start_time = time.time() * 1000
    results = mp_pose.process(image)
    end_time = time.time() * 1000
    time_taken = end_time - start_time
    return {"time_taken": time_taken}


@app.post("/dance_assessment_json")
def dance_assessment_json(student_pose_json: UploadFile = File(...),
                          student_angle_json: UploadFile = File(...),
                          teacher_pose_json: UploadFile = File(...),
                          teacher_angle_json: UploadFile = File(...), ):
    student_json_filename = generate_new_name("student_pose.json")
    teacher_json_filename = generate_new_name("teacher_pose.json")

    student_angle_json_filename = generate_new_name("student_angle.json")
    teacher_angle_json_filename = generate_new_name("teacher_angle.json")

    # save file
    with open(student_json_filename, "wb") as buffer:
        shutil.copyfileobj(student_pose_json.file, buffer)
    with open(teacher_json_filename, "wb") as buffer:
        shutil.copyfileobj(teacher_pose_json.file, buffer)

    with open(student_angle_json_filename, "wb") as buffer:
        shutil.copyfileobj(student_angle_json.file, buffer)
    with open(teacher_angle_json_filename, "wb") as buffer:
        shutil.copyfileobj(teacher_angle_json.file, buffer)

    ai_model = "model_weight_1"
    model_path = ai_model + '.pt'
    # * model test
    ai_score = ai_dance_evaluation(
        model_path, student_json_filename, teacher_json_filename)
    timing_score, velocity_score, accuracy_score, = calculate_dance_evaluation(
        student_angle_json_filename, teacher_angle_json_filename)
    return {"ai_score": ai_score, "timing_score": timing_score, "velocity_score": velocity_score, "accuracy_score": accuracy_score}


@app.post("/dance_assessment")
def dance_assessment(ai_model, student: UploadFile = File(...), teacher: UploadFile = File(...), ):
    model_path = ai_model + '.pt'
    # generate new name for video file
    student_video_filename = generate_new_name("student.mp4")
    teacher_video_filename = generate_new_name("teacher.mp4")

    # save file
    with open(student_video_filename, "wb") as buffer:
        shutil.copyfileobj(student.file, buffer)
    with open(teacher_video_filename, "wb") as buffer:
        shutil.copyfileobj(teacher.file, buffer)

    student_video_60fps_filename = generate_new_name("student_60fps.mp4")
    teacher_video_60fps_filename = generate_new_name("teacher_60fps.mp4")

    # convert video to 60 fps
    if not preprocess_video(student_video_filename, student_video_60fps_filename):
        return {"predictions": []}
    if not preprocess_video(teacher_video_filename, teacher_video_60fps_filename):
        return {"predictions": []}

    # extract music from input video using ffmpeg
    student_music_filename = generate_new_name("student_music.mp3")
    teacher_music_filename = generate_new_name("teacher_music.mp3")
    os.system(
        f"ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {student_video_60fps_filename} -f mp3 -ab 192000 -vn {student_music_filename}")
    os.system(
        f"ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i {teacher_video_60fps_filename} -f mp3 -ab 192000 -vn {teacher_music_filename}")
    # add current time to logs with caption "music extracted"

    # synchronize music using dynamic time warping
    student_matching_frames, teacher_matching_frames = music_synchronize(
        student_music_filename, teacher_music_filename)

    teacher_start_time = teacher_matching_frames[-1]
    teacher_end_time = teacher_matching_frames[1]
    student_start_time = student_matching_frames[-1]
    student_end_time = student_matching_frames[1]

    student_crop_filename = generate_new_name("student_crop.mp4")
    teacher_crop_filename = generate_new_name("teacher_crop.mp4")

    # crop video using ffmpeg based on start and end time
    crop_video(student_video_60fps_filename, student_start_time,
               student_end_time, student_crop_filename)
    crop_video(teacher_video_60fps_filename, teacher_start_time,
               teacher_end_time, teacher_crop_filename)

    # load sections_same_length.json file and read "sections" data as an array of integers
    with open("sections_same_length.json", "r") as f:
        sections = json.load(f)["sections"]
        # split the video into sections using ffmpeg
        for i, section in enumerate(sections):
            if i == 0:
                continue

            student_section_time = section - student_start_time
            student_previous_section_time = sections[i-1] - student_start_time
            if student_previous_section_time < 0:
                continue
            if student_section_time > student_end_time:
                break

            # generate new name for student section file
            student_section_filename = generate_new_name(
                f"student_{sections[i-1]}_{section}.mp4")
            # extract section from input video using ffmpeg
            crop_video(student_crop_filename, student_previous_section_time,
                       student_section_time, student_section_filename)

            # extract pose from student section video using extract_pose_from_video function
            student_json_filename, student_json_angle_filename = extract_pose_from_video(
                student_section_filename)

            teacher_section_time = section - teacher_start_time
            teacher_previous_section_time = sections[i-1] - teacher_start_time
            if teacher_previous_section_time < 0:
                print("teacher_previous_section_time < 0")
                continue
            if teacher_section_time > teacher_end_time:
                print("teacher_section_time > teacher_end_time")
                break
            # generate new name for teacher section file
            teacher_section_filename = generate_new_name(
                f"teacher_{sections[i-1]}_{section}.mp4")
            # extract section from input video using ffmpeg
            crop_video(teacher_crop_filename, teacher_previous_section_time,
                       teacher_section_time, teacher_section_filename)
            # extract pose from teacher section video using extract_pose_from_video function
            teacher_json_filename, teacher_json_angle_filename = extract_pose_from_video(
                teacher_section_filename)

            # * model test
            ai_score = ai_dance_evaluation(
                model_path, student_json_filename, teacher_json_filename)

            timing_score, velocity_score, accuracy_score, = calculate_dance_evaluation(
                student_json_angle_filename, teacher_json_angle_filename)

            return {
                "predicted_score": ai_score,
                "timing_score": timing_score,
                "velocity_score": velocity_score,
                "accuracy_score": accuracy_score, }


def ai_dance_evaluation(model_path, student_json_filename, teacher_json_filename):
    # * model test
    predicted_score = model_test(
        model_path, student_json_filename, teacher_json_filename)
    ai_score = round(predicted_score.item(), 2)
    return ai_score


def calculate_dance_evaluation(student_json_angle_filename, teacher_json_angle_filename):
    # Read data from student_json_angle_filename and teacher_json_angle_filename
    student_angle_data = json.load(open(student_json_angle_filename))
    teacher_pose_data = json.load(open(teacher_json_angle_filename))
    student_angle_dictionary = {}
    teacher_angle_dictionary = {}
    for index, angleJointName in enumerate(angleJointNames):
        for std_pose, teacher_pose in zip(student_angle_data, teacher_pose_data):
            if len(std_pose) != 10 or len(teacher_pose) != 10:
                continue
            if angleJointName not in student_angle_dictionary:
                student_angle_dictionary[angleJointName] = []
            student_angle_dictionary[angleJointName].append(std_pose[index])

            if angleJointName not in teacher_angle_dictionary:
                teacher_angle_dictionary[angleJointName] = []
            teacher_angle_dictionary[angleJointName].append(
                teacher_pose[index])

        # student_angle_dictionary[angleJointName] = [
        #     pose[index] for pose in student_angle_data]
        # teacher_angle_dictionary[angleJointName] = [
        #     pose[index] for pose in teacher_pose_data]

    joint_index_timing_count = dict(dict_joint_index_count)
    joint_index_velocity_count = dict(dict_joint_index_count)
    joint_index_accuracy_count = dict(dict_joint_index_count)
    joint_timing_score = dict(dict_joint_index_score)
    joint_velocity_score = dict(dict_joint_index_score)
    joint_accuracy_score = dict(dict_joint_index_score)

    student_json_angle_folder = os.path.basename(student_json_angle_filename).replace('student_', '').replace('_angle.json', '')        
    # timing_acc_path = os.path.join(os.path.dirname(
    #     student_json_angle_filename), f'{student_json_angle_folder}_accuracy')
    # if os.path.exists(timing_acc_path):
    #     shutil.rmtree(timing_acc_path)
    # os.mkdir(timing_acc_path)

    # timing_path = os.path.join(os.path.dirname(
    #     student_json_angle_filename), f'{student_json_angle_folder}_time')
    # if os.path.exists(timing_path):
    #     shutil.rmtree(timing_path)
    # os.mkdir(timing_path)

    # velocity_path = os.path.join(os.path.dirname(
    #     student_json_angle_filename), f'{student_json_angle_folder}_velocity')
    # if os.path.exists(velocity_path):
    #     shutil.rmtree(velocity_path)
    # os.mkdir(velocity_path)

    for index, angleJointName in enumerate(angleJointNames):
        student_velocity = calculate_instantaneous_velocity_v2(
            student_angle_dictionary[angleJointName])
        teacher_velocity = calculate_instantaneous_velocity_v2(
            teacher_angle_dictionary[angleJointName])
        
        try:
            path_velocity, total_velocity_level, non_dupe_index_velocity, level_from_dtw_distance_velocity = start_3D_data_evaluation_fastdtw(
                'velocity', teacher_velocity, student_velocity, None, angleJointName)
        except Exception as e:
            print("Error line 1385")
            print(e)
        try:
            path_accuracy, total_accuracy_level, non_dupe_index_accuracy, level_from_dtw_distance_accuracy = start_3D_data_evaluation_fastdtw(
                'accuracy', teacher_angle_dictionary[angleJointName], student_angle_dictionary[angleJointName],
                None, angleJointName)
        except Exception as e:
            print("Error line 1391")
            print(e)
            
        if total_accuracy_level == 0:
            total_velocity_level = 0
            
        # # Save student_velocity and teacher_velocity to csv file
        # with open(student_json_angle_filename.replace('.json', f'{angleJointName}_velocity.csv').replace('student_',''), 'w', newline='') as file:
        #     file.write("student_velocity, teacher_velocity, velocity_score\n")
        #     for student_velocity, teacher_velocity, velocity_score, velocity_level_score in zip(student_velocity, teacher_velocity, non_dupe_index_velocity, level_from_dtw_distance_velocity):
        #         file.write(f"{student_velocity}, {teacher_velocity}, {velocity_score}, {velocity_level_score}\n")
        
        index_a, index_b = zip(*path_accuracy)
        # # Save student_velocity and teacher_velocity to csv file
        # with open(student_json_angle_filename.replace('.json', f'{angleJointName}_accuracy.csv').replace('student_',''), 'w', newline='') as file:
        #     file.write("student_index, teacher_index, accuracy_score\n")
        #     for index_student, index_teacher, accuracy_score, accuracy_level_score in zip(index_b, index_a, non_dupe_index_accuracy, level_from_dtw_distance_accuracy):
        #         file.write(f"{student_angle_dictionary[angleJointName][index_student]}, {teacher_angle_dictionary[angleJointName][index_teacher]}, {accuracy_score}, {accuracy_level_score}\n")
        
        # Calculate timing level
        # # index1 = index[i+1] - index[i]
        index1 = [index_a[i+1] - index_a[i] for i in range(len(index_a)-1)]
        index1 = [0] + index1
        # # index2 = index[i+1] - index[i]
        index2 = [index_b[i+1] - index_b[i] for i in range(len(index_b)-1)]
        index2 = [0] + index2
        # Subtract index2 from index1
        dtw_distance_timing_list = [abs(a - b) for a, b in zip(index2, index1)]

        # Save index_a, index_b and dtw_distance_timing_list to csv file
        # with open(os.path.join(timing_path, f'{angleJointName}_timing.csv'), 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["index_a", "index_b", "index1",
        #                     "index2", "dtw_distance_timing_list"])
        #     writer.writerows(zip(index_a, index_b, index1,
        #                      index2, dtw_distance_timing_list))

        timing_levels = [check_level_timing(
            abs(x)) for x in dtw_distance_timing_list]
        total_timing_level = sum(
            abs(number) for number in timing_levels)
        
        # Save student_velocity and teacher_velocity to csv file
        # with open(student_json_angle_filename.replace('.json', f'{angleJointName}_timing.csv').replace('student_',''), 'w', newline='') as file:
        #     file.write("timing_score, timing_level_score\n")
        #     for timing_score, timing_level_score in zip(dtw_distance_timing_list, timing_levels):
        #         file.write(f"{timing_score}, {timing_level_score}\n")
        
        
        if total_accuracy_level == 0:
            total_timing_level = 0

        joint_index_accuracy_count[index] = len(
            non_dupe_index_accuracy)
        joint_accuracy_score[index] += total_accuracy_level
        joint_index_velocity_count[index] = len(
            non_dupe_index_velocity)
        joint_velocity_score[index] += total_velocity_level
        joint_index_timing_count[index] = len(
            dtw_distance_timing_list)
        joint_timing_score[index] += total_timing_level

    timing_score = 0
    velocity_score = 0
    accuracy_score = 0
    for index, angleJointName in enumerate(angleJointNames):
        timing_score += (round(joint_timing_score[index] /
                               joint_index_timing_count[index], 2)) * angleJointWeight[index]
        velocity_score += round(joint_velocity_score[index] /
                                joint_index_velocity_count[index], 2) * angleJointWeight[index]
        accuracy_score += round(joint_accuracy_score[index] /
                                joint_index_accuracy_count[index], 2) * angleJointWeight[index]
    
    timing_score = timing_score / 25 * 5
    velocity_score = velocity_score / 25 * 5
    accuracy_score = accuracy_score / 25 * 5

    return timing_score, velocity_score, accuracy_score


@app.get("/pose/{type}/{predict_id}/{section_name}")
async def get_pose(type: str, predict_id: str, section_name: str):
    pose_path = os.path.join(
        "Output", f"{predict_id}", f"{type}_{section_name}_angle.json")
    pose_angle_data = json.load(open(pose_path))
    pose_angle_dictionary = {}
    for index, angleJointName in enumerate(angleJointNames):
        if angleJointName not in pose_angle_dictionary:
            pose_angle_dictionary[angleJointName] = []
        for pose in pose_angle_data:
            if len(pose) != 10:
                continue
            pose_angle_dictionary[angleJointName].append(pose[index])    
    return pose_angle_dictionary


@app.post("/delete_section")
def delete_section_by_id(section_id: int):
    db = DatabaseManager('students.db')
    db.delete_section_score_by_id(section_id)
    db.close()
    return "success"

@app.get("/poses/{predict_id}/{section_name}")
async def get_poses(predict_id: str, section_name: str):
    student_pose_path = os.path.join(
        "Output", f"{predict_id}", f"student_{section_name}_angle.json")
    student_pose_angle_data = json.load(open(student_pose_path))
    student_pose_angle_dictionary = {}
    for index, angleJointName in enumerate(angleJointNames):
        if angleJointName not in student_pose_angle_dictionary:
            student_pose_angle_dictionary[angleJointName] = []
        for pose in student_pose_angle_data:
            if len(pose) != 10:
                continue
            student_pose_angle_dictionary[angleJointName].append(pose[index])    
            
    teacher_pose_path = os.path.join(
        "Output", f"{predict_id}", f"teacher_{section_name}_angle.json")
    teacherpose_angle_data = json.load(open(teacher_pose_path))
    teacher_pose_angle_dictionary = {}
    for index, angleJointName in enumerate(angleJointNames):
        if angleJointName not in teacher_pose_angle_dictionary:
            teacher_pose_angle_dictionary[angleJointName] = []
        for pose in teacherpose_angle_data:
            if len(pose) != 10:
                continue
            teacher_pose_angle_dictionary[angleJointName].append(pose[index])    
    return {'student': student_pose_angle_dictionary, 'teacher': teacher_pose_angle_dictionary}