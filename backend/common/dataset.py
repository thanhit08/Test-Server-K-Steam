# pose & music mel specturm as dataset


from torch.utils.data import Dataset 
import numpy as np 
import torch 
import pandas as pd 
import os 
import json 


class CustomPoseMusicDataset(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform=None):
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'music_file_name','score'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform = transform
        # self.data_dir = data_dir

    def read_data(self, pose_data_path, music_data_path):
        pose_data = json.load(open(pose_data_path))
        
        numpy_3d_arrays = [[joint for joint in pose]for pose in pose_data]
        np_pose_ret = np.array(numpy_3d_arrays)
        # print(f"pose data shape : {np_pose_ret.shape}")
        np_pose_ret = np_pose_ret[:self.bar_length]
        # print(f"pose data shape : {np_pose_ret.shape}")

        np_music_ret = np.load(music_data_path)
        # print(f"music data shape : {np_music_ret.shape}")
        np_music_ret = np_music_ret[:,:self.bar_length]
        # print(f"music data shape : {np_music_ret.shape}")
        return np_pose_ret, np_music_ret
        

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        pose_data_path = self.data_labels.iloc[idx, 0]
        music_data_path = self.data_labels.iloc[idx, 1]
        # print(pose_data_path)
        image, music = self.read_data(pose_data_path, music_data_path)
        label = self.data_labels.iloc[idx, 2]
        image = torch.tensor(image, dtype=torch.float32)
        music = torch.tensor(music, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        # label = self.target_transform(label)
        return image, music, label

class CustomPoseMusicDataset2D(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform_pos=None, transform_music=None,
                 normalize=False,
                 pos_min=None, pos_max=None,
                 music_min=None, music_max=None):
        super(CustomPoseMusicDataset2D, self).__init__()
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'music_file_name','score', 'part'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform_pos = transform_pos
        self.transform_music= transform_music
        self.normalize=normalize
        self.pos_min=pos_min 
        self.pos_max = pos_max 
        self.music_min=music_min 
        self.music_max=music_max 

        # self.data_dir = data_dir

    def read_data(self, pose_data_path, music_data_path):
        pose_data = json.load(open(pose_data_path))
        
        numpy_3d_arrays = [[joint for joint in pose]for pose in pose_data]
        np_pose_ret = np.array(numpy_3d_arrays)
        # print(f"pose data shape : {np_pose_ret.shape}")
        np_pose_ret = np_pose_ret[:self.bar_length]
        # print(f"pose data shape : {np_pose_ret.shape}")

        np_music_ret = np.load(music_data_path)

        # print(f"music data shape : {np_music_ret.shape}")
        np_music_ret = np_music_ret[:,:self.bar_length]
        # print(f"music data shape : {np_music_ret.shape}")
        np_music_ret = np.reshape(np_music_ret, (1, *np_music_ret.shape))
        return np_pose_ret, np_music_ret
        

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        pose_data_path = self.data_labels.iloc[idx, 0]
        music_data_path = self.data_labels.iloc[idx, 1]
        # print(pose_data_path)
        image, music = self.read_data(pose_data_path, music_data_path)
        label = self.data_labels.iloc[idx, 2] 
        part = self.data_labels.iloc[idx, 3]
        # if self.transform_pos:
        #     image = self.transform_pos(image)
        # if self.transform_music:
        #     music = self.transform_music(music)

        
        image = torch.tensor(image, dtype=torch.float32)
        music = torch.tensor(music, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.normalize:
            image = (image - self.pos_min) / (self.pos_max - self.pos_min)
            music = (music - self.music_min) / (self.music_max - self.music_min)
            label = label /5. 

        return image, music, label, part

class CustomPosePoseDataset2D(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform_pos=None, transform_music=None,
                 normalize=False,
                 pos_min=None, pos_max=None
                 ):
        super(CustomPosePoseDataset2D, self).__init__()
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'pose_techer_file_name','score', 'part'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform_pos = transform_pos
        self.transform_music= transform_music
        self.normalize=normalize
        self.pos_min=pos_min 
        self.pos_max = pos_max 


    def read_data(self, pose_data_path, pose_teacher_data_path):
        pose_data = json.load(open(pose_data_path))
        
        numpy_3d_arrays_student = [[joint for joint in pose]for pose in pose_data]
        np_pose_student_ret = np.array(numpy_3d_arrays_student)
        # print(f"pose data shape : {np_pose_ret.shape}")
        np_pose_student_ret = np_pose_student_ret[:self.bar_length]
        # print(f"pose data shape : {np_pose_ret.shape}")

        pose_teacher_data = json.load(open(pose_teacher_data_path))
        numpy_3d_arrays_teacher = [[joint for joint in pose]for pose in pose_teacher_data]
        np_pose_teacher_ret = np.array(numpy_3d_arrays_teacher)
        # print(f"pose data shape : {np_pose_ret.shape}")
        np_pose_teacher_ret = np_pose_teacher_ret[:self.bar_length]

        return np_pose_student_ret, np_pose_teacher_ret
        

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        pose_student_data_path = self.data_labels.iloc[idx, 0]
        pose_teacher_data_path = self.data_labels.iloc[idx, 1]
        # print(pose_data_path)
        image_student, image_teacher = self.read_data(pose_student_data_path, pose_teacher_data_path)
        label = self.data_labels.iloc[idx, 2] 
        part = self.data_labels.iloc[idx, 3]
        # if self.transform_pos:
        #     image = self.transform_pos(image)
        # if self.transform_music:
        #     music = self.transform_music(music)

        
        image_student = torch.tensor(image_student, dtype=torch.float32)
        image_teacher = torch.tensor(image_teacher, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.normalize:
            image_student = (image_student - self.pos_min) / (self.pos_max - self.pos_min)
            image_teacher = (image_teacher - self.pos_min) / (self.pos_max - self.pos_min)
            # label = label /5. 

        return image_student, image_teacher, label, part

class CustomPosePoseDataset2Dxy(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform_pos=None, transform_music=None,
                 normalize=False,
                 pos_min=None, pos_max=None
                 ):
        super(CustomPosePoseDataset2Dxy, self).__init__()
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'pose_techer_file_name','score', 'part'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform_pos = transform_pos
        self.transform_music= transform_music
        self.normalize=normalize
        self.pos_min=pos_min 
        self.pos_max = pos_max 


    def read_data(self, pose_data_path, pose_teacher_data_path):
        pose_data = json.load(open(pose_data_path))
        
        numpy_3d_arrays_student = [[joint[:2] for joint in pose]for pose in pose_data]
        np_pose_student_ret = np.array(numpy_3d_arrays_student)
        # print(f">>> pose data shape : {np_pose_student_ret.shape}")
        np_pose_student_ret = np_pose_student_ret[:self.bar_length]
        # print(f"pose data shape : {np_pose_ret.shape}")

        pose_teacher_data = json.load(open(pose_teacher_data_path))
        numpy_3d_arrays_teacher = [[joint[:2] for joint in pose]for pose in pose_teacher_data]
        np_pose_teacher_ret = np.array(numpy_3d_arrays_teacher)
        # print(f"pose data shape : {np_pose_ret.shape}")
        np_pose_teacher_ret = np_pose_teacher_ret[:self.bar_length]

        return np_pose_student_ret, np_pose_teacher_ret
        

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        pose_student_data_path = self.data_labels.iloc[idx, 0]
        pose_teacher_data_path = self.data_labels.iloc[idx, 1]
        score = self.data_labels.iloc[idx, 2]
        part_info = self.data_labels.iloc[idx, 3]
        print("=============================================================")
        print(pose_student_data_path)
        print(pose_teacher_data_path)
        print(score)
        print(part_info)
        print("=============================================================")
        image_student, image_teacher = self.read_data(pose_student_data_path, pose_teacher_data_path)
        label = self.data_labels.iloc[idx, 2] 
        part = self.data_labels.iloc[idx, 3]
        # if self.transform_pos:
        #     image = self.transform_pos(image)
        # if self.transform_music:
        #     music = self.transform_music(music)

        
        image_student = torch.tensor(image_student, dtype=torch.float32)
        image_teacher = torch.tensor(image_teacher, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.normalize:
            image_student = (image_student - self.pos_min) / (self.pos_max - self.pos_min)
            image_teacher = (image_teacher - self.pos_min) / (self.pos_max - self.pos_min)
            # label = label /5. 

        return image_student, image_teacher, label, part

class CustomPosePoseDataset2Dxy_debug(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform_pos=None, transform_music=None,
                 normalize=False,
                 pos_min=None, pos_max=None
                 ):
        super(CustomPosePoseDataset2Dxy_debug, self).__init__()
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'pose_techer_file_name','score', 'part'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform_pos = transform_pos
        self.transform_music= transform_music
        self.normalize=normalize
        self.pos_min=pos_min 
        self.pos_max = pos_max 


    def read_data(self, pose_data_path, pose_teacher_data_path):
        pose_data = json.load(open(pose_data_path))
        
        # time
        # keypoints
        # x, y, z
        # numpy_3d_arrays_student = [[joint[:2] for joint in pose]for pose in pose_data]
        numpy_3d_arrays_student = []
        for pose in pose_data:
            temp = []
            for idx, joint in enumerate(pose):
                if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    temp.append(joint[:2])
                else :
                    temp.append([0., 0.])
            numpy_3d_arrays_student.append(temp)
        
        print(numpy_3d_arrays_student)


        np_pose_student_ret = np.array(numpy_3d_arrays_student)
        # print(f">>> pose data shape : {np_pose_student_ret.shape}")
        np_pose_student_ret = np_pose_student_ret[:self.bar_length]
        # print(f"pose data shape : {np_pose_ret.shape}")

        pose_teacher_data = json.load(open(pose_teacher_data_path))
        # numpy_3d_arrays_teacher = [[joint[:2] for joint in pose]for pose in pose_teacher_data] 
        numpy_3d_arrays_teacher = []
        for pose in pose_teacher_data[::2]:
        # for pose in pose_teacher_data:
            temp = []
            for idx, joint in enumerate(pose):
                if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    temp.append(joint[:2])
                else :
                    temp.append([0., 0.])
            numpy_3d_arrays_teacher.append(temp)
        np_pose_teacher_ret = np.array(numpy_3d_arrays_teacher)
        # print(f"pose data shape : {np_pose_ret.shape}")
        np_pose_teacher_ret = np_pose_teacher_ret[:self.bar_length]

        return np_pose_student_ret, np_pose_teacher_ret
        

    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        pose_student_data_path = self.data_labels.iloc[idx, 0]
        pose_teacher_data_path = self.data_labels.iloc[idx, 1]
        score = self.data_labels.iloc[idx, 2]
        part_info = self.data_labels.iloc[idx, 3]
        # print("=============================================================")
        # print(pose_student_data_path)
        # print(pose_teacher_data_path)
        # print(score)
        # print(part_info)
        # print("=============================================================")
        image_student, image_teacher = self.read_data(pose_student_data_path, pose_teacher_data_path)
        label = self.data_labels.iloc[idx, 2] 
        part = self.data_labels.iloc[idx, 3]
        # if self.transform_pos:
        #     image = self.transform_pos(image)
        # if self.transform_music:
        #     music = self.transform_music(music)

        
        image_student = torch.tensor(image_student, dtype=torch.float32)
        image_teacher = torch.tensor(image_teacher, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.normalize:
            image_student = (image_student - self.pos_min) / (self.pos_max - self.pos_min)
            image_teacher = (image_teacher - self.pos_min) / (self.pos_max - self.pos_min)
            # label = label /5. 

        return image_student, image_teacher, label, part, pose_student_data_path, pose_teacher_data_path, part_info

# convert json to numpy 
class CustomPosePoseDataset2Dxy_convert_numpy(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform_pos=None, transform_music=None,
                 normalize=False,
                 pos_min=None, pos_max=None
                 ):
        super(CustomPosePoseDataset2Dxy_convert_numpy, self).__init__()
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'pose_techer_file_name','score', 'part'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform_pos = transform_pos
        self.transform_music= transform_music
        self.normalize=normalize
        self.pos_min=pos_min 
        self.pos_max = pos_max 


    def read_data(self, pose_data_path, pose_teacher_data_path):
        print(f"student path: {pose_data_path}")
        pose_data = json.load(open(pose_data_path))
        
        # time
        # keypoints
        # x, y, z
        # numpy_3d_arrays_student = [[joint[:2] for joint in pose]for pose in pose_data]
        numpy_3d_arrays_student = []
        for pose in pose_data:
            temp = []
            for idx, joint in enumerate(pose):
                if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    temp.append(joint[:2])
                else :
                    temp.append([0., 0.])
            numpy_3d_arrays_student.append(temp)
        
        # print(numpy_3d_arrays_student)


        np_pose_student_ret = np.array(numpy_3d_arrays_student)
        # print(f">>> pose data shape : {np_pose_student_ret.shape}")
        np_pose_student_ret = np_pose_student_ret[:self.bar_length]
        # print(f"pose data shape : {np_pose_ret.shape}")

        print(f"teahcer path: {pose_teacher_data_path}")
        pose_teacher_data = json.load(open(pose_teacher_data_path))
        # numpy_3d_arrays_teacher = [[joint[:2] for joint in pose]for pose in pose_teacher_data] 
        numpy_3d_arrays_teacher = []
        for pose in pose_teacher_data [::2]:
            temp = []
            for idx, joint in enumerate(pose):
                if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    temp.append(joint[:2])
                else :
                    temp.append([0., 0.])
            numpy_3d_arrays_teacher.append(temp)
        np_pose_teacher_ret = np.array(numpy_3d_arrays_teacher)
        # print(f"pose data shape : {np_pose_ret.shape}")
        np_pose_teacher_ret = np_pose_teacher_ret[:self.bar_length]

        return np_pose_student_ret, np_pose_teacher_ret
        

    def __len__(self):
        return len(self.data_labels)

    def convert (self, idx):
        pose_student_data_path = self.data_labels.iloc[idx, 0]
        pose_teacher_data_path = self.data_labels.iloc[idx, 1]
        score = self.data_labels.iloc[idx, 2]
        part_info = self.data_labels.iloc[idx, 3]
        # print("=============================================================")
        # print(pose_student_data_path)
        # print(pose_teacher_data_path)
        # print(score)
        # print(part_info)
        # print("=============================================================")
        image_student, image_teacher = self.read_data(pose_student_data_path, pose_teacher_data_path)
        label = self.data_labels.iloc[idx, 2] 
        part = self.data_labels.iloc[idx, 3]
        # if self.transform_pos:
        #     image = self.transform_pos(image)
        # if self.transform_music:
        #     music = self.transform_music(music)

        
        image_student = torch.tensor(image_student, dtype=torch.float32)
        image_teacher = torch.tensor(image_teacher, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.normalize:
            image_student = (image_student - self.pos_min) / (self.pos_max - self.pos_min)
            image_teacher = (image_teacher - self.pos_min) / (self.pos_max - self.pos_min)
            # label = label /5. 
        
        save_path = os.path.dirname(pose_student_data_path) + "/save_data.npy"
        # if os.path.exists(save_path) == False:
        np.save(save_path, image_student)
        # else:
        #     print("already exists")
        print(f"student_path : {save_path}")

        save_path = os.path.dirname(pose_teacher_data_path) + "/save_data.npy"
    # if os.path.exists(save_path) == False:
        np.save(save_path, image_teacher)
        # else :
        #     print("already exists")
        print(f"teacher_path : {save_path}")


        return image_student, image_teacher, label, part, pose_student_data_path, pose_teacher_data_path, part_info
        pass 

# convert without ommit keypoints 1 ~ 10
class CustomPosePoseDataset2Dxy_convert_numpy_with_keypoint(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform_pos=None, transform_music=None,
                 normalize=False,
                 pos_min=None, pos_max=None
                 ):
        super(CustomPosePoseDataset2Dxy_convert_numpy_with_keypoint, self).__init__()
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'pose_techer_file_name','score', 'part'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform_pos = transform_pos
        self.transform_music= transform_music
        self.normalize=normalize
        self.pos_min=pos_min 
        self.pos_max = pos_max 


    def read_data(self, pose_data_path, pose_teacher_data_path):
        pose_data = json.load(open(pose_data_path))
        
        # time
        # keypoints
        # x, y, z
        numpy_3d_arrays_student = [[joint[:2] for joint in pose]for pose in pose_data]
        # numpy_3d_arrays_student = []
        # for pose in pose_data:
        #     temp = []
        #     for idx, joint in enumerate(pose):
        #         if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        #             temp.append(joint[:2])
        #         else :
        #             temp.append([0., 0.])
        #     numpy_3d_arrays_student.append(temp)
        
        # print(numpy_3d_arrays_student)


        np_pose_student_ret = np.array(numpy_3d_arrays_student)
        # print(f">>> pose data shape : {np_pose_student_ret.shape}")
        np_pose_student_ret = np_pose_student_ret[:self.bar_length]
        # print(f"pose data shape : {np_pose_ret.shape}")

        pose_teacher_data = json.load(open(pose_teacher_data_path))
        # numpy_3d_arrays_teacher = [[joint[:2] for joint in pose]for pose in pose_teacher_data] 
        numpy_3d_arrays_teacher = []
        for pose in pose_teacher_data [::2]:
            temp = []
            for idx, joint in enumerate(pose):
                temp.append(joint[:2])
            numpy_3d_arrays_teacher.append(temp)
        np_pose_teacher_ret = np.array(numpy_3d_arrays_teacher)
        # print(f"pose data shape : {np_pose_ret.shape}")
        np_pose_teacher_ret = np_pose_teacher_ret[:self.bar_length]

        return np_pose_student_ret, np_pose_teacher_ret
        

    def __len__(self):
        return len(self.data_labels)

    def convert (self, idx):
        pose_student_data_path = self.data_labels.iloc[idx, 0]
        pose_teacher_data_path = self.data_labels.iloc[idx, 1]
        score = self.data_labels.iloc[idx, 2]
        part_info = self.data_labels.iloc[idx, 3]
        # print("=============================================================")
        # print(pose_student_data_path)
        # print(pose_teacher_data_path)
        # print(score)
        # print(part_info)
        # print("=============================================================")
        image_student, image_teacher = self.read_data(pose_student_data_path, pose_teacher_data_path)
        label = self.data_labels.iloc[idx, 2] 
        part = self.data_labels.iloc[idx, 3]
        # if self.transform_pos:
        #     image = self.transform_pos(image)
        # if self.transform_music:
        #     music = self.transform_music(music)

        
        image_student = torch.tensor(image_student, dtype=torch.float32)
        image_teacher = torch.tensor(image_teacher, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.normalize:
            image_student = (image_student - self.pos_min) / (self.pos_max - self.pos_min)
            image_teacher = (image_teacher - self.pos_min) / (self.pos_max - self.pos_min)
            # label = label /5. 
        
        save_path = os.path.dirname(pose_student_data_path) + "/save_data_with_keypoint.npy"
        # if os.path.exists(save_path) == False:
        np.save(save_path, image_student)
        # else:
        #     print("already exists")
        print(f"student_path : {save_path}")

        save_path = os.path.dirname(pose_teacher_data_path) + "/save_data_with_keypoint.npy"
    # if os.path.exists(save_path) == False:
        np.save(save_path, image_teacher)
        # else :
        #     print("already exists")
        print(f"teacher_path : {save_path}")


        return image_student, image_teacher, label, part, pose_student_data_path, pose_teacher_data_path, part_info
        pass 
# read numpy data
class CustomPosePoseDataset2Dxy_numpy(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform_pos=None, transform_music=None,
                 normalize=False,
                 pos_min=None, pos_max=None
                 ):
        super(CustomPosePoseDataset2Dxy_numpy, self).__init__()
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'pose_techer_file_name','score', 'part'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform_pos = transform_pos
        self.transform_music= transform_music
        self.normalize=normalize
        self.pos_min=pos_min 
        self.pos_max = pos_max 


    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        pose_student_data_path = self.data_labels.iloc[idx, 0]
        pose_teacher_data_path = self.data_labels.iloc[idx, 1]

        label = self.data_labels.iloc[idx, 2] 
        part = self.data_labels.iloc[idx, 3]

        save_path = os.path.dirname(pose_student_data_path) + "/save_data.npy"
        image_student = np.load(save_path)

        save_path = os.path.dirname(pose_teacher_data_path) + "/save_data.npy"
        image_teacher = np.load(save_path)

        part_info = self.data_labels.iloc[idx, 3]

        
        image_student = torch.tensor(image_student, dtype=torch.float32)
        image_teacher = torch.tensor(image_teacher, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.normalize:
            image_student = (image_student - self.pos_min) / (self.pos_max - self.pos_min)
            image_teacher = (image_teacher - self.pos_min) / (self.pos_max - self.pos_min)
            # label = label /5. 

        return image_student, image_teacher, label, part, pose_student_data_path, pose_teacher_data_path, part_info

class CustomPosePoseDataset2Dxy_numpy_with_keypoint(Dataset):
    def __init__(self, annotations_file, bar_length=1200, transform_pos=None, transform_music=None,
                 normalize=False,
                 pos_min=None, pos_max=None
                 ):
        super(CustomPosePoseDataset2Dxy_numpy_with_keypoint, self).__init__()
        # annotations_file 
        self.data_labels = pd.read_csv(annotations_file,sep='\t', names=['pose_file_name', 'pose_techer_file_name','score', 'part'], header=None, skiprows=1)
        self.bar_length = bar_length # frames
        self.transform_pos = transform_pos
        self.transform_music= transform_music
        self.normalize=normalize
        self.pos_min=pos_min 
        self.pos_max = pos_max 


    def __len__(self):
        return len(self.data_labels)

    def __getitem__(self, idx):
        pose_student_data_path = self.data_labels.iloc[idx, 0]
        pose_teacher_data_path = self.data_labels.iloc[idx, 1]

        label = self.data_labels.iloc[idx, 2] 
        part = self.data_labels.iloc[idx, 3]

        save_path = os.path.dirname(pose_student_data_path) + "/save_data_with_keypoint.npy"
        image_student = np.load(save_path)

        save_path = os.path.dirname(pose_teacher_data_path) + "/save_data_with_keypoint.npy"
        image_teacher = np.load(save_path)

        part_info = self.data_labels.iloc[idx, 3]

        
        image_student = torch.tensor(image_student, dtype=torch.float32)
        image_teacher = torch.tensor(image_teacher, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        if self.normalize:
            image_student = (image_student - self.pos_min) / (self.pos_max - self.pos_min)
            image_teacher = (image_teacher - self.pos_min) / (self.pos_max - self.pos_min)
            # label = label /5. 

        return image_student, image_teacher, label, part, pose_student_data_path, pose_teacher_data_path, part_info