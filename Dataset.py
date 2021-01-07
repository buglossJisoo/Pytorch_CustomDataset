from __future__ import print_function, division
import torch
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # 반응형 모드

class Dataset(Dataset):
    # download, read data 하는 부분
    def __init__(self,root_dir, csv_file, transform=None):
        super(Dataset, self).__init__()

        self.root_dir = root_dir
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform


    # index에 해당하는 data 넘기는 부분
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[index,0])
        image = io.imread(img_name)

        landmarks = self.landmarks_frame.iloc[index, 1:]
        landmarks = np.array([landmarks])
        # x,y 좌표라서 [68,2] 로 만들어줌
        landmarks = landmarks.astype('float').reshape(-1,2)

        sample = {'image': image, 'landmarks': landmarks}

        return sample

    # data의 size를 넘겨주는 부분
    def __len__(self):
        return len(self.landmarks_frame)

if __name__ == "__main__":
    def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        """ 랜드마크(landmark)와 이미지를 보여줍니다. """
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        plt.pause(0.001)  # 갱신이 되도록 잠시 멈춥니다.


    face_dataset = Dataset(csv_file='faces/face_landmarks.csv',
                                            root_dir='faces/')

    fig = plt.figure()
    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break




