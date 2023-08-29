import os
import random
from typing import Sequence

import cv2
import torchvision.transforms.functional as TF

import numpy as np
import torch
from numpy import transpose
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image

from canny import CannyDetector
from dataset.ReadLmdb import lmdbDataset
from recognizer.TransOCR.utils import converter



class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MyTrainDataSet(Dataset):
    def __init__(self, path,scale=2):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = path

        self.scale = scale
        self.dataset = lmdbDataset(root=path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        scale = self.scale
        # print(inputPath)
        dataset = self.dataset
        # print(len(dataset))
        # print(dataset[3])
        index = index % len(self.dataset)
        # print(index)
        targetImage = dataset[index][0]
        inputImage = dataset[index][1]
        labels = dataset[index][2]
        randombir = random.uniform(0.8, 1.6)
        targetImage = TF.adjust_brightness(targetImage, randombir)
        inputImage = TF.adjust_brightness(inputImage, randombir)


        # print(text_input)

        inputImage_Tensor = transforms.ToTensor()(inputImage)

        targetImage_Tensor = transforms.ToTensor()(targetImage)


        # label = torch.tensor(label)
        # print(label)
        input_ =  inputImage_Tensor
        target = targetImage_Tensor
        return input_, target, labels


class MyTestDataSet(Dataset):
    def __init__(self, path,scale=2):
        super(MyTestDataSet, self).__init__()

        self.inputPath = path

        self.scale = scale
        self.dataset = lmdbDataset(root=path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        scale = self.scale
        # print(inputPath)
        dataset = self.dataset
        # print(len(dataset))
        # print(dataset[3])
        index = index % len(self.dataset)
        # print(index)
        targetImage = dataset[index][0]
        inputImage = dataset[index][1]
        labels = dataset[index][2]


        # print(text_input)

        inputImage_Tensor = transforms.ToTensor()(inputImage)

        targetImage_Tensor = transforms.ToTensor()(targetImage)


        # label = torch.tensor(label)
        # print(label)
        input_ =  inputImage_Tensor
        target = targetImage_Tensor
        return input_, target, labels