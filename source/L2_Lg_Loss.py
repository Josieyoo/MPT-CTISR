import cv2
from matplotlib import pyplot as plt
from skimage import io


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from IPython import embed
from torchvision import transforms
from skimage.filters import threshold_sauvola
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from torchvision.utils import save_image

def to_gray_tensor(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 1:2, :, :]
    B = tensor[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


class l2_lg_lce(nn.Module):
    def __init__(self, gradient=True, loss_weight=[1, 1e-4, 1e-5]):
        super(l2_lg_lce, self).__init__()
        self.l2 = nn.MSELoss()
        if gradient:
            self.GPLoss = GradientPriorLoss()
        self.gradient = gradient
        self.loss_weight = loss_weight
        self.toTensor = transforms.ToTensor()
        self.crossEntropy = torch.nn.CrossEntropyLoss()



    def forward(self, out_images, target_images,text_pred,text_gt):
        # print("text_pred:", text_pred.shape)
        # print("text_gt:", text_gt.shape)
        # exit(0)
        if self.gradient:
            out_images_mask = torch.zeros(target_images.shape[0],target_images.shape[1],target_images.shape[2],target_images.shape[3])
            # print(out_images_mask.shape)
            # exit(0)
            for index in range(len(out_images)):
                sr_mask = transforms.ToPILImage()(out_images[index])
                # mask.show()
                sr_mask = sr_mask.convert('L')
                thres = np.array(sr_mask).mean()
                sr_mask = sr_mask.point(lambda x: 0 if x > thres else 255)
                # target_image.show()
                sr_mask = self.toTensor(sr_mask)
                out_images_mask[index]=sr_mask
            # print(out_images_mask.shape)
            #
            target_images_mask = torch.zeros(target_images.shape[0],target_images.shape[1],target_images.shape[2],target_images.shape[3])
            for index in range(len(target_images)):
                # print(type(target_images))
                hr_mask = transforms.ToPILImage()(target_images[index])
                # mask.show()
                hr_mask = hr_mask.convert('L')
                thres = np.array(hr_mask).mean()
                hr_mask = hr_mask.point(lambda x: 0 if x > thres else 255)
                # target_image.show()
                hr_mask = self.toTensor(hr_mask)
                target_images_mask[index]=hr_mask


            loss = self.loss_weight[0] * self.l2(out_images, target_images) + \
                   self.loss_weight[1] * self.GPLoss(out_images_mask, target_images_mask)+\
                   self.loss_weight[2] * self.crossEntropy(text_pred, text_gt)
            print("l2:{},lg:{},lce:{}".format(self.loss_weight[0] * self.l2(out_images, target_images),
                                              self.loss_weight[1] * self.GPLoss(out_images_mask, target_images_mask),
                                              self.loss_weight[2] * self.crossEntropy(text_pred, text_gt)))

        else:
            loss = self.loss_weight[0] * self.l2(out_images, target_images)
        return loss


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss()

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        return self.func(map_out, map_target)

    @staticmethod
    def gradient_map(x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad


if __name__ == '__main__':
    im1=Image.open('../tt.jpg')
    im1=transforms.ToTensor()(im1)
    im1=im1.unsqueeze(0)
    im2 = Image.open('../tt1.jpg')
    im2 = transforms.ToTensor()(im2)
    im2 = im2.unsqueeze(0)
    embed()
