import os
import cv2
import shutil
import sys


# 模糊影像检测函数，阈值默认为0.07
def blurImagesDetection(folder_path, thres=0.0007):
    # 新建一个用于存放模糊影像的文件夹
    blurImageDirPath = r'F:\DataSets\baidu\blurs'
    if not os.path.exists(blurImageDirPath):
        os.mkdir(blurImageDirPath)
    # 获取影像文件夹中的影像名列表
    imageNameList = os.listdir(folder_path)
    for imageName in imageNameList:
        # 得到影像路径
        imagePath = os.path.join(folder_path, imageName)
        # 读取影像为灰度图
        img = cv2.imread(imagePath, 0)
        # 缩小影像，加快处理速度
        tiny_img = cv2.resize(img, (400, 300), fx=0, fy=0)
        # 获取影像尺寸
        width, height = tiny_img.shape
        # 计算影像的模糊程度
        blurness = cv2.Laplacian(tiny_img, cv2.CV_64F).var() / (width * height)
        # 如果影像模糊程度小于阈值就将其移动到存放模糊影像的文件夹中
        if blurness < thres:
            print(imageName + "  bulrness:%f   模糊" % (blurness))
            blurImagePath = os.path.join(blurImageDirPath, imageName)
            shutil.move(imagePath, blurImagePath)
        else:
            print(imageName + "  blurness:%f   不模糊" % (blurness))


if __name__ == '__main__':
    # 指定要处理的文件夹路径，sys.argv[1]为第一个参数
    folder_path = r'F:\DataSets\baidu\train_images'
    # 调用函数
    blurImagesDetection(folder_path)
