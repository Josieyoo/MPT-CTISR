import os
import time

import lmdb  # install lmdb by "pip install lmdb"
import cv2
import re
from PIL import Image
import numpy as np
import imghdr
import argparse


def init_args():
    args = argparse.ArgumentParser()
    args.add_argument('-hr',
                      '--hr_imagePathList',
                      type=str,
                      help='The directory of the dataset , which contains the images',
                      default=r'F:\DataSets\baidu\my_test\hr')
    args.add_argument('-lr',
                      '--lr_imagePathList',
                      type=str,
                      help='The directory of the dataset , which contains the images',
                      default=r'F:\DataSets\baidu\my_test\lr\X2')
    args.add_argument('-l',
                      '--label_file',
                      type=str,
                      help='The file which contains the paths and the labels of the data set',
                      default=r'F:\DataSets\baidu\my_test\test_label.txt')
    args.add_argument('-s',
                      '--save_dir',
                      type=str
                      , help='The generated mdb file save dir',
                      default=r'F:\DataSets\baidu\my_test\testx2_lmdb')
    args.add_argument('-m',
                      '--map_size',
                      help='map size of lmdb',
                      type=int,
                      default=536870912)
    #   256/4GB

    return args.parse_args()


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    txn = env.begin(write=True)
    for k, v in cache.items():
        txn.put(k.encode(), v)
    txn.commit()


def _is_difficult(word):
    assert isinstance(word, str)
    return not re.match('^[\w]+$', word)


def createDataset(outputPath, hr_imagePathList,lr_imagePathList, labelList, SIZ=20971520, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
    outputPath  : LMDB output path
    imagePathList : list of image path
    labelList   : list of corresponding groundtruth texts
    lexiconList  : (optional) list of lexicon lists
    checkValid  : if true, check the validity of every image
    """
    assert (len(hr_imagePathList) == len(labelList))
    assert (len(lr_imagePathList) == len(labelList))
    nSamples = len(hr_imagePathList)
    # print(nSamples)
    # print('imagePathList:',imagePathList)
    # print('labelList:',labelList)

    env = lmdb.open(outputPath, map_size=SIZ)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        hr_imagePath = hr_imagePathList[i]
        lr_imagePath = lr_imagePathList[i]
        label = labelList[i]
        if len(label) == 0:
            continue
        if not os.path.exists(hr_imagePath):
            print('%s does not exist' % hr_imagePath)
            continue
        with open(hr_imagePath, 'rb') as hrf:
            HR = hrf.read()
        with open(lr_imagePath, 'rb') as lrf:
            LR = lrf.read()
        if checkValid:
            if not checkImageIsValid(HR):
                print('%s is not a valid image' % hr_imagePath)
                continue
            if not checkImageIsValid(LR):
                print('%s is not a valid image' % lr_imagePath)
                continue
        # 数据库中都是二进制数据
        image_hr_Key = 'image_HR-%09d' % cnt
        image_lr_Key = 'image_LR-%09d' % cnt

        labelKey = 'label-%09d' % cnt
        cache[image_hr_Key] = HR
        cache[image_lr_Key] = LR

        cache[labelKey] = label.encode()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    args = init_args()
    imgdata = open(args.label_file, mode='r', encoding='utf-8')
    lines = list(imgdata)
    # print('len(lines):',len(lines))
    hr_imgDir = args.hr_imagePathList
    lr_imgDir = args.lr_imagePathList
    hr_imgPathList = []
    lr_imgPathList = []
    labelList = []
    # SIZ = 0
    for index in range (len(lines)):
        # print("LINE=",line)
        # print('line.split()[0]:',line)
        hr_imgPath = os.path.join(hr_imgDir, 'test'+str(index+1).zfill(4)+'.jpg').replace('\\', '/')
        lr_imgPath = os.path.join(lr_imgDir, 'test' + str(index+1).zfill(4) + 'X2.jpg').replace('\\', '/')
        # print('imgPath:',imgPath)
        hr_imgPathList.append(hr_imgPath)
        lr_imgPathList.append(lr_imgPath)
        labelList.append(lines[index])
        # imgPath = imgPath + line
        # SIZ += os.path.getsize(imgPath)
        # print(imgPath,word)
    # print("SIZ=", SIZ, "ALL=", len(labelList))

    createDataset(args.save_dir, hr_imgPathList,lr_imgPathList, labelList,SIZ=args.map_size)