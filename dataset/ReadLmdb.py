import os

import six
import sys
import lmdb
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms

class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None):

        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'Error %d' % index
        index += 1

        with self.env.begin(write=False) as txn:
            HR = 'image_HR-%09d' % index
            # print('img_key:', img_key)
            hr_imgbuf = txn.get(HR.encode())
            hr_buf = six.BytesIO()
            hr_buf.write(hr_imgbuf)
            hr_buf.seek(0)
            try:
                hr = Image.open(hr_buf)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            LR = 'image_LR-%09d' % index
            # print('img_key:', img_key)
            lr_imgbuf = txn.get(LR.encode())
            lr_buf = six.BytesIO()
            lr_buf.write(lr_imgbuf)
            lr_buf.seek(0)
            try:
                lr = Image.open(lr_buf)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8')).strip()
            # print(label)

            if len(label) <= 0:
                return self[index + 1]

            label = label.lower()

            if self.transform is not None:
                lr = self.transform(lr)
                hr = self.transform(hr)

        return hr,lr, label

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):

        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):

        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

if __name__ == '__main__':
    dataset = lmdbDataset(root=r'F:\DataSets\baidu\my_eval\eval_lmdb')
    print(f'Length of dataset {len(dataset)}')
    hr_imagepath = r'F:\DataSets\baidu\my_eval\read_hr'
    lr_imagepath = r'F:\DataSets\baidu\my_eval\read_lr'
    # print(dataset[2])

    for index in range(len(dataset)):
        hr , lr , label = dataset[index]
        print(index, label)
        with open(r"F:\DataSets\baidu\my_eval\read_label.txt", 'a', encoding="utf-8") as f:
            f.write(label+'\n')
        hr.save(os.path.join(hr_imagepath, 'eval'+str(index).zfill(4)+'.jpg'))
        lr.save(os.path.join(lr_imagepath, 'eval' + str(index).zfill(4) + '.jpg'))
