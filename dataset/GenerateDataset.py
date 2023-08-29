import os
import argparse
import cv2

# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument("-k", "--keepdims", help="keep original image dimensions in downsampled images",
                    action="store_true")
parser.add_argument('--hr_img_dir', type=str, default=r'F:\DataSets\baidu\my_test\hr',
                    help='path to high resolution image dir')
parser.add_argument('--lr_img_dir', type=str, default=r'F:\DataSets\baidu\my_test\lr',
                    help='path to desired output dir for downsampled images')
args = parser.parse_args()

hr_image_dir = args.hr_img_dir
lr_image_dir = args.lr_img_dir

print(args.hr_img_dir)
print(args.lr_img_dir)

# create LR image dirs
os.makedirs(lr_image_dir + "/X2", exist_ok=True)
os.makedirs(lr_image_dir + "/X3", exist_ok=True)
os.makedirs(lr_image_dir + "/X4", exist_ok=True)
# os.makedirs(lr_image_dir + "/X6", exist_ok=True)
# os.makedirs(lr_image_dir + "/X8", exist_ok=True)

supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")

# Downsample HR images
for filename in os.listdir(hr_image_dir):
    if not filename.endswith(supported_img_formats):
        continue

    name, ext = os.path.splitext(filename)

    # Read HR image
    hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])

    # Blur with Gaussian kernel of width sigma = 1
    hr_img = cv2.GaussianBlur(hr_img, (0, 0), 1, 1)
    x, y = hr_img[0:2]
    # Downsample image 2x

    lr_image_2x = cv2.resize(hr_img, (int(hr_img_dims[0] / 2), int(hr_img_dims[1] / 2)), fx=1, fy=1,
                             interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
         lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(lr_image_dir + "/X2", filename.split('.')[0] + 'x2' + ext), lr_image_2x)

    # # Downsample image 3x
    lr_img_3x = cv2.resize(hr_img, (int(hr_img_dims[0] / 3), int(hr_img_dims[1] / 3)), fx=1, fy=1,
                            interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_3x = cv2.resize(lr_img_3x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X3", filename.split('.')[0] + 'x3' + ext), lr_img_3x)
    #
    # Downsample image 4x
    lr_img_4x = cv2.resize(hr_img, (int(hr_img_dims[0] / 4), int(hr_img_dims[1] / 4)), fx=1, fy=1,
                           interpolation=cv2.INTER_CUBIC) #双三次插值
    if args.keepdims:
        lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X4", filename.split('.')[0] + 'x4' + ext), lr_img_4x)
    #
    # # Downsample image 6x
    # lr_img_6x = cv2.resize(hr_img, (int(hr_img_dims[0] / 6), int(hr_img_dims[1] / 6)), fx=1, fy=1,
    #                        interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_img_4x = cv2.resize(lr_img_6x, hr_img_dims,
    #                            interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(lr_image_dir + "/X6", filename.split('.')[0] + 'x6' + ext), lr_img_6x)
    #
    # # Downsample image 8x
    # lr_img_6x = cv2.resize(hr_img, (int(hr_img_dims[0] / 8), int(hr_img_dims[1] / 8)), fx=1, fy=1,
    #                        interpolation=cv2.INTER_CUBIC)
    # if args.keepdims:
    #     lr_img_4x = cv2.resize(lr_img_6x, hr_img_dims,
    #                            interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(os.path.join(lr_image_dir + "/X8", filename.split('.')[0] + 'x8' + ext), lr_img_6x)
