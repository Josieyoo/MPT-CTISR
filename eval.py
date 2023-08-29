import sys
import time

import cv2
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import utils
from CARN.carn import CARN_Net
from IMDN.architecture import IMDN_RTC
from torch.autograd import Variable
from MyNettransformerpca import MyNet, weights_init
from bicubic import BICUBIC
from recognizer.CRNN.utils import strLabelConverter,val
from srresnet import SRResNet
from swinir import SwinIR
from util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
from utils import *
from data import *
from datasets import *
from recognizer.TransOCR.model.transocr import Transformer
from recognizer.TransOCR.utils import converter, get_alphabet, tensor2str,val

def my_collate(batch):

    input_ = [item[0] for item in batch]
    target = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    length, text_input, text_gt, string_label = converter(labels)
    input_=torch.stack(input_)
    target=torch.stack(target)
    #
    # print("input_:",input_.shape)
    # print("target:",target.shape)
    # print("text_input:",text_input)


    return input_,target,text_input,length,string_label,text_gt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PathTest = r'F:\DataSets\baidu\my_test\testx4_lmdb'
    resultPathTest = './dataset/resultTest/mvt3+pca+lg-131/'  
    # resultPathTest_lr = './dataset/resultTest/lr/'

    CHECKPOINT = './mvt3+pca+lg-131.pth'
    alpha_file = './benchmark.txt'
    TP_Generator_path = './scene_base.pth'

    eval_record_path = './dataset/eval_record/mvt3+pca+lg-131'
    BATCH_SIZE = 1
    PATCH_SIZE = 2  
    NUM_WORKERS = 0
    IMAGE_SIZE = 32
    SCALE = 4
    N_Blocks = 7
    MYNET = 'mobilevit'


    print('--------------------------------------------------------------')
    if SCALE==2:
        IMAGE_W=96
        IMAGE_H=24

    elif SCALE==3:
        IMAGE_W=64
        IMAGE_H=16

    elif SCALE==4:
        IMAGE_W=48
        IMAGE_H=12
    if MYNET == 'swinir':
        myNet = SwinIR(upscale=SCALE, in_chans=3, img_size=32, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                       mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')

    elif MYNET == 'mobilevit':
        myNet =myNet = MyNet(in_channels=3,n_blocks=N_Blocks,scale=SCALE, triple_clues=True)

    elif MYNET == 'srresnet':
        myNet = SRResNet(large_kernel_size=9,
                         small_kernel_size=3,
                         n_channels=64,
                         n_blocks=16,
                         scaling_factor=SCALE)
    elif MYNET == 'imdn':
        myNet = IMDN_RTC(upscale=SCALE)
    elif MYNET == 'carn':
        myNet = CARN_Net(scale=SCALE)
    elif MYNET == 'bicubic':
        myNet = BICUBIC(scale_factor=SCALE)
    myNet = myNet.to(device)

    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        myNet = nn.DataParallel(myNet, device_ids=device_ids)

    myNet.load_state_dict(torch.load(CHECKPOINT),False)



    
    # ImageNames = os.listdir(PathTest)  
    datasetTest = MyTestDataSet(PathTest, scale=SCALE)
    testLoader = DataLoader(dataset=datasetTest, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,collate_fn=my_collate,
                             num_workers=NUM_WORKERS, pin_memory=False)

    alphabet = get_alphabet(alpha_file)
    nclass = len(alphabet)

    TP_Generator_model = Transformer(nclass=nclass).to(device)
    TP_Generator_model = nn.DataParallel(TP_Generator_model)
    TP_Generator_model.load_state_dict(torch.load(TP_Generator_path))


    myNet.eval()
    
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()
    PSNRs_Y = AverageMeter()
    SSIMs_Y = AverageMeter()

    
    PRED_SRs_ACC = utils.AverageMeter()
    PRED_LRs_ACC = utils.AverageMeter()
    PRED_HRs_ACC = utils.AverageMeter()
    labels_values = []
    PRED_LRs=[]
    PRED_SRs=[]
    PRED_HRs=[]

    with torch.no_grad(): 
        
        timeStart = time.time()  
        for index, (x, y, z, u, v,w) in enumerate(testLoader):
            
            lr_imgs, hr_imgs, labels_value, length_value, string_label, text_gt= Variable(x).to(device), \
                                            Variable(y).to(device), Variable(z).to(device), Variable(u).to(device), v,w
            rec_image = torch.nn.functional.interpolate(lr_imgs, size=(32, 256))

            pred = torch.zeros(BATCH_SIZE, 1).long().cuda()
            image_features = None
            TP_Generator_model.eval()
            for i in range(max(max(length_value),32)):
                length_tmp = torch.zeros(BATCH_SIZE).long().cuda() + i + 1
                result = TP_Generator_model(rec_image, length_tmp, pred, conv_feature=image_features, test=True)
                prediction = result['pred']
                now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
                pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
            # print('result:', result)
            # print('pred:', pred.shape)
            # exit(0)
            label_vecs_final = prediction.unsqueeze(1).permute(0, 3, 1, 2)

            # sr_imgs = myNet(lr_imgs,tp_net=TP_Generator_model, length=length_value)
            sr_imgs = myNet(lr_imgs=lr_imgs,tp=label_vecs_final)
            preds_lr, gt_lr, acc_lr = val(net=TP_Generator_model, image=lr_imgs, batch_size=BATCH_SIZE,
                                alphabet=alphabet, labels=text_gt, length=length_value)

            preds_sr,gt_sr,acc_sr = val(net=TP_Generator_model, image=sr_imgs, batch_size=BATCH_SIZE,
                                alphabet=alphabet, labels=text_gt, length=length_value)

            preds_hr,gt_hr,acc_hr = val(net=TP_Generator_model, image=hr_imgs, batch_size=BATCH_SIZE,
                                alphabet=alphabet, labels=text_gt, length=length_value)




            PRED_LRs_ACC.update(acc_lr, x.size(0))
            PRED_SRs_ACC.update(acc_sr, x.size(0))
            PRED_HRs_ACC.update(acc_hr, x.size(0))


            PRED_LRs.append(preds_lr)
            PRED_SRs.append(preds_sr)
            PRED_HRs.append(preds_hr)
            labels_values.append(gt_hr)

            save_image(sr_imgs, resultPathTest + str(index + 1).zfill(4) + '.jpg')

           
            hr_imgs,sr_imgs = before_calculate(hr_imgs, sr_imgs)
            # print("hr:",hr_imgs.shape)
            # print("sr:",sr_imgs.shape)
            PSNR = calculate_psnr(sr_imgs, hr_imgs, input_order='HWC')
            SSIM = calculate_ssim(sr_imgs, hr_imgs, input_order='HWC')
            PSNR_Y = calculate_psnr(sr_imgs, hr_imgs, input_order='HWC', test_y_channel=True)
            SSIM_Y = calculate_ssim(sr_imgs, hr_imgs, input_order='HWC', test_y_channel=True)

            print("{}  PSNR:{:.3f}".format(str(index+1).zfill(4)+'.jpg', PSNR))
            print("{}  SSIM:{:.3f}".format(str(index+1).zfill(4)+'.jpg', SSIM))
            print("{}  PSNR_Y:{:.3f}".format(str(index+1).zfill(4)+'.jpg', PSNR_Y))
            print("{}  SSIM_Y:{:.3f}".format(str(index+1).zfill(4)+'.jpg', SSIM_Y))

            PSNRs.update(PSNR, lr_imgs.size(0))
            SSIMs.update(SSIM, lr_imgs.size(0))
            PSNRs_Y.update(PSNR_Y, lr_imgs.size(0))
            SSIMs_Y.update(SSIM_Y, lr_imgs.size(0))

        eval_record_path = os.path.join(eval_record_path+'.txt')
        # print('type(labels_values):',type(labels_values[1][]))
        # print('type(eval_preds_lr):',type(eval_preds_lr[1]))

        for i in range(len(PRED_LRs)):
            with open(eval_record_path, 'a', encoding="utf-8") as f:
                f.write(str(PRED_LRs[i]) + '|' + str(PRED_SRs[i]) + '|' + str(PRED_HRs[i]) + '|' + str(labels_values[i]) + '|'+ str(PRED_SRs[i]==labels_values[i])+'\n')


 
    print('PSNR  {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM  {ssims.avg:.3f}'.format(ssims=SSIMs))
    print('PSNR_Y  {psnrs.avg:.3f}'.format(psnrs=PSNRs_Y))
    print('SSIM_Y  {ssims.avg:.3f}'.format(ssims=SSIMs_Y))

    print('eval_total:', len(PRED_LRs) * BATCH_SIZE)
    print('eval_acc_lr:{:.6f}% | eval_acc_sr:{:.6f}% | eval_acc_hr:{:.6f}%'.format(PRED_LRs_ACC.avg,PRED_SRs_ACC.avg,PRED_HRs_ACC.avg))
    print('AVERAGE TIME : {:.3f} 秒'.format((time.time() - timeStart) / len(datasetTest)))



print("\n")