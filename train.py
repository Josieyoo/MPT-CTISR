import sys
import time
import datetime
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from zhconv import zhconv

import image_loss
import utils
from CARN.carn import CARN_Net
from IMDN.architecture import IMDN_RTC
from L2_Lg_Loss import l2_lg_lce
# from MyNet import MyNet, weights_init
from MyNettransformerpca import MyNet


from data import *
from srresnet import SRResNet
from swinir import SwinIR
from tsrn import TSRN_TL_TRANS
from util_calculate_psnr_ssim import calculate_psnr
from utils import before_calculate
from recognizer.TransOCR.model.transocr import Transformer
from recognizer.TransOCR.utils import converter, get_alphabet, tensor2str,val
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main():

    # random.seed(9)
    # torch.manual_seed(9)
    CHECKPOINT = 'mvt3+pca+lg.pth'
    START_EPOCH = 0
    EPOCH = 199 
    BATCH_SIZE = 32
    MYNET = 'mobilevit'
    TP = 'transocr'


    LEARNING_RATE = 1e-3
    NUM_WORKERS = 0
    SCALE = 4
    N_Blocks = 7

    PathTrain = r'F:\DataSets\baidu\my_train\trainx4_lmdb'
    PathEval = r'F:\DataSets\baidu\my_eval\evalx4_lmdb'

    alpha_file = './benchmark.txt'
    TP_Generator_path = './scene_base.pth'




    best_psnr = 21.898
    best_psnr_epoch = START_EPOCH
    best_acc = 0
    best_acc_epoch = START_EPOCH

    L1_Loss_criterion = nn.L1Loss().to(device) # L1
    # msssim_l1_loss_criterion = MS_SSIM_L1_LOSS().to(device)
    l2_lg_ce_criterion = l2_lg_lce().to(device)
    mse_criteriin = nn.MSELoss().to(device)
    l2_lg_criterion = image_loss.ImageLoss().to(device)


    writer = SummaryWriter()  # python -m tensorboard.main --logdir runs
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
        myNet = SwinIR(upscale=SCALE, in_chans=3, img_size=(IMAGE_W,IMAGE_H), window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        criterion = L1_Loss_criterion

    elif MYNET == 'mobilevit':
        myNet = MyNet(in_channels=3,n_blocks=N_Blocks,scale=SCALE, triple_clues=True)
        criterion = l2_lg_ce_criterion

    elif MYNET == 'srresnet':
        myNet = SRResNet(large_kernel_size=9,
                 small_kernel_size=3,
                 n_channels=64,
                 n_blocks=16,
                 scaling_factor=SCALE)
        criterion = mse_criteriin

    elif MYNET == 'imdn':
        myNet = IMDN_RTC(upscale=SCALE)
        criterion = L1_Loss_criterion

    elif MYNET == 'carn':
        myNet = CARN_Net(scale=SCALE)
        criterion = L1_Loss_criterion

    elif MYNET =='tatt':
        myNet = TSRN_TL_TRANS(width=192, height=48,scale_factor=SCALE)
        criterion = l2_lg_criterion

    myNet = myNet.to(device)


    alphabet = get_alphabet(alpha_file)
    nclass = len(alphabet)
    print(nclass)

    TP_Generator_model = Transformer(nclass=nclass).to(device)
    TP_Generator_model = nn.DataParallel(TP_Generator_model)
    TP_Generator_model.load_state_dict(torch.load(TP_Generator_path))


    optimizer = optim.Adam(myNet.parameters(), lr=LEARNING_RATE)  
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, threshold=0.1,
    #                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-05)
    # scheduler = MultiStepLR(optimizer, milestones=[20,60,80,100,120,140], gamma=0.8)
    # scheduler = CyclicLR(optimizer, base_lr=0.000025, max_lr=0.0001, step_size_up=1500, mode='triangular2',cycle_momentum=False)

    # print(PathTrain)
    datasetTrain = MyTrainDataSet(path = PathTrain, scale=SCALE)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS,collate_fn=my_collate,
                             pin_memory=False)


    datasetEval = MyTestDataSet(path = PathEval, scale=SCALE)
    evalLoader = DataLoader(dataset=datasetEval, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS,collate_fn=my_collate,
                             pin_memory=False)

    
    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        myNet = nn.DataParallel(myNet, device_ids=device_ids)

    print('-------------------------------------------------------------------------------------------------------')
    
    if os.path.exists(CHECKPOINT):
        print(datetime.datetime.now(),"     Continue the last training task!")
        myNet.load_state_dict(torch.load(CHECKPOINT))
    else:
        print(datetime.datetime.now(),"     Start a new training task!")
        print('-------------------------------------------------------------------------------------------------------')

    for epoch in range(START_EPOCH, EPOCH+1):
        print(datetime.datetime.now(), "     Start Epoch %d! " % (epoch + 1))
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        Train_epochLoss = utils.AverageMeter()  
        Eval_epochLoss = utils.AverageMeter()
        timeStart = time.time()

        for index, (x, y, z, u, v, w) in enumerate(iters, 0):
            myNet.zero_grad()
            input_train, target_train, labels_train, length_train, string_labels_train,text_gt_train = Variable(x).to(device), \
                        Variable(y).to(device), Variable(z).to(device), Variable(u).to(device), v, w
            # print("text_gt_train:",text_gt_train)
            # exit(0)
            
            # print("input_train:", input_train)
            # print("target_train:", target_train)
            # print("labels:", labels)
            # print("length:", length)
            # exit(0)
            rec_image = torch.nn.functional.interpolate(input_train, size=(32, 256))
            pred = torch.zeros(BATCH_SIZE, 1).long().cuda()
            image_features = None
            TP_Generator_model.eval()
            for i in range(max(length_train)):
                length_tmp = torch.zeros(BATCH_SIZE).long().cuda() + i + 1
                result = TP_Generator_model(rec_image, length_tmp, pred, conv_feature=image_features, test=True)
                prediction = result['pred']
                now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
                pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
            # print('result:', result)
            # print('pred:', pred.shape)
            # exit(0)
            label_vecs_final = prediction.unsqueeze(1).permute(0, 3, 1, 2)
            # print(label_vecs_final.shape)
            # exit(0)
            output_train = myNet(input_train,tp=label_vecs_final)
            output_train_rec_image = torch.nn.functional.interpolate(output_train, size=(32, 256))
            output_train_result = TP_Generator_model(output_train_rec_image, length_train, labels_train)
            output_train_text_pred = output_train_result['pred']
            # print(output_train_text_pred)
            # print(text_gt_train)
            # exit(0)

            
            trainloss = criterion(output_train, target_train, output_train_text_pred, text_gt_train)
            # trainloss = criterion(output_train, target_train)

            # print(output_train.shape)
            # print(target_train.shape)
            # exit(0)

            # trainloss = criterion(output_train, target_train)
            # print(trainloss)
            
            Train_epochLoss.update(trainloss.item(),x.size(0))

            
            iters.set_description('Training....... Epoch %d / %d, Train Batch Loss: %.6f, learning rate: %.8f'
                                  %(epoch+1, EPOCH+1, trainloss.item(), optimizer.state_dict()['param_groups'][0]['lr']))
            writer.add_scalar('Learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch + 1)

            optimizer.zero_grad()
            trainloss.backward()  
            optimizer.step()  

            del x, y, z, u, output_train
            torch.cuda.empty_cache()
        myNet.eval()
        PSNRs = utils.AverageMeter()
        PREDs = utils.AverageMeter()

        eval_preds_sr = []
        print('evaling.......')
        with torch.no_grad():
            for index, (x, y, z, u, v, w) in enumerate(evalLoader, 0):
                input_value, target_value, labels_value, length_value, string_label_value,text_gt_value= Variable(x).to(device), \
                        Variable(y).to(device), Variable(z).to(device), Variable(u).to(device), v,w

                rec_image = torch.nn.functional.interpolate(input_value, size=(32, 256))
                pred = torch.zeros(BATCH_SIZE, 1).long().cuda()
                image_features = None
                TP_Generator_model.eval()
                for i in range(max(length_value)):
                    length_tmp = torch.zeros(BATCH_SIZE).long().cuda() + i + 1
                    result = TP_Generator_model(rec_image, length_tmp, pred, conv_feature=image_features, test=True)
                    prediction = result['pred']
                    now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
                    pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)

                label_vecs_final = prediction.unsqueeze(1).permute(0, 3, 1, 2)
                # print("label_vecs_final:",label_vecs_final)
                output_value = myNet(input_value, tp=label_vecs_final)
                # preds_sr, gt, acc = val(net=TP_Generator_model, image=output_value, batch_size=BATCH_SIZE,
                #                                      alphabet=alphabet, labels=text_gt_value,length=length_value)
                # eval_preds_sr.append(preds_sr)
                # PREDs.update(acc, x.size(0))
                output_value_rec_image = torch.nn.functional.interpolate(output_value, size=(32, 256))
                output_value_result = TP_Generator_model(output_value_rec_image, length_value, labels_value)
                output_value_text_pred = output_value_result['pred']
                evalloss = criterion(output_value, target_value,output_value_text_pred, text_gt_value)
                # evalloss = criterion(output_value, target_value)
                Eval_epochLoss.update(evalloss.item(), x.size(0))

                for sr_imgs, hr_imgs in zip(output_value, target_value):
                    hr_imgs, sr_imgs = before_calculate(hr_imgs, sr_imgs)
                    psnr = calculate_psnr(sr_imgs, hr_imgs, input_order='HWC')
                    PSNRs.update(psnr, x.size(0))

            if PSNRs.avg > best_psnr:
                best_psnr = PSNRs.avg
                best_psnr_epoch = epoch + 1
                print('saving the best_psnr model.....')
                torch.save(myNet.state_dict(), save_model_psnr + '.pth')

            # if PREDs.avg > best_acc:
            #     best_acc = PREDs.avg
            #     best_acc_epoch = epoch + 1
            #     print('saving the best_acc model.....')
            #     torch.save(myNet.state_dict(), save_model_acc + '.pth')

        print(datetime.datetime.now(),
              '     Epoch {}  Evaluated, Eval Average Loss: {:.6f}, Eval Average PSNR:{:.3f},eval_acc_sr:{:.3f}%'
              .format(epoch + 1, Eval_epochLoss.avg, PSNRs.avg, PREDs.avg))

        writer.add_scalars("loss", {"train_loss_avg": Train_epochLoss.avg, "val_loss_avg": Eval_epochLoss.avg}, epoch)
        writer.add_image('epoch_' + str(epoch + 1)+'_eval_LR',make_grid(input_value[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
        writer.add_image('epoch_' + str(epoch + 1)+'_eval_SR',make_grid(output_value[:4, :3, :, :].cpu(), nrow=4, normalize=True),epoch)
        writer.add_image('epoch_' + str(epoch + 1)+'_eval_HR',make_grid(target_value[:4, :3, :, :].cpu(), nrow=4, normalize=True),epoch)

        timeEnd = time.time()
        print(datetime.datetime.now(),
              "     Epoch {}  Finished, Best PSNR : {:.3f}, Best PSNR Epoch:{}, Time:  {:.4f} s "
              .format(epoch + 1, best_psnr, best_psnr_epoch,  timeEnd - timeStart))
        # scheduler.step()
        print('-------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    main()





