import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
from datetime import datetime
from data_train import get_loader, test_in_train
from loss_git import prediction_consistency_loss
from utils import AvgMeter, fourrier_update, dice_ce_loss
import cv2
import argparse
import logging
from DaCrack.Discriminator import DomainDiscriminator
from SCST.net import Net
from utils import adjust_learning_rate_D, adjust_learning_rate


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model.uda_crack import UDA_CRACK

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Epoch number')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='Training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='Training dataset size')
parser.add_argument('--patchsize', type=int, default=8, help='Patch number of the vision transformer')
parser.add_argument('--reg_weight', type=float, default=1.0, help='Weighting the regularization term')

opt = parser.parse_args()

# build models
model = UDA_CRACK(opt.patchsize, opt.trainsize)
model = torch.nn.DataParallel(model)
model.to(device)

net_scst_target = Net()
net_scst_target = torch.nn.DataParallel(net_scst_target)
net_scst_target.to(device)

net_scst_source = Net()
net_scst_source = torch.nn.DataParallel(net_scst_source)
net_scst_source.to(device)


D_target = DomainDiscriminator(768, 1)
D_target = torch.nn.DataParallel(D_target)
D_target.to(device)
optimizer_Dt =  torch.optim.Adam(D_target.parameters(), lr=1e-4, betas=(0.9, 0.999))


D_source = DomainDiscriminator(768, 1)
D_source = torch.nn.DataParallel(D_source)
D_source.to(device)
optimizer_Ds =  torch.optim.Adam(D_source.parameters(), lr=1e-4, betas=(0.9, 0.999))

# set optimizer
segModParams, adaModParams = [], []
for name, param in model.named_parameters():
    if 'adaMod' in name:
       # print(name)
        adaModParams.append(param)
    else:
        segModParams.append(param)

optimizer = torch.optim.Adam([{'params':segModParams}, {'params':adaModParams}], opt.lr)
learning_rate_scst = 1e-4
optimizer_scst_target = torch.optim.Adam(net_scst_target.parameters(), lr=learning_rate_scst, betas=(0.9, 0.99))
optimizer_scst_source = torch.optim.Adam(net_scst_source.parameters(), lr=learning_rate_scst, betas=(0.9, 0.99))


# set loss function
seg_loss = torch.nn.BCELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

# set path
#target =style target
target_image_root = r"C:\Users\yvona\Documents\NPU_research\research3\domain adaptation\UDA-CRACK-main\data\from_wkst\test_small\images/"
target_pseudo_gt_root = r"C:\Users\yvona\Documents\NPU_research\research3\domain adaptation\UDA-CRACK-main\data\from_wkst\test_small\masks/"

#source
source_image_root = r"C:\Users\yvona\Documents\NPU_research\research3\domain adaptation\UDA-CRACK-main\data\from_wkst\crack500\image/"
source_pseudo_gt_root = r"C:\Users\yvona\Documents\NPU_research\research3\domain adaptation\UDA-CRACK-main\data\from_wkst\crack500\mask/"

save_path = 'ckpt/new_checkpoints_c500_200/'
if not os.path.exists(save_path): os.makedirs(save_path)

train_loader = get_loader(target_image_root, target_pseudo_gt_root,
                          source_image_root, source_pseudo_gt_root,
                          batchsize=opt.batchsize, trainsize=opt.trainsize)

total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("UDA-CRACK-Train")
logging.info("Config")
logging.info('epoch:{}; lr:{}; batchsize:{}; trainsize:{}; save_path:{}'.
             format(opt.epochs, opt.lr, opt.batchsize, opt.trainsize, save_path))

LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
LAMBDA_ADV_DF = 0.005
LAMBDA_MMD = 2.0

C_source_label = 0
C_target_label = 1
# ----------------------------------------------------------------------------------------------------------------------
best_mae = 1
best_epoch = 0

def TRAIN(train_loader, model, optimizer, epoch, save_path):
    optimizer.param_groups[0]['lr'] = opt.lr * 0.2 ** (epoch - 1)
    optimizer.param_groups[1]['lr'] = opt.lr * 0.2 ** (epoch - 1) * 0.1
    print("curret learning rate of target model: " + str(optimizer.param_groups[0]['lr']))
    print("curret learning rate of ADA module: " + str(optimizer.param_groups[1]['lr']))

    net_scst_target.train()
    net_scst_source.train()
    model.train()
    D_source.train()
    D_target.train()

    total_loss_record = AvgMeter()
    segC2_s_loss_record = AvgMeter()
    segC2_st_loss_record = AvgMeter()
 

    for i, pack in enumerate(train_loader, start=1):
        # optimizer
        optimizer_scst_target.zero_grad()
        optimizer_scst_source.zero_grad()
        optimizer.zero_grad()

        adjust_learning_rate(optimizer, (epoch - 1) * (10000 / opt.batchsize) + i, opt.epochs * (10000 / opt.batchsize))

        optimizer_Ds.zero_grad()
        optimizer_Dt.zero_grad()
        adjust_learning_rate_D(optimizer_Ds, (epoch - 1) * (10000 / opt.batchsize) + i, opt.epochs * (10000 / opt.batchsize))
        adjust_learning_rate_D(optimizer_Dt, (epoch - 1) * (10000 / opt.batchsize) + i, opt.epochs * (10000 / opt.batchsize))
 

        # data
        imgsC1, pgtsC1, imgsC2, pgtsC2, bound_2  = pack
        imgsC1, pgtsC1, imgsC2, pgtsC2,bound_2 = Variable(imgsC1), Variable(pgtsC1), Variable(imgsC2), Variable(pgtsC2), Variable(bound_2)
        imgsC1, pgtsC1, imgsC2, pgtsC2, bound_2   = imgsC1.to(device), pgtsC1.to(device), imgsC2.to(device), pgtsC2.to(device), bound_2.to(device)


        # forward
        loss_c_target, loss_s_target, loss_aug_target, loss_ssm_target, out_target = net_scst_target( imgsC2, imgsC1, pgtsC2)
        loss_c_source, loss_s_source, loss_aug_source, loss_ssm_source, out_source = net_scst_source( imgsC1, imgsC2, pgtsC1)
        imgST = out_target
        imgST = fourrier_update(imgST, imgsC1)

        imgTS = out_source
        imgTS = fourrier_update(imgTS, imgsC2)

        pred_t, feat_t, pred_s, feat_s, pred_ts, feat_ts, pred_st, feat_st, B_out, B_outst = model(imgsC1, imgsC2, imgTS, imgST, bound_2)

        loss_boundary_s = dice_ce_loss(B_out, bound_2)
        loss_boundary_st = dice_ce_loss(B_outst, bound_2)

        
        segC2_s_loss = dice_ce_loss(pred_s, pgtsC2)*0.8 + loss_boundary_s*0.2
        segC2_st_loss = dice_ce_loss(pred_st, pgtsC2)*0.8 + loss_boundary_st*0.2


        consistency_loss_t = prediction_consistency_loss(pred_ts, pred_t)
        consistency_loss_s = prediction_consistency_loss(pred_st, pred_s)
        
        D_out_s = F.sigmoid(D_source(feat_s))
        D_out_ts = F.sigmoid(D_source(feat_ts))
        D_out_t = F.sigmoid(D_target(feat_t))
        D_out_st =F.sigmoid( D_target(feat_st))

        source_label = (F.sigmoid(Variable(torch.FloatTensor(D_out_s.data.size()).fill_(C_source_label)))).to(device)
        target_label = (F.sigmoid(Variable(torch.FloatTensor(D_out_s.data.size()).fill_(C_target_label)))).to(device)
        #
        loss_Ds = (bce_loss(D_out_s, source_label))/ 2
        loss_Dst = (bce_loss(D_out_st, source_label))/ 2
        loss_Ds =loss_Ds +loss_Dst

        loss_Dt = (bce_loss(D_out_t,target_label)) / 2
        loss_Dts = (bce_loss(D_out_ts,target_label)) / 2
        loss_Dt = loss_Dt + loss_Dts

        loss = consistency_loss_t+consistency_loss_s+ LAMBDA_SEG *segC2_s_loss +LAMBDA_SEG *segC2_st_loss +  LAMBDA_ADV_TARGET1 * loss_Ds +   LAMBDA_ADV_TARGET2 * loss_Dt

        loss_scst_t = loss_c_target + loss_s_target + loss_aug_target + loss_ssm_target
        loss_scst_s = loss_c_source  + loss_s_source + loss_aug_source + loss_ssm_source

        # back-propagation
        loss.backward(retain_graph=True)
        loss_scst_t.backward(retain_graph=True)
        loss_scst_s.backward(retain_graph=True)
        loss_Ds.backward(retain_graph=True)
        loss_Dt.backward()

        optimizer_scst_source.step()
        optimizer_scst_target.step()
        optimizer.step()
        optimizer_Ds.step()
        optimizer_Dt.step()

        total_loss_record.update(loss.data, opt.batchsize)
        segC2_s_loss_record.update(segC2_s_loss.data, opt.batchsize)
        segC2_st_loss_record.update(segC2_st_loss.data, opt.batchsize)
  

        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total Loss: {:.4f}, '
                  'Seg Cls2 s Loss: {:.4f} , Seg Cls2 st Loss: {:.4f},  '.format(datetime.now(), epoch, opt.epochs, i, total_step, total_loss_record.show(),
                         segC2_s_loss_record.show(), segC2_st_loss_record.show()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total Loss: {:.4f}, '
                  'Seg Cls1 Loss: {:.4f},  Seg Cls2 Loss: {:.4f} '.format(epoch, opt.epochs, i, total_step, total_loss_record.show(),
                         segC2_s_loss_record.show(), segC2_st_loss_record.show() ))

        if epoch % 1 == 0:
            torch.save(model.state_dict(), save_path + 'UDA-CRACK' + '_%d' % epoch + '.pth')

def TEST(validation_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum_c1, mae_sum_c2 = 0, 0
        for i in range(validation_loader.size):
            image_c1, pgt_c1, name_c1, HH_c1, WW_c1, \
                image_c2, pgt_c2, name_c2, HH_c2, WW_c2 = validation_loader.load_data()

            pgt_c1 = np.asarray(pgt_c1, np.float32)
            pgt_c1 /= (pgt_c1.max() + 1e-8)
            image_c1 = image_c1.to(device)

            pgt_c2 = np.asarray(pgt_c2, np.float32)
            pgt_c2 /= (pgt_c2.max() + 1e-8)
            image_c2 = image_c2.to(device)

            res_c1, _, res_c2, _ = model(image_c1, image_c2)

            res_c1 = F.upsample(res_c1, size=[WW_c1, HH_c1], mode='bilinear', align_corners=False)
            res_c1 = res_c1.sigmoid().data.cpu().numpy().squeeze()
            mae_sum_c1 += np.sum(np.abs(res_c1 - pgt_c1)) * 1.0 / (pgt_c1.shape[0] * pgt_c1.shape[1])

            res_c2 = F.upsample(res_c2, size=[WW_c2, HH_c2], mode='bilinear', align_corners=False)
            res_c2 = res_c2.sigmoid().data.cpu().numpy().squeeze()
            mae_sum_c2 += np.sum(np.abs(res_c2 - pgt_c2)) * 1.0 / (pgt_c2.shape[0] * pgt_c2.shape[1])

        mae = (mae_sum_c1 + mae_sum_c2) / (2 * validation_loader.size)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'UDA-CRACK_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Let's go!")
    for epoch in range(1, (opt.epochs+1)):
        # beta = update_beta_with_epoch( n_epochs, beta_opt, cl_strategy, epoch_ratio, cur_epoch)
        TRAIN(train_loader, model, optimizer, epoch, save_path)
        #TEST(validation_loader, model, epoch, save_path)
    print("Training Done!")