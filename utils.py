import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
# import keras.backend as K
from torch.autograd import Variable
from torch import sigmoid
from monai.losses import  DiceLoss

def adjust_learning_rate(optimizer, i_iter, NUM_STEPS):
    lr = lr_poly(LEARNING_RATE, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr

def adjust_learning_rate_D(optimizer, i_iter, NUM_STEPS):
    lr = lr_poly(LEARNING_RATE_D, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr


def dice_ce_loss(output, source_label):

    seg_loss = F.binary_cross_entropy_with_logits(output, source_label)
    dc_loss = DiceLoss(output, source_label)
    loss = dc_loss + seg_loss
    return loss

def seg_loss(pred, mask):
    # adaptive weighting mask
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    # wbce = focal_tversky(pred, mask)
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    pred = torch.sigmoid(pred)

    # weighted iou loss function
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (wbce + wiou).mean()

def dice_loss(pred, mask):
    # criterion = DiceLoss( )

    seg_loss = F.binary_cross_entropy_with_logits(pred, mask)
    dc_loss = DiceLoss(pred, mask)
    loss = dc_loss + seg_loss
    return loss

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = torch.flatten(y_true.to(device))
    y_pred_pos = torch.flatten(y_pred.to(device))
    true_pos = sum(y_true_pos * y_pred_pos)
    false_neg = sum(y_true_pos * (1-y_pred_pos))
    false_pos = sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return torch.pow((1-pt_1), gamma)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr

LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
POWER = 0.9



def loss_calc(pred, label, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.float())
    # pred = torch.sigmoid(pred).squeeze(0)
    pred = torch.sigmoid(pred)

    # criterion = CrossEntropy2d()
    # criterion = DiceBCELoss()
    # criterion =  F.binary_cross_entropy()
    # print(label.shape)
    # print(pred.shape)
    # return criterion(pred.cpu(), label.cpu(), weights.cpu())
    return F.binary_cross_entropy(pred.cpu(), label.cpu())

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, NUM_STEPS):
    lr = lr_poly(LEARNING_RATE, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr

def adjust_learning_rate_D(optimizer, i_iter, NUM_STEPS):
    lr = lr_poly(LEARNING_RATE_D, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr

def discrepancy(attx, atty):
    return torch.mean(torch.abs(attx - atty))

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]
        #print(c)
        #d = torch.mean(torch.stack(c))
        #print(d)
        return torch.mean(torch.stack(c))

def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))



def calculate_loss( outputs, labels):
    loss = 0
    loss = cross_entropy_loss_RCF(outputs, labels)
    return loss

def calculate_loss_bce_dice( outputs, labels):
    loss = 0
    outputs = sigmoid(outputs)

    bce_loss = BCELoss(outputs.cpu(), labels.cpu())
    dice_loss = DiceLoss(outputs.cpu(), labels.cpu())
    loss = bce_loss + dice_loss
    return loss



def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    # label2 = label.float()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    prediction = sigmoid(prediction)
    # print(label.shape)
    prediction = np.squeeze(prediction,0) # 删掉一个维度
    # print(prediction.shape)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(),weight = mask, reduce=False)
    # weight = mask
    # return torch.sum(cost)
    # loss = F.binary_cross_entropy(prediction, label2, mask, size_average=True)
    return torch.sum(cost) /(num_positive+num_negative)


import torch
import torch.nn as nn
import torch.nn.functional as F


def BCELoss( predictions, targets):
    bce_loss = F.binary_cross_entropy(predictions, targets)
    return bce_loss


def DiceLoss( predictions, targets):
    smooth = 1.0  # Add a smoothing term to prevent division by zero

        # Flatten the tensors
    predictions = predictions.view(-1)
    targets = targets.view(-1)

        # Compute Dice coefficient
    intersection = (predictions * targets).sum()
    dice_coefficient = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

        # Dice loss
    dice_loss = 1 - dice_coefficient
    return dice_loss

from model.fourier_data.dataset_fourier import augmix
from model.fourier import fourier_image_perturbation
from albumentations.pytorch.transforms import img_to_tensor
from PIL import Image

def fourrier_update(x_source, x_target):
    x_source = fourier_image_perturbation(x_source, x_target, beta=0.006000, ratio=1.0)
    x_source = img_to_tensor(x_source).uniform_(0, 1)
    x_source = x_source.numpy()
    x_source *= 255
    x_source = x_source.astype(np.uint8)
    x_source = x_source.reshape(384, 384, 3)

    # Create a PIL Image from the NumPy array
    x_source = Image.fromarray(x_source)
    x_source = augmix(x_source, alpha=1, mixture_width=3, level=3, im_size=384, mixture_depth=0)

    x_source = (torch.tensor(x_source, dtype=torch.float32)).unsqueeze(0)
    return x_source
 