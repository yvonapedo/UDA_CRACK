from keras.losses import binary_crossentropy
import keras.backend as K

epsilon = 1e-5
smooth = 1

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 
#
# def tversky(y_true, y_pred):
#     y_true_pos = K.flatten(y_true.cpu().detach().numpy() )
#     y_pred_pos = K.flatten(y_pred.cpu().detach().numpy() )
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1-y_pred_pos))
#     false_pos = K.sum((1-y_true_pos)*y_pred_pos)
#     alpha = 0.7
#     return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


import torch
import torch.nn.functional as F


def tversky(y_true, y_pred, alpha=0.7, smooth=1e-6):
    # Flatten the tensors
    y_true_pos = y_true.view(-1).cpu()
    y_pred_pos = y_pred.view(-1).cpu()

    # Compute true positives, false negatives, and false positives
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)

    # Compute the Tversky index
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    return tversky_index

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

# def focal_tversky(y_true,y_pred):
#     pt_1 = tversky(y_true, y_pred)
#     gamma = 0.75
#     out = K.pow((1-pt_1), gamma)
#     return out.cpu()


def focal_tversky(y_true, y_pred, alpha=0.7, gamma=0.75, smooth=1e-6):
    # Compute the Tversky index
    pt_1 = tversky(y_true, y_pred, alpha, smooth)

    # Compute Focal Tversky Loss
    focal_tversky_loss = torch.pow((1 - pt_1), gamma)

    return focal_tversky_loss