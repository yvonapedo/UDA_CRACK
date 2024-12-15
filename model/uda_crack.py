import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

import model.dino as DINO

from model.dcn import UpBlock, SE_Block, ConvTransform

url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
backbone = DINO.ViTFeat(url, feat_dim=768, vit_arch="base", vit_feat="k", patch_size=8)
from model.dcn import DeformConv2d, ResidualBlock

class UDA_CRACK(nn.Module):
    def __init__(self, patch_size, train_size):
        super(UDA_CRACK, self).__init__()
        self.sMod = backbone
        self.flatten = nn.Unflatten(2, torch.Size([train_size // patch_size, train_size // patch_size]))
        self.tMod = nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.downSample = nn.Upsample(scale_factor=1 / patch_size, mode='bicubic')
        self.out_size = train_size
        self.sigmoid = nn.Sigmoid()
        self.first_bound = ResidualBlock(in_channels=3, out_channels=32)
        self.boundary = nn.Sequential(DeformConv2d(32, 32, modulation=True),
                                      nn.BatchNorm2d(32), nn.ReLU(),
                                      nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False))
        self.se = SE_Block(c=32, r=4)
        self.ConvT = ConvTransform()
        self.second_bound = ResidualBlock(in_channels=1, out_channels=32)

    def forward(self, imgsC1, imgsC2= None, imgTS= None, imgST= None,bound_2=None ):


        if not imgsC2 == None:
            # feature extraction from source domain
            feat_t = self.sMod(imgsC1)
            pred_t = self.flatten(feat_t)
            pred_t =  self.tMod(pred_t)
            feat_t = feat_t.view(1, 768, 48, 48)

            feat_s = self.sMod(imgsC2)
            feat_s = feat_s.view(1, 768, 48, 48)

            # # boundary enhancement moudel(BEM)
            b_2 =  self.first_bound(imgsC2)
            B_out = self.boundary(b_2)

            B_outt = self.second_bound(B_out)

            B_outt = self.ConvT(B_outt)
            B_outt = self.downSample(B_outt)
            featb_s = feat_s+ B_outt
            pred_s = self.tMod(featb_s)

            feat_ts = self.sMod(imgTS)
            pred_ts = self.flatten(feat_ts)
            pred_ts =  self.tMod(pred_ts)
            feat_ts = feat_ts.view(1, 768, 48, 48)

            feat_st = self.sMod(imgST)
            feat_st = feat_st.view(1, 768, 48, 48)

            # # boundary enhancement moudel(BEM)
            b_2st = self.first_bound(imgST)
            B_outst = self.boundary(b_2st)
            B_outtst = self.second_bound(B_outst)
            B_outtst = self.ConvT(B_outtst)
            B_outtst = self.downSample(B_outtst)

            featb_st = feat_st + B_outtst
            pred_st = self.tMod(featb_st)

            pred_t = F.interpolate(pred_t, size=self.out_size, mode='bicubic', align_corners=False)
            pred_s = F.interpolate(pred_s, size=self.out_size, mode='bicubic', align_corners=False)
            pred_ts = F.interpolate(pred_ts, size=self.out_size, mode='bicubic', align_corners=False)
            pred_st = F.interpolate(pred_st, size=self.out_size, mode='bicubic', align_corners=False)

            return   self.sigmoid(pred_t), feat_t, self.sigmoid(pred_s), feat_s, self.sigmoid(pred_ts), feat_ts, self.sigmoid(pred_st), feat_st, B_out, B_outst
        else:
            with torch.no_grad():
                feats = self.sMod(imgsC1)
            feats = self.flatten(feats)
            y = self.tMod(feats)
            y = F.interpolate(y, size=self.out_size, mode='bicubic', align_corners=False)

            return y