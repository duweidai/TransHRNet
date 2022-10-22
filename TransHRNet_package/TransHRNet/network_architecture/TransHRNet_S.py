# ------------------------------------------------------------------------
# TransHRNet:  3D Medical Image Segmentation using Parallel Transformers
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import CNNBackbone
from neural_network import SegmentationNetwork
from EffTrans_Block.effTrans_block_layers import effTrans_layers
from EffTrans_Block.position_encoding import build_position_encoding


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):

    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg,activation_cfg,kernel_size,stride=(1, 1, 1),padding=(0, 0, 0),dilation=(1, 1, 1),bias=False,weight_std=False):
        super(Conv3dBlock,self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out

class U_ResTran3D(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False):
        super(U_ResTran3D, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.upsamplex2 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        self.upsamplex222 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear')

        self.transposeconv_stage2 = nn.ConvTranspose3d(384, 384, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(384, 192, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(192, 64, kernel_size=(2,2,2), stride=(2,2,2), bias=False)

        self.stage2_de = ResBlock(384, 384, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(192, 192, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)

        self.ds2_cls_conv = nn.Conv3d(384, self.MODEL_NUM_CLASSES, kernel_size=1)  # num_classes 14
        self.ds1_cls_conv = nn.Conv3d(192, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds0_cls_conv = nn.Conv3d(64, self.MODEL_NUM_CLASSES, kernel_size=1)
        
        self.conv_s2 = nn.Conv3d(384, 384, kernel_size=(2,2,2), stride=(2,2,2), bias=False)

        self.cls_conv = nn.Conv3d(64, self.MODEL_NUM_CLASSES, kernel_size=1)
 
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.backbone = CNNBackbone.Backbone(depth=9, norm_cfg=norm_cfg, weight_std=weight_std)
        total = sum([param.nelement() for param in self.backbone.parameters()])
        print('  + Number of Backbone Params: %.2f(e6)' % (total / 1e6))

        self.position_embed_23 = build_position_encoding(mode='v2', hidden_dim=384)
        self.position_embed_1 = build_position_encoding(mode='v2', hidden_dim=192)
        self.Delight_Trans  = effTrans_layers(384)
        
        total = sum([param.nelement() for param in self.Delight_Trans.parameters()])
        print('  + Number of Delight_Trans Params: %.2f(e6)' % (total / 1e6))
        
        # total = sum([param.nelement() for param in self.encoder_Detrans_23.parameters()])
        # print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

    def posi_mask(self, x):

        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl == 1:     # 1th, 2th, 3th  layers exchange
                x_fea.append(fea)
                x_posemb.append(self.position_embed_1(fea))    
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())
            if lvl > 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed_23(fea))  
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())
                
        return x_fea, masks, x_posemb
        
    
    def unify_shape(self, x_trans_lv2, x_trans_lv3):
        # untify to level 3, and fusion them
        x_rs_lv2 = x_trans_lv2.permute(1, 2, 0).view(-1, 384, 12, 24, 24)
        x_rs_lv3 = x_trans_lv3.permute(1, 2, 0).view(-1, 384, 6, 12, 12)         
        
        x_lv2_3 = self.conv_s2(x_rs_lv2)
        x_3_fused = x_lv2_3 + x_rs_lv3       

        # untify to level 2, and fusion them
        x_lv3_2 = self.upsamplex222(x_rs_lv3)
        x_2_fused = x_lv3_2 + x_rs_lv2
        
        return x_2_fused, x_3_fused                         
                                               
        
    def forward(self, inputs):
        # ################################# TransHRNet
        x_convs = self.backbone(inputs) 
                                                  
        x_fea, masks, x_posemb = self.posi_mask(x_convs)    

        x_lv2 = x_fea[1] + x_posemb[1]             
        x_lv2 = x_lv2.flatten(2).permute(2, 0, 1)  
        lv2_mask = masks[1].flatten(1)               
        y_lv2 = self.Delight_Trans(x_lv2, lv2_mask)  
       
        x_lv3 = x_fea[2] + x_posemb[2]             
        x_lv3 = x_lv3.flatten(2).permute(2, 0, 1)  
        lv3_mask = masks[2].flatten(1)               
        y_lv3 = self.Delight_Trans(x_lv3, lv3_mask)  
        
        # fuse first result
        x_lv2_fused, x_lv3_fused = self.unify_shape(y_lv2, y_lv3)  
        
        x_lv2_fused = x_lv2_fused.flatten(2).permute(2, 0, 1)  
        y_lv2_fused = self.Delight_Trans(x_lv2_fused, lv2_mask)  
       
        x_lv3_fused = x_lv3_fused.flatten(2).permute(2, 0, 1)  
        y_lv3_fused = self.Delight_Trans(x_lv3_fused, lv3_mask)  
        
        # fuse SECOND result
        x_lv2_fused_2, x_lv3_fused_2 = self.unify_shape(y_lv2_fused, y_lv3_fused)  
        
        x_lv2_fused_2 = x_lv2_fused_2.flatten(2).permute(2, 0, 1)  
        y_lv2_fused_2 = self.Delight_Trans(x_lv2_fused_2, lv2_mask)  
       
        x_lv3_fused_2 = x_lv3_fused_2.flatten(2).permute(2, 0, 1)  
        y_lv3_fused_2 = self.Delight_Trans(x_lv3_fused_2, lv3_mask)  
               
        # decoder
        x = self.transposeconv_stage2(y_lv3_fused_2.permute(1, 2, 0).view(x_convs[-1].shape))     
        skip2 = y_lv2_fused_2.permute(1, 2, 0).view(x_convs[-2].shape) 
                
        x = x + skip2
        x = self.stage2_de(x)    
        ds2 = self.ds2_cls_conv(x)  
        
        x = self.transposeconv_stage1(x)   

        skip1 = x_convs[-3]               
        x = x + skip1
        x = self.stage1_de(x)              
        ds1 = self.ds1_cls_conv(x)         
        
        x = self.transposeconv_stage0(x)   
        skip0 = x_convs[-4]                
        x = x + skip0
        x = self.stage0_de(x)              
        ds0 = self.ds0_cls_conv(x)         

        result = self.upsamplex2(x)        
        result = self.cls_conv(result)     

        return [result, ds0, ds1, ds2]
        

class TransHRNet_v2(SegmentationNetwork):
    """
    ResTran-3D Unet
    """
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False, deep_supervision=False):
        super().__init__()
        self.do_ds = False
        self.U_ResTran3D = U_ResTran3D(norm_cfg, activation_cfg, img_size, num_classes, weight_std) # U_ResTran3D

        if weight_std==False:
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = Conv3d_wd
        if norm_cfg=='BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg=='SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg=='GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg=='IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x):
        seg_output = self.U_ResTran3D(x)        
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]
