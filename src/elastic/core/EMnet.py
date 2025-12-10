import numpy as np
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import sys
import inspect
from .unet import Fusion

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2

def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons

class ContextNet2d(nn.Module):
    def __init__(self):
        super(ContextNet2d, self).__init__()

        self.netOne = self.convBlock(1, 8)
        self.netTwo = self.convBlock(8, 16)
        self.netThr = self.convBlock(16, 32)
        self.netFou = self.convBlock(32, 64)
        
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        
    def convBlock(self,inchannel, outchannel):
        return nn.Sequential(
        nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x_2x = self.netOne(x)
        # print('2x: ',x_2x.size())
        x_4x = self.netTwo(x_2x)
        # print('4x: ',x_4x.size())
        x_8x = self.netThr(x_4x)
        # print('8x: ',x_8x.size())
        x_16x = self.netFou(x_8x)
        # print('16x: ',x_16x.size())

        return {'1/4': x_4x, '1/8': x_8x, '1/16': x_16x}


class ExtractNet2d(nn.Module):
    def __init__(self):
        super(ExtractNet2d, self).__init__()

        self.netOne = self.convBlock(1, 8)
        self.netTwo = self.convBlock(8, 16)
        self.netThr = self.convBlock(16, 32)
        self.netFou = self.convBlock(32, 64)
        
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        
    def convBlock(self,inchannel, outchannel):
        return nn.Sequential(
        nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x_2x = self.netOne(x)
        # print('2x: ',x_2x.size())
        x_4x = self.netTwo(x_2x)
        # print('4x: ',x_4x.size())
        x_8x = self.netThr(x_4x)
        # print('8x: ',x_8x.size())
        x_16x = self.netFou(x_8x)
        # print('16x: ',x_16x.size())

        return {'1/2':x_2x,'1/4': x_4x, '1/8': x_8x, '1/16': x_16x}


class BasicMotionEncoder(nn.Module):
    def __init__(self, inchannel):
        super(BasicMotionEncoder, self).__init__()
        # layers for correlation
        self.convc1 = nn.Conv2d(inchannel, 256, 1, padding=0)          # outsize = insize
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)                 # outsize = insize

        # layers for flow
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)           # outsize = insize
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)          # outsize = insize
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)      # outsize = insize

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))      # dimension 192

        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))      # dimension 64

        cor_flo = torch.cat([cor, flo], dim=1)  # dimension 192+64
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1) # 126 + 2
    
class GRU(nn.Module):
    def __init__(self,input_dim,hidden_dim, motion_dim=128):
        super(GRU, self).__init__()
        self.convz = nn.Conv2d(motion_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(motion_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(motion_dim+input_dim, hidden_dim, 3, padding=1)
   
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h
    
class baseGRUBlock(nn.Module):
    def __init__(self, inchannel, hdim, feature_dim, up=2):
        super(baseGRUBlock, self).__init__()
        self.inchannel = inchannel # inchannel 是 cat_fea的通道数
        self.hdim = hdim # hid的通道数
        self.feature_dim = feature_dim # cont的通道数
        
        self.encoder = BasicMotionEncoder(inchannel=self.inchannel)
        self.gru = GRU(input_dim=self.feature_dim, hidden_dim=self.feature_dim//2)
        
        self.conv1 = nn.Conv2d(self.hdim, self.hdim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.hdim, 2, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        
        self.mask = nn.Sequential(
            nn.Conv2d(self.feature_dim//2, self.feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim, up*up*9, 1, padding=0))
        
    def forward(self, cont, hid, cat_fea, flow):
        motion_features = self.encoder(flow, cat_fea)   # 这一步就是把上一层的flow和{cat[f_fea,c_fea]}进行encoder融合  
        cont = torch.cat([cont, motion_features], dim=1)
        hid = self.gru(hid, cont) # input_dim+self.hdim+128(motion_dim) = cont + hid +128(motion_features) 的通道数
        flow_up = self.conv2(self.lrelu(self.conv1(hid)))
        up_mask = .25 * self.mask(hid)
        return flow_up,up_mask

class baseRaft(nn.Module):
    def __init__(self, inchannel, hdim, feature_dim):
        super(baseRaft, self).__init__()
        self.GRU_block = baseGRUBlock(inchannel=inchannel, hdim=hdim, feature_dim=feature_dim)
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/n, W/n, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        up = int((mask.size(1)//9)**0.5)
        mask = mask.view(N, 1, 9, up, up, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(up * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, up*H, up*W)
    
    def forward(self, cont, hid, cat_fea, coords0 ,coords1, flow=None):
        # # # print(' cont, hid, cat_fea, coords0 ,coords1: ', cont.size(), hid.size(), cat_fea.size(), coords0.size() ,coords1.size())
        if flow is not None:
            coords1 = coords1 + flow    
        coords1 = coords1.detach()
        flow = coords1 - coords0
        flow_up,up_mask = self.GRU_block(cont,hid,cat_fea,flow)
        flow_up = self.upsample_flow(flow_up, up_mask)
        # # # print('flow_up: ',flow_up.size())
        return flow_up
        
        

    
class EMNet(nn.Module):
    def __init__(self,flow_multiplier=1.0,use_fusion=True):
        super(EMNet, self).__init__() 
        self.flow_multiplier = flow_multiplier
        self.fusion = Fusion(inputc1=16, inputc2=2)
        self.extraction = ExtractNet2d()

        self.baseRaft_16x = baseRaft(inchannel=128,feature_dim=32+32,hdim=32)
        self.baseRaft_8x = baseRaft(inchannel=64,feature_dim=16+16,hdim=16)
        self.baseRaft_4x = baseRaft(inchannel=32,feature_dim=8+8,hdim=8)
        
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H, W, device=img.device)
        coords1 = coords_grid(N, H, W, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
        
    
    def forward(self, image1, image2, f_fea, c_fea, h_fea):
        m_fea = self.extraction(image2)
        hid_4x, hid_8x, hid_16x = h_fea
        cont_4x, cont_8x, cont_16x = c_fea

        coords0_4, coords1_4 = self.initialize_flow(f_fea['1/4'])
        coords0_8, coords1_8 = self.initialize_flow(f_fea['1/8'])
        coords0_16, coords1_16 = self.initialize_flow(f_fea['1/16'])
        
        

        flow_16x_up = self.baseRaft_16x(cont_16x,hid_16x, torch.cat([f_fea['1/16'], m_fea['1/16']], 1), 
                                        coords0_16,coords1_16,flow=None)# 这里。。。是f_fea和m_fea使用到的地方。。。同时需要注意:cont和hid也是类似f_fea的特征。。。
        flow_8x_up = self.baseRaft_8x(cont_8x,hid_8x, torch.cat([f_fea['1/8'], m_fea['1/8']], 1),
                                        coords0_8,coords1_8,flow=flow_16x_up)
        flow_4x_up = self.baseRaft_4x(cont_4x,hid_4x, torch.cat([f_fea['1/4'], m_fea['1/4']], 1),
                                        coords0_4,coords1_4,flow=flow_8x_up)


        flow = self.fusion(f_fea['1/2'],m_fea['1/2'] ,flow_4x_up) 

        b, c, h, w = image1.shape
        final_flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False) * 2
        return {'flow': final_flow * self.flow_multiplier}
        

         
 
class Framework(nn.Module):
    def __init__(self,args):
        super(Framework, self).__init__()
        self.args = args
        self.cdim = 32
        self.hdim = 32
        self.flow_multiplier = 1.0/args.iters
        
        self.context = ContextNet2d()
        self.extract = ExtractNet2d()
        self.defnet = nn.ModuleList([EMNet(flow_multiplier = self.flow_multiplier) for _ in range(args.iters)])
      

    def forward(self, Img1s, Img2):
        contexts = self.context(Img1s) # 
        f_fea = self.extract(Img1s)

        hid_4x, cont_4x = torch.split(contexts['1/4'], [8,8], dim=1)
        hid_8x, cont_8x = torch.split(contexts['1/8'], [16,16], dim=1)
        hid_16x, cont_16x = torch.split(contexts['1/16'], [32,32], dim=1)
        

        cont = [torch.relu(cont_4x),# 1,16,128,128
                torch.relu(cont_8x),# 1,24,64,64
                torch.relu(cont_16x)]# 1, 64, 32, 32
        hid = [torch.tanh(hid_4x), # b,a1,a2,a3
                torch.tanh(hid_8x),
                torch.tanh(hid_16x)]
        # # print(cont_4x.size())
        # # print(cont_8x.size())
        # # print(cont_16x.size())

        deforms_0 = self.defnet[0](Img1s, Img2, f_fea, cont, hid) # # 这里注意输入的hid应该是上次的输出hid还是初始的hid
        warpImg = flow_warp(Img2, deforms_0['flow'])
        
        Deforms = [deforms_0]
        WarpImgs = [warpImg]
        agg_flow = deforms_0['flow']
        for i in range(self.args.iters - 1):
            deforms = self.defnet[i+1](Img1s, warpImg, f_fea, cont, hid) # 这里注意输入的hid应该是上次的输出hid还是初始的hid
            agg_flow = flow_warp(agg_flow, deforms['flow']) + deforms['flow']
            warpImg = flow_warp(Img2, agg_flow)
        
            Deforms.append(deforms)
            WarpImgs.append(warpImg)
               
        return Deforms,WarpImgs,warpImg,agg_flow
    
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='EMNet', help='name your experiment')
    parser.add_argument('--dataset', type=str, default='brain', help='which dataset to use for training')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=1, help='number of image pairs per batch on single gpu')
    parser.add_argument('--sum_freq', type=int, default=1000)
    parser.add_argument('--val_freq', type=int, default=2000)
    parser.add_argument('--round', type=int, default=20000, help='number of batches per epoch')
    parser.add_argument('--data_path', type=str, default='E:/Registration/Code/TMI2022/Github/Data_MRIBrain/')
    parser.add_argument('--base_path', type=str, default='E:/Registration/Code/TMI2022/Github/')
    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    args.model_path = args.base_path + args.name + '/output/checkpoints_' + args.dataset
    args.eval_path = args.base_path + args.name + '/output/eval_' + args.dataset
    model=Framework(args)
    model = model.to('cuda')
    input1 = torch.randn(1,1,3,512,512).to('cuda')
    input2 = torch.randn(1,1,512,512).to('cuda')
    deforms, warpImgs, warpImg2, agg_flow = model(input1,input2)
    # # print('flow: ',deforms[0]['flow'].size()) # flow:  torch.Size([1, 2, 512, 512])
    # # print('flow: ',deforms[0]['flow'].permute(0,2,3,1).size()) 
