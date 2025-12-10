import torch.nn as nn
import torch.nn.functional as F
import torch
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
import sys
import os
sys.path.append('..')
from Utils.warp_utils import flow_warp
from Utils.utils import *
from Utils.LPM import *
from Utils.superPointsNet import *
import math



class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg,sp_model):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg
        self.superPoint = sp_model

    def loss_photomatric(self, im1_scaled, im1_recons):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs()]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons,im1_scaled )]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons,im1_scaled)]
            
        if self.cfg.w_mse > 0:
            loss_f=nn.MSELoss()
            loss += [loss_f(im1_recons,im1_scaled)]
        return sum([l.mean() for l in loss])
    
    def get_match(self,pts1,desc1,pts2,desc2):
        pts1 = pts1.cpu().detach().numpy()
        desc1 = desc1.cpu().detach().numpy()
        pts2 = pts2.cpu().detach().numpy()
        desc2 = desc2.cpu().detach().numpy()
        kp1=pts1.T[:,:2]
        kp2=pts2.T[:,:2]
        dsp1,dsp2=desc1.T,desc2.T
        row_indices,col_indices,X,Y=bi_match(kp1,kp2,dsp1,dsp2)
        ok=LPM_filter(X,Y)
        ok_idx = np.where(ok)[0]
        return row_indices[ok_idx],col_indices[ok_idx]

    def sp_loss(self,im1,im2):# 1,1,512,512
        pts1,desc1,_=self.superPoint.run(im1)
        pts2,desc2,_=self.superPoint.run(im2)
        idx1,idx2 = self.get_match(pts1.clone(),desc1.clone(),pts2.clone(),desc2.clone())
        idx1 = torch.from_numpy(idx1)
        idx1.to(im1.device)
        idx2 = torch.from_numpy(idx2)
        idx2.to(im2.device)
        dspX_ok=desc1.permute(1,0)[idx1]
        dspY_ok=desc2.permute(1,0)[idx2]
        descriptor_distances = torch.norm(dspX_ok - dspY_ok, dim=1).sum()/len(dspX_ok)
        
        draw_matching(im1.clone(),im2.clone(),pts1.clone(),pts2.clone(),idx1.clone(),idx2.clone(),self.cfg.match_path)
        return descriptor_distances
    
    def loss_sparse(self,im1_scaled, im1_recons):
        sp_loss=0.0
        B,_,_,_=im1_scaled.size()
        for i in range(B):
            im1 = torch.mean(im1_scaled[i], dim=0, keepdim=True).unsqueeze(1)
            im2 = torch.mean(im1_recons[i], dim=0, keepdim=True).unsqueeze(1)
            sp_loss+=self.sp_loss(im1,im2)
        return sp_loss/B
        
    def loss_smooth(self,flow,im1_scaled):
        # dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        # dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        # # dxy = torch.abs(y_pred[:, 1:, 1:, :] - y_pred[:, :-1, :-1, :])

        # dx = torch.mul(dx, dx)
        # dy = torch.mul(dy, dy)
        # # dxy = torch.mul(dxy, dxy)
        # d = torch.mean(dx) + torch.mean(dy)
        # return d/2.0
        func_smooth = smooth_grad_2nd
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def forward(self,output,img1,img2,epoch):
        pyramid_flows = output
        im1_origin = img1
        im2_origin = img2
        pyramid_smooth_losses = []
        pyramid_dense_losses = []

        for i, flows in enumerate(pyramid_flows):
            if len(flows)==4:
                flow_f,flow_b=flows[2],flows[3]
            else:
                flow_f,flow_b=flows[0],flows[1]
            flow = torch.cat([flow_f,flow_b],1)
            
            b, _, h, w = flow.size()
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')
            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)

            loss_dense = self.loss_photomatric(im1_scaled, im1_recons)

            loss_smooth = 2*self.loss_smooth(flow[:, :2],im1_scaled)
            
            pyramid_dense_losses.append(loss_dense)
            pyramid_smooth_losses.append(loss_smooth)
            
            
            
        dense_loss = 0.0
        for i,loss in enumerate(pyramid_dense_losses):
            dense_loss += loss
        smooth_loss = 0.0    
        for i,loss in enumerate(pyramid_smooth_losses):
            smooth_loss += loss
        if self.cfg.dsp:
            dsp_loss=self.loss_sparse(im1_scaled,im1_recons)
        else:
            dsp_loss = 0.0
        store_img(im1_scaled, im1_recons, im2_scaled, self.cfg.tmp_path,epoch)

        total_loss = dense_loss+smooth_loss+dsp_loss
        return total_loss,dense_loss,smooth_loss,dsp_loss
            

            
            