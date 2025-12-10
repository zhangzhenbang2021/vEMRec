import os
import cv2
import torch
import torch.nn as nn
import sys
import numpy as np
import argparse
from dataset import LoadData
import torch.optim as optim
import random
import Loss.losses as losses
from torchvision import transforms
from flow_display import dense_flow
from EMnet import Framework

import torch.distributed as dist
torch.autograd.set_detect_anomaly(True)

def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    np.random.seed(seed)              
    random.seed(seed)                 
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def store_img(forward_fixed,backward_fixed,val_mov,warpImgs,flows,tmp_path,epoch,step):

    denormalize = transforms.Normalize(-1, 2)  
    mov = ((np.array(denormalize(val_mov[0]).data.numpy().squeeze()))*255).astype(np.uint8)
    mov = np.asarray(mov)  
    cv2.imwrite(os.path.join(tmp_path, 'img_'+str(epoch)+'_'+str(step)+'_'+'mov_.png'), mov)

    forward_fixed = forward_fixed[0]
    fix = ((np.array(denormalize(forward_fixed).data.numpy().squeeze()))*255).astype(np.uint8)
    fix = np.asarray(fix)
    cv2.imwrite(os.path.join(tmp_path, 'img_'+str(epoch)+'_'+str(step)+'_forward.png'), fix)

    backward_fixed = backward_fixed[0]
    fix = ((np.array(denormalize(backward_fixed).data.numpy().squeeze()))*255).astype(np.uint8)
    fix = np.asarray(fix)  
    cv2.imwrite(os.path.join(tmp_path, 'img_'+str(epoch)+'_'+str(step)+'_backword.png'), fix)
          
    for idx,val_warp in enumerate(warpImgs):
        warp = ((np.array(denormalize(val_warp[0]).cpu().data.numpy().squeeze()))*255).astype(np.uint8)
        warp = np.asarray(warp)  
        cv2.imwrite(os.path.join(tmp_path, 'img_'+str(epoch)+'_'+str(step)+'_'+f'warp_{idx}.png'), warp)
    for idx,flow in enumerate(flows):
        flow = flow.permute(0,2,3,1)
        flow = flow[0].cpu().data.numpy()
        dense_flow(flow,save_path = os.path.join(tmp_path, 'img_'+str(epoch)+'_'+str(step)+'_'+f'flow_{idx}.png'))



            

    
def fetch_optimizer(args, model,epoch_num):

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    milestones = [int(9*epoch_num/10)] 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return optimizer, scheduler



def fetch_loss(deforms,image1,image3,image2_warped,theta):
    '''
    img1:(B,1,H,W)  img2:(B,1,H,W) img3:(B,1,H,W)
    '''

    sim_loss = 0.0
    sim_loss += losses.similarity_loss(image1, image2_warped)
    sim_loss += losses.similarity_loss(image3, image2_warped)


    reg_loss = 0.0
    for i in range(len(deforms)):
        print(deforms[i].size())
        reg_loss += losses.regularize_loss(deforms[i])
    reg_loss*=10.0
    whole_loss = sim_loss + theta * reg_loss
    metrics = {
        'sim_loss': sim_loss.item(),
        'reg_loss': reg_loss.item()
    }
    return whole_loss, metrics


def validate(model, dataloader, epoch):
    with torch.no_grad():
        model.eval()  
        total_loss = 0.0

        avgloss = {'sim_loss':0.0,'reg_loss':0.0}
        for i_batch, data_blob in enumerate(dataloader):
            image1, image2, image3 = data_blob['forward'].cuda(non_blocking=True), data_blob['moving'].cuda(non_blocking=True), data_blob['backward'].cuda(non_blocking=True)

            deforms, warpImgs, warpImg2,_ = model(image1,image2)
            loss, metrics = fetch_loss(deforms, image1, image3, warpImg2,theta=args.theta)
            
            avgloss['reg_loss']+=metrics['reg_loss']
            avgloss['sim_loss']+=metrics['sim_loss']
            total_loss += loss.item()
            
            if (args.tmp and i_batch % 50 == 0):
                store_img(image1.cpu(),image3.cpu(),image2.cpu(),warpImgs,deforms,args.val_path,epoch,i_batch)


        avgloss['reg_loss']/=float(len(dataloader))
        avgloss['sim_loss']/=float(len(dataloader))
        return avgloss


def load_checkpoint(model_path):
    weights = torch.load(model_path,map_location='cpu')
    epoch = None
    if 'epoch' in weights:
        epoch = weights.pop('epoch')
    if 'state_dict' in weights:
        state_dict = (weights['state_dict'])
    else:
        state_dict = weights
    return epoch, state_dict

def restore_model(model, pretrained_file):
    epoch, weights = load_checkpoint(pretrained_file)

    model_keys = set(model.state_dict().keys())
    weight_keys = set(weights.keys())

    # load weights by name
    weights_not_in_model = sorted(list(weight_keys - model_keys))
    model_not_in_weights = sorted(list(model_keys - weight_keys))
    if len(model_not_in_weights):
        print('Warning: There are weights in model but not in pre-trained.')
        for key in (model_not_in_weights):
            print(key)
            weights[key] = model.state_dict()[key]
    if len(weights_not_in_model):
        print('Warning: There are pre-trained weights not in model.')
        for key in (weights_not_in_model):
            print(key)
        from collections import OrderedDict
        new_weights = OrderedDict()
        for key in model_keys:
            new_weights[key] = weights[key]
        weights = new_weights

    model.load_state_dict(weights)
    return model

def load_latest_model(model, args):
    if args.model_path:
        pth_files = [f for f in os.listdir(args.model_path) if f.endswith('.pth')]
        if not pth_files:
            print("No .pth files found in the specified directory.")
            return 0, model
        
        max_num = -1
        latest_model_path = None
        for file in pth_files:
            num = int(file.split('.')[0])  
            if num > max_num:
                max_num = num
                latest_model_path = os.path.join(args.model_path, file)
        if latest_model_path:
            print(f"Loading model from {latest_model_path}")
            model.load_state_dict(torch.load(latest_model_path))
            print(f"Loaded model at epoch {max_num}")
            return max_num, model
        else:
            print("Not valid .pth files found in the specified directory.")
            return 0, model
    else:
        print("Model path not provided.")
        return 0, model




if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--theta', type=float, default=18.0)
    parser.add_argument('--size', type=int, default=1184)
    parser.add_argument("--epoch_num", type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument("--root_dataset", type=str, default='./train_data')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument('--base_path', type=str, default='./EMnet')
    parser.add_argument('--iters', type=int, default=3) 
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tmp", type=int, default=1)
    parser.add_argument('--sub_path', type=str, default='')
    parser.add_argument('--fine', type=int, default=0)
    parser.add_argument('--pre_model', type=str, default='')
    
    args = parser.parse_args()
    
    seed = 42
    init_seed(seed)
    
    args.sub_path = f'seed_{seed}_theta_{str(args.theta)}_lr_{str(args.lr)}_iter_{str(args.iters)}_fine_{str(args.fine)}'
    args.model_path = os.path.join(args.base_path,args.sub_path,'checkpoints')
    if args.tmp:
        args.tmp_path = os.path.join(args.base_path,args.sub_path,'tmp')
        args.val_path = os.path.join(args.base_path,args.sub_path,'val')

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.tmp_path, exist_ok=True)
    os.makedirs(args.val_path, exist_ok=True)
    
    print(args)


    train_loader,valid_loader=LoadData(args)
    print("data loaded.")
    
    model=Framework(args)
    print("model loaded.")
    optimizer, scheduler = fetch_optimizer(args, model ,args.epoch_num)
    model = model.cuda()
    start_epoch = 0
    if args.fine and args.pre_model != '':
        model = restore_model(model,args.pre_model)
        print('load per_model from:',args.pre_model)
    else:
        start_epoch,model = load_latest_model(model,args) 
        
    print('train.....')
    for epoch in range(start_epoch,args.epoch_num):
        avgloss = {'sim_loss':0.0,'reg_loss':0.0}
        for i_batch, data_blob in enumerate(train_loader):
            model.train()
            image1, image2, image3 = data_blob['forward'].cuda(non_blocking=True), data_blob['moving'].cuda(non_blocking=True), data_blob['backward'].cuda(non_blocking=True)
            print(image1.size(),image2.size(),image3.size())
            optimizer.zero_grad()
            deforms, warpImgs, warpImg2,_ = model(image1,image2)
            loss, metrics = fetch_loss(deforms,image1,image3,warpImg2,theta=args.theta)
            loss.backward()
            avgloss['reg_loss']+=metrics['reg_loss']
            avgloss['sim_loss']+=metrics['sim_loss']
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            if (args.tmp and i_batch % 500 == 0):
                store_img(image1.cpu(),image3.cpu(),image2.cpu(),warpImgs,deforms,args.tmp_path,epoch,i_batch)


        avgloss['reg_loss']/=float(len(train_loader))
        avgloss['sim_loss']/=float(len(train_loader))

        scheduler.step()
        valloss = validate(model, valid_loader, epoch)
        print("avg epoch={},lr={}".format(epoch+1,optimizer.param_groups[0]['lr']), avgloss,valloss)
            
        PATH = args.model_path + '/%d.pth' % (epoch + 1)
        torch.save(model.state_dict(), PATH)

    PATH = args.model_path + '/final.pth' 
    torch.save(model.state_dict(), PATH)
    


