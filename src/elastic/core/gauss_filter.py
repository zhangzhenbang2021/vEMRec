import os
import cv2
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import torch
from .EMnet import Framework
import argparse
import inspect
import torch.nn as nn
import re
from scipy.signal import gaussian


def sort_by_number(s):
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else 0

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

def preprocessing(img_reference,img_mov,input_transform=None):

    tensor_mov = input_transform(img_mov)
    tensor_reference = input_transform(img_reference)
    
    return tensor_reference,tensor_mov

def postprocessing(warpTensor):
    denormalize = transforms.Normalize(-1, 2)
    warp = ((np.array(denormalize(warpTensor).cpu().data.numpy().squeeze()))*255).astype(np.uint8)
    warp = np.asarray(warp) 
    
    return warp

def flow_estimation(mov,ref,model,input_transform=None):
    '''
    mov ref:np.array
    return flow (12HW)
    '''
    tensor_reference,tensor_mov = preprocessing(ref,mov,input_transform)
    tensor_reference,tensor_mov = tensor_reference.unsqueeze(0).to('cuda'),tensor_mov.unsqueeze(0).to('cuda')
    _, _, _, flow = model(tensor_reference,tensor_mov)
    tensor_reference = None
    tensor_mov = None
    torch.cuda.empty_cache()

    return flow.cpu()
    
def img_update(img_orgin,flow,input_transform=None):
    tensor_mov = input_transform(img_orgin).unsqueeze(0)
    tensor_warp = flow_warp(tensor_mov,flow)
    img_warp = postprocessing(tensor_warp)

    return img_warp
    
def read_imgs_raduis(img_paths,start_i,radius):
    imgs_cur = []
    for k in range(-radius,radius+1):
        # print('感受野：',img_paths[start_i+k])
        img = cv2.imread(img_paths[start_i+k],0)
        imgs_cur.append(img)
    return imgs_cur

def read_img(img_paths):
    return cv2.imread(img_paths,0)

def gaussian_weights(sigma, r):
    weights = gaussian(2 * r + 1, sigma)
    normalized_weights = weights / np.sum(weights)
    return normalized_weights


def gauss_filter(model,r,L,args):
    '''
    start_i:当前要处理的目标图像在img_paths中的索引
    imgs_cur:目标图像I_i的周围[-r,r]的图像
    imgs_tmp:产生的中间图像
    ws_tmp:中间图像的权重
    img_paths:实时更新的图像栈路径
    '''

    
    img_names = []     
    img_paths = []
    weights = gaussian_weights(args.sigma,r)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])
    print(args)
    if args.reverse == False:
        all_image_names = [file for file in os.listdir(args.input_dir)]
        all_image_names = sorted(all_image_names,key=sort_by_number)
        # print('all_image_names ',all_image_names)
        img_names = all_image_names[args.start:args.end]
        for img_name in img_names:
            img_paths.append(os.path.join(args.input_dir,img_name))

        #前补充
        if args.start == 0: # 0 1 2 3 ... 60 --> 0 0 0 1 2 3 ... 60
            for _ in range(r-1):
                img_paths.insert(0,os.path.join(args.input_dir,img_names[0]))
        else: # 33 34 ... 69 --> 31 32 33 34 ... 69
            for i in range(r-1):
                img_paths.insert(0,os.path.join(args.input_dir,all_image_names[args.start-i-1]))
        
        #后补充
        if args.end == len(all_image_names): # 190 ... 197 198 --> 190 ... 197 198 198 198
            for _ in range(r):
                img_paths.insert(-1,os.path.join(args.input_dir,img_names[-1]))
        else:# # 77 ... 111 112 --> 77 ... 111 112 113 114
            for i in range(r):
                img_paths.insert(-1,os.path.join(args.input_dir,all_image_names[args.end+i]))
        
        # print('img_paths ',img_paths)
        first_img = cv2.imread(os.path.join(args.input_dir,img_names[0]),0)
        cv2.imwrite(os.path.join(args.output_dir,img_names[0]),first_img)
        
    elif args.reverse == True:
        all_image_names = [file for file in os.listdir(args.output_dir)]
        all_image_names = sorted(all_image_names,key=sort_by_number)
        # print('all_image_names ',all_image_names)
        img_names = all_image_names[args.start:args.end]
        img_names = img_names[::-1]
        for img_name in img_names:
            img_paths.append(os.path.join(args.output_dir,img_name))
        
        #后补充  
        if args.start == 0: # 87 86 85 ... 0 --> 87 86 85 ... 0 0 0
            for _ in range(r):
                img_paths.insert(-1,os.path.join(args.output_dir,img_names[-1]))
        else:# 87 86 85 ... 34 --> 87 86 85 ... 34 33 32
            for i in range(r):
                img_paths.insert(-1,os.path.join(args.output_dir,all_image_names[args.start-i-1]))
        
        #前补充
        if args.end == len(all_image_names):# 198 197 196 ... 80 --> 198 198 198 197 196 ... 80
            for _ in range(r-1):
                img_paths.insert(0,os.path.join(args.output_dir,img_names[0]))
        else:# 134 133 132 ... 66 --> 136 135 134 133 132 ... 66
            for i in range(r-1):
                img_paths.insert(0,os.path.join(args.output_dir,all_image_names[args.end+i]))

        # print('img_paths ',img_paths)
    
    start_i=1
    start_i+=(r-1)
                
    # print(img_paths)
    pid = os.getpid()
    progress = tqdm(range(start_i,len(img_names)+r-1))
    imgs_cur = []
    for i in progress:
        progress.set_postfix_str(f"Process ID: {pid}")
        imgs_tmp = []
        ws_tmp = []
        if len(imgs_cur)==0:
            for k in range(-r,r+1):
                img = cv2.imread(img_paths[i+k],0)
                imgs_cur.append(img)
        else:
            imgs_cur=imgs_cur[1:]
            imgs_cur.append(cv2.imread(img_paths[i+r],0))
        
        target_idx = len(imgs_cur)//2 
        for k in range(-r+1,r+1):
            # print('对于感受野的第',k,k+r,end=' ')
            w_tmp=0.0
            phi_sum = torch.zeros((1,2,args.height,args.width),dtype=torch.float32)
            for l in range(max(-r,k-L),k):
                # print(f'ta要与{l+r}进行warp',end=' ')
                w_tmp+=(weights[r+l]+weights[r+k]/(k-max(-r,k-L)))
                phi = weights[l+r]*flow_estimation(mov=imgs_cur[k+r], # read_img(args.input_dir,img_names[i+k]
                                                ref=imgs_cur[l+r], # read_img(args.input_dir,img_names[i+l]
                                                model=model,input_transform=input_transform)
                phi_sum+=phi
            # print()
            img_tmp = img_update(imgs_cur[target_idx+k],phi_sum,input_transform=input_transform)
            ws_tmp.append(w_tmp)
            imgs_tmp.append(img_tmp)
            imgs_cur[target_idx+k]=img_tmp
            
        ws_tmp = np.array(ws_tmp)
        ws_tmp = ws_tmp/float(np.sum(ws_tmp))
        phi_1=flow_estimation(mov=imgs_cur[target_idx],ref=imgs_cur[target_idx-1],
                              model=model,input_transform=input_transform)
        phi_2=torch.zeros((1,2,args.height,args.width),dtype=torch.float32)
        

        for j,img_tmp in enumerate(imgs_tmp):
            phi_tmp = flow_estimation(mov=imgs_cur[target_idx],ref=img_tmp,
                                            model=model,input_transform=input_transform)
            phi_2+=ws_tmp[j]*phi_tmp
        final_flow = (phi_1+phi_2)/2.0 # torch B2HW
        img_i_warp = img_update(imgs_cur[target_idx],final_flow,
                                input_transform=input_transform) 
        imgs_cur[target_idx]=img_i_warp
        out_path = os.path.join(args.output_dir,img_names[i-r+1])
        cv2.imwrite(out_path,img_i_warp)
        img_paths[i]=out_path
        
    
    

        

def stack_reg(params):
    torch.cuda.set_device(params['gpu_id'])
    process_id = os.getpid()
    print(f'PID: {process_id} ',params)
    
    args = argparse.Namespace()  
    for key in params:
        setattr(args, key, params[key])
    os.makedirs(args.output_dir, exist_ok=True)

    aligner = Framework(args).to('cuda')
    aligner = restore_model(aligner,args.model_path)
    aligner.eval()
    
    with torch.no_grad():
        gauss_filter(model=aligner,r=args.r,L=args.L,args=args)


                                    