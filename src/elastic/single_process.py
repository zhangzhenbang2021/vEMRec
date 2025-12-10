import subprocess
import os
import random
import numpy as np
import torch
import cv2
import time
import argparse

def split_range(start, end, n, k):
    interval = (end - start + 1) // n  
    start_nums, end_nums = [], []
    for i in range(n):
        if i==0:
            sublist_start = start
            sublist_end = sublist_start + interval - 1  
        elif i==n-1:
            sublist_start = start + i * interval - k 
            sublist_end = end
        else:
            sublist_start = start + i * interval - k 
            sublist_start = max(start, sublist_start)  
            sublist_end = sublist_start + interval - 1  
            sublist_end = min(end, sublist_end) 
        start_nums.append(sublist_start)
        end_nums.append(sublist_end)
    return start_nums, end_nums


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


from core.gauss_filter import stack_reg
def run_process(params):
    stack_reg(params)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian filtering and optical flow estimation.")
    parser.add_argument('--model_path', type=str, default='', help='Path to the trained model.')
    parser.add_argument('--iters', type=int, default=3, help='Number of iterations for the model.')
    parser.add_argument('--iter_T', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=3.0)
    parser.add_argument('--r', type=int, default=1)
    parser.add_argument('--L', type=int, default=1)
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')

    args = parser.parse_args()

    hp = {
        'gpu_id': 0,  
        'start': None,
        'end': None,
        'input_dir': args.input_dir,
        'output_dir':  args.output_dir,
        'flow': False,
        'model_path': args.model_path,
        'iters': args.iters,
        'sigma': args.sigma,
        'r': args.r,
        'L': args.L,
        'reverse': False,
        'height': 0,
        'width': 0,
    }

    start = 0
    imgs = [file for file in os.listdir(hp['input_dir'])]
    img0 = cv2.imread(os.path.join(hp['input_dir'], imgs[0]), 0)
    hp['height'], hp['width'] = img0.shape
    end = len(imgs)

    hp['start'] = start
    hp['end'] = end

    for _ in range(args.iter_T):
        run_process(hp)  
        hp['reverse'] ^= True 

