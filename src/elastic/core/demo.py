import os
import cv2
from tqdm import tqdm
from torchvision import transforms
import time
import numpy as np
import torch
import torch.nn as nn
import argparse
from typing import List, Optional, Tuple
import re
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import gaussian
import inspect
from .EMnet import Framework
from .bigImage import BigImage
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import subprocess
import threading

model_lock = threading.Lock()
model = None  # 需要在主程序中初始化模型

def preprocessing(img_reference: np.ndarray, img_mov: np.ndarray, input_transform: transforms.Compose) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess the input images using the specified transform.

    Args:
        img_reference (np.ndarray): The reference image.
        img_mov (np.ndarray): The moving image.
        input_transform (transforms.Compose): The transformation to apply to the images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The transformed reference and moving images as tensors.
    """
    tensor_mov = input_transform(img_mov)
    tensor_reference = input_transform(img_reference)

    return tensor_reference, tensor_mov


def flow_estimation(
    mov: "BigImage", ref: "BigImage", model: torch.nn.Module, args: argparse.Namespace, input_transform=None
) -> torch.Tensor:
    """
    Estimate optical flow between the moving and reference images.

    Args:
        mov (BigImage): The moving image encapsulated in the BigImage class.
        ref (BigImage): The reference image encapsulated in the BigImage class.
        model (torch.nn.Module): The deep learning model used for flow estimation.
        args (argparse.Namespace): Configuration arguments.
        input_transform (callable, optional): Transformation function to apply to the images.

    Returns:
        torch.Tensor: The estimated flow tensor with shape (1, 2, args.height, args.width).
    """
    print('number of patches: ',len(ref))
    flow_list = []
    idx_list = []
    with torch.no_grad():
        st0 = time.time()
        for i in range(len(ref)):# 这里优化的话，可以考虑多batch的GPU预测，或者多进程，奇怪的是并没有加速，反倒是减速了
            st = time.time()
            patch_ref, idx_ref = ref[i]
            patch_mov, idx_mov = mov[i]
            tensor_ref, tensor_mov = preprocessing(patch_ref, patch_mov, input_transform)
            tensor_ref = tensor_ref.unsqueeze(0).to('cuda')
            tensor_mov = tensor_mov.unsqueeze(0).to('cuda')
            ed = time.time()
            print('preprocessing ',ed-st)
            st = time.time()
            _, _, _, patch_flow = model(tensor_ref, tensor_mov)
            ed = time.time()
            print('model ',ed-st)
            st = time.time()
            flow_list.append(patch_flow.cpu())
            idx_list.append(idx_mov)
            ed = time.time()
            print('cpu ',ed-st)
            # Clear memory
            tensor_ref, tensor_mov, patch_flow = None, None, None
            torch.cuda.empty_cache()
        ed0 = time.time()
        print('Total time  ',ed0-st0)

def flow_estimation_batch_size(
    mov: "BigImage", ref: "BigImage", model: torch.nn.Module, args: argparse.Namespace, input_transform=None, batch_size=4
) -> torch.Tensor:
    """
    Estimate optical flow between the moving and reference images using batch processing.

    Args:
        mov (BigImage): The moving image encapsulated in the BigImage class.
        ref (BigImage): The reference image encapsulated in the BigImage class.
        model (torch.nn.Module): The deep learning model used for flow estimation.
        args (argparse.Namespace): Configuration arguments.
        input_transform (callable, optional): Transformation function to apply to the images.
        batch_size (int): Number of patches to process in each batch.

    Returns:
        torch.Tensor: The estimated flow tensor with shape (1, 2, args.height, args.width).
    """
    print('Number of patches: ', len(ref))
    flow_list = []
    idx_list = []
    with torch.no_grad():
        st0 = time.time()
        for i in range(0, len(ref), batch_size):  # Use batches to process
            st = time.time()

            # Create batch lists
            batch_ref = []
            batch_mov = []
            batch_idx_mov = []

            # Prepare batch data
            for j in range(i, min(i + batch_size, len(ref))):
                patch_ref, idx_ref = ref[j]
                patch_mov, idx_mov = mov[j]
                tensor_ref, tensor_mov = preprocessing(patch_ref, patch_mov, input_transform)
                batch_ref.append(tensor_ref)
                batch_mov.append(tensor_mov)
                batch_idx_mov.append(idx_mov)

            # Convert batch lists to tensors and move to GPU
            batch_ref = torch.stack(batch_ref).to('cuda')  # (batch_size, 1, H, W)
            batch_mov = torch.stack(batch_mov).to('cuda')

            ed = time.time()
            print('Preprocessing ', ed - st)

            # Model prediction
            st = time.time()
            _, _, _, batch_flow = model(batch_ref, batch_mov)
            ed = time.time()
            print('Model inference ', ed - st)

            # Process the batch results
            st = time.time()
            flow_list.extend([flow.cpu() for flow in batch_flow])
            idx_list.extend(batch_idx_mov)
            ed = time.time()
            print('CPU processing ', ed - st)

            # Clear memory
            batch_ref, batch_mov, batch_flow = None, None, None
            torch.cuda.empty_cache()

        ed0 = time.time()
        print('Total time ', ed0 - st0)

    return flow_list, idx_list


def check_gpu_memory():
    """
    检查 GPU 显存使用情况并返回当前显存使用情况。
    """
    # 执行 nvidia-smi 命令并获取输出
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, text=True
    )

    # 解析输出
    lines = result.stdout.strip().split('\n')
    if len(lines) > 0:
        # 处理多行输出，仅考虑第一行数据
        memory_free, memory_total = map(lambda x: int(x.replace(',', '').strip()), lines[0].split(','))
        return memory_total - memory_free  # 返回已用显存
    else:
        raise RuntimeError("Failed to get GPU memory information")


def flow_estimation_thread(
    mov: "BigImage", ref: "BigImage", args: argparse.Namespace, input_transform=None
) -> torch.Tensor:
    """
    Estimate optical flow between the moving and reference images using multi-threading.

    Args:
        mov (BigImage): The moving image encapsulated in the BigImage class.
        ref (BigImage): The reference image encapsulated in the BigImage class.
        args (argparse.Namespace): Configuration arguments.
        input_transform (callable, optional): Transformation function to apply to the images.

    Returns:
        torch.Tensor: The estimated flow tensor with shape (1, 2, args.height, args.width).
    """
    print('number of patches: ', len(ref))
    flow_list = []
    idx_list = []

    def process_patch(i):
        st = time.time()
        patch_ref, idx_ref = ref[i]
        patch_mov, idx_mov = mov[i]
        tensor_ref, tensor_mov = preprocessing(patch_ref, patch_mov, input_transform)
        tensor_ref = tensor_ref.unsqueeze(0).to('cuda')
        tensor_mov = tensor_mov.unsqueeze(0).to('cuda')

        # 打印 GPU 显存使用情况
        print(f'Thread {i} GPU memory used before model inference: {check_gpu_memory()} MiB')

        ed = time.time()
        print(f'Thread {i} preprocessing time: ', ed - st)

        st = time.time()
        with model_lock:
            _, _, _, patch_flow = model(tensor_ref, tensor_mov)
        ed = time.time()
        print(f'Thread {i} model inference time: ', ed - st)

        # 打印 GPU 显存使用情况
        print(f'Thread {i} GPU memory used after model inference: {check_gpu_memory()} MiB')

        patch_flow = patch_flow.cpu()
        idx_mov = idx_mov

        # Clear memory
        tensor_ref, tensor_mov = None, None
        torch.cuda.empty_cache()

        return patch_flow, idx_mov

    st0 = time.time()

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_patch, i) for i in range(len(ref))]

        for future in as_completed(futures):
            patch_flow, idx_mov = future.result()
            flow_list.append(patch_flow)
            idx_list.append(idx_mov)

    ed0 = time.time()
    print('Total time: ', ed0 - st0)




def main():
    parser = argparse.ArgumentParser(description="Gaussian filtering and optical flow estimation.")
    parser.add_argument('--iters', type=int, default=3, help='Number of iterations for the model.')
    parser.add_argument('--patch_sz', type=int, default=1024, help='Size of patches to process.')
    parser.add_argument('--overlap', type=int, default=100, help='Overlap size for patches.')
    parser.add_argument('--num_threads', type=int, default=1)
    args = parser.parse_args()
    
    global model
    model = Framework(args).to('cuda')
    model.eval()
    
    img_mov = BigImage(img=np.random.rand(8192,8192).astype(np.float32),
                       patch_sz=args.patch_sz, overlap=args.overlap)
    img_ref = BigImage(img=np.random.rand(8192,8192).astype(np.float32),
                       patch_sz=args.patch_sz, overlap=args.overlap)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    # flow_estimation(img_mov,img_ref,model,args,input_transform)
    # flow_estimation(img_mov,img_ref,model,args,input_transform)
    
    # flow_estimation_batch_size(img_mov,img_ref,model,args,input_transform,batch_size=1)
    # flow_estimation_batch_size(img_mov,img_ref,model,args,input_transform,batch_size=1)
    
    flow_estimation_thread(img_mov,img_ref,args,input_transform)
    flow_estimation_thread(img_mov,img_ref,args,input_transform)