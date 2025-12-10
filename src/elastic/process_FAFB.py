import os
import cv2
import subprocess
import multiprocessing
from core.gauss_filter_fafb import stack_reg
from core.bigImage import BigImage
from typing import List, Optional, Tuple
import argparse
import time

def split_range(start: int, end: int, n: int, overlap: int) -> Tuple[List[int], List[int]]:
    """
    Split the range from start to end into n intervals with a specified overlap.

    Args:
        start (int): Starting index.
        end (int): Ending index.
        n (int): Number of intervals to split into.
        overlap (int): Number of overlapping elements between consecutive intervals.

    Returns:
        Tuple[List[int], List[int]]: Two lists containing the start and end indices of the intervals.
    """
    interval = (end - start + 1) // n
    start_nums, end_nums = [], []

    for i in range(n):
        sublist_start = start + i * interval
        sublist_end = sublist_start + interval - 1
        sublist_start = max(0, sublist_start - overlap)
        start_nums.append(sublist_start)
        end_nums.append(sublist_end)

    return start_nums, end_nums


def run_process(params: dict) -> None:
    """
    Run the stack registration process with the given parameters.

    Args:
        params (dict): Dictionary containing the parameters for stack registration.
    """
    stack_reg(params)


def main(args):
    
    if args.start != -1 and args.end != -1:
        start, end = args.start, args.end
    else: # 保证start=0，end=len(images)
        start = 0
        sub_folders = [sub_dir for sub_dir in os.listdir(args.input_dir) if sub_dir.startswith(args.block) and os.path.isdir(os.path.join(args.input_dir, sub_dir))]
        print(sub_folders)
        end = 0
        for sub_folder in sub_folders:
            end += len(os.listdir(os.path.join(args.input_dir, sub_folder)))
        
    overlap_num = args.overlap_num
    gpu_ids = args.gpu_ids
    hp_list = []

    hp_template = {
        'gpu_id': None,
        'start': None,
        'end': None,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'model_path': args.model_path,
        'iters': args.iters,
        'sigma': args.sigma,
        'r': args.r,
        'L': args.L,
        'reverse': False,
        'height': args.height,
        'width': args.width,
        'patch_sz': args.patch_sz,
        'overlap': args.overlap,
        'row_p': args.row_p,
        'col_p': args.col_p,
        'block':args.block,
        'merge_method':args.merge_method,
    }

    start_nums, end_nums = split_range(start, end, len(gpu_ids), overlap_num)

    for i in range(len(gpu_ids)):
        hp = hp_template.copy()
        hp['gpu_id'] = gpu_ids[i]
        hp['start'] = start_nums[i]
        hp['end'] = end_nums[i]
        hp_list.append(hp)
    
    if not args.restart:
        for _ in range(args.times): 
            run_process(hp_list[0])
            hp_list[0]['reverse'] ^= True
    else:
        for _ in range(args.re_st_time):
            hp_list[0]['reverse'] ^= True
        
        if hp_list[0]['reverse']:
            hp_list[0]['end'] -= args.st
        else:
            hp['start'] = args.st
        for _ in range(args.re_st_time,args.times):
            run_process(hp_list[0])
            hp_list[0]['reverse'] ^= True
            hp_list[0]['start'] = start_nums[0]
            hp_list[0]['end'] = end_nums[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian filtering and optical flow estimation.")
    parser.add_argument('--overlap_num', type=int, default=3, help='Number of overlapping images between splits.')
    parser.add_argument('--model_path', type=str, default='/home/zhangzhenbang/storage/mult_work/model/EMnet/seed_42_theta_18.0_lr_0.005_iter_3/checkpoints/90.pth', help='Path to the trained model.')
    parser.add_argument('--iters', type=int, default=3, help='Number of iterations for the model.')
    parser.add_argument('--sigma', type=float, default=3.0, help='Sigma value for Gaussian weights.')
    parser.add_argument('--r', type=int, default=1, help='Radius for context images.')
    parser.add_argument('--L', type=int, default=1, help='Range of layers to consider.')
    parser.add_argument('--height', type=int, default=8192, help='Height of the images.')
    parser.add_argument('--width', type=int, default=8192, help='Width of the images.')
    parser.add_argument('--patch_sz', type=int, default=1024, help='Size of patches to process.')
    parser.add_argument('--overlap', type=int, default=100, help='Overlap size for patches.')
    parser.add_argument('--times', type=int, default=2, help='')
    
    parser.add_argument('--row_p', type=int, default=7, help='Number of rows in the patch grid.')
    parser.add_argument('--col_p', type=int, default=18, help='Number of columns in the patch grid.')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='List of GPU IDs to use.')
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--start', type=int, default=-1, help='Start index for processing images.')
    parser.add_argument('--end', type=int, default=-1, help='End index for processing images.')
    parser.add_argument('--block', type=str, default=None)
    parser.add_argument('--merge_method', type=str, default=None)

    parser.add_argument('--restart', action='store_true', help='Restart the processing if specified.')
    parser.add_argument('--re_st_time', type=int, default=-1, help='')
    parser.add_argument('--re_st_slice', type=int, default=-1, help='Start index for processing images.')

    args = parser.parse_args()
    main(args)