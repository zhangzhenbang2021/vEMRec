import os
from core.gauss_filter_big import stack_reg
from typing import List, Optional, Tuple
import argparse

def run_process(params: dict) -> None:
    """
    Run the stack registration process with the given parameters.

    Args:
        params (dict): Dictionary containing the parameters for stack registration.
    """
    stack_reg(params)

def main(args):
    start, end = args.start, args.end
    if args.start != -1 and args.end != -1:
        start, end = args.start, args.end
    else: 
        start = 0
        end = len(os.listdir(args.input_dir))

    hp_template = {
        'gpu_id': args.gpu_id,
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
    }

    hp = hp_template.copy()
    hp['start'] = start
    hp['end'] = end

    for _ in range(2):  
        run_process(hp) 

        hp['reverse'] ^= True  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian filtering and optical flow estimation.")
    parser.add_argument('--start', type=int, default=-1, help='Start index for processing images.')
    parser.add_argument('--end', type=int, default=-1, help='End index for processing images.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use.')  # 修改为单个 GPU ID
    parser.add_argument('--input_dir', type=str, default='', help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, default='', help='Directory to save output images.')
    parser.add_argument('--model_path', type=str, default='', help='Path to the trained model.')
    parser.add_argument('--iters', type=int, default=3, help='Number of iterations for the model.')
    parser.add_argument('--sigma', type=float, default=3.0, help='Sigma value for Gaussian weights.')
    parser.add_argument('--r', type=int, default=1, help='Radius for context images.')
    parser.add_argument('--L', type=int, default=1, help='Range of layers to consider.')
    
    parser.add_argument('--height', type=int, default=8192, help='Height of the images.')
    parser.add_argument('--width', type=int, default=8192, help='Width of the images.')
    parser.add_argument('--patch_sz', type=int, default=1024, help='Size of patches to process.')
    parser.add_argument('--overlap', type=int, default=50, help='Overlap size for patches.')
    parser.add_argument('--merge_method', type=str, default='average')

    args = parser.parse_args()
    main(args)
