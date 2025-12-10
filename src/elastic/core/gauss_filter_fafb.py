import os
import cv2
from tqdm import tqdm
from datetime import datetime
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

global_base_grid = None
global_gaussian_map = None

def get_current_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PatchReader:
    def __init__(self, parent_folder, base_folder: str):
        """
        Initialize a PatchReader for a given base folder.
        
        Args:
            base_folder (str): Path to the base folder containing patch sections.
        """
        
        if parent_folder[-1]!='/': # '/storage/zhangzhenbang/FAFB_TEMCA2/' 注意这里必须以/结尾
            self.parent_folder = parent_folder+'/' 
        else:
            self.parent_folder = parent_folder
            
        self.base_folder = base_folder
        self.sections = sorted(
            [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))],
            key=int
        )
        self.img_h = 8192
        self.img_w = 8192

    def __getitem__(self, index: Tuple[int, int]) -> List[Optional[str]]:
        """
        Get the list of patch file paths for a given (row, column) position.
        
        Args:
            index (Tuple[int, int]): (row, column) position to query.
        
        Returns:
            List[Optional[str]]: List of patch file paths, or None if not found.
        """
        row, column = index
        patches = []
        for section in self.sections:
            patch_filename = f"{section}.0.{row}.{column}.png"
            patch_path = os.path.join(self.base_folder, section, patch_filename)
            if os.path.exists(patch_path):
                patch_path = patch_path.replace(self.parent_folder, "")
                patches.append(patch_path)
            else:
                patches.append('None')
        return patches

class ImageReader:
    def __init__(self, block, base_folders: List[str]):
        """
        Initialize an ImageReader with multiple base folders.
        
        Args:
            base_folders (List[str]): List of paths to base folders containing patch sections.
        """
        # print(base_folders)
        self.base_folders = base_folders
        self.sub_folders = sorted([folder for folder in os.listdir(self.base_folders) if folder.startswith(block) and os.path.isdir(os.path.join(self.base_folders, folder))])
        # print(self.sub_folders)
        self.patch_readers = [PatchReader(self.base_folders, os.path.join(self.base_folders, folder)) for folder in self.sub_folders]
        
     

    def __getitem__(self, index: Tuple[int, int]) -> List[Optional[str]]:
        """
        Get the list of patch file paths across all base folders for a given (row, column) position.
        
        Args:
            index (Tuple[int, int]): (row, column) position to query.
        
        Returns:
            List[Optional[str]]: List of patch file paths from all base folders, or None if not found.
        """
        row, column = index
        patches = []
        
        for reader in self.patch_readers:
            patches.extend(reader[row, column])
        
        return patches

def get_all_image_names(args):
    """
    Retrieves all image names corresponding to the specified row and column 
    from the input directory using the ImageReader class.

    Args:
        args: An argument parser object that contains the input directory, 
              row position (row_p), and column position (col_p).

    Returns:
        List of image file names at the specified row and column positions.
    """
    image_reader = ImageReader(args.block, args.input_dir)
    return image_reader[args.row_p, args.col_p]


def sort_by_number(s: str) -> int:
    """
    Extract the first number found in a string and return it as an integer.
    If no numbers are found, return 0.

    Args:
        s (str): Input string.

    Returns:
        int: The first number found in the string, or 0 if no numbers are found.
    """
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else 0

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

def mesh_grid(B, H, W):
    y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((x, y), 2).float()
    grid = grid.permute(2, 0, 1)  # Change shape to (2, H, W)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Repeat for batch size
    return grid

def norm_grid(grid):
    # Normalize grid from [0, H] and [0, W] to [-1, 1]
    H, W = grid.shape[-2], grid.shape[-1]
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / (W - 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / (H - 1) - 1.0
    return grid.permute(0, 2, 3, 1)  # Change shape to (B, H, W, 2)

def flow_warp(x, flow12, pad='border', mode='bilinear'):
    global global_base_grid
    B, _, H, W = x.size()

    if global_base_grid is None or global_base_grid.size() != (B, 2, H, W):
        global_base_grid = mesh_grid(B, H, W).type_as(x)

    v_grid = norm_grid(global_base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def load_checkpoint(model_path: str) -> Tuple[Optional[int], dict]:
    """
    Load model weights from a checkpoint file.

    Args:
        model_path (str): Path to the checkpoint file.

    Returns:
        Tuple[Optional[int], dict]: Tuple containing the epoch number (if available) and the state dictionary.
    """
    weights = torch.load(model_path, map_location='cpu')
    epoch = weights.pop('epoch', None)
    state_dict = weights.get('state_dict', weights)

    return epoch, state_dict


def restore_model(model: nn.Module, pretrained_file: str) -> nn.Module:
    """
    Restore model weights from a pretrained file.

    Args:
        model (nn.Module): The model to restore.
        pretrained_file (str): Path to the pretrained file.

    Returns:
        nn.Module: The model with restored weights.
    """
    epoch, weights = load_checkpoint(pretrained_file)

    model_keys = set(model.state_dict().keys())
    weight_keys = set(weights.keys())

    # Load weights by name, matching keys between model and weights
    weights_not_in_model = sorted(list(weight_keys - model_keys))
    model_not_in_weights = sorted(list(model_keys - weight_keys))

    if model_not_in_weights:
        print('Warning: There are weights in model but not in pre-trained:')
        for key in model_not_in_weights:
            print(key)
            weights[key] = model.state_dict()[key]

    if weights_not_in_model:
        print('Warning: There are pre-trained weights not in model:')
        for key in weights_not_in_model:
            print(key)
        
        # Create a new weights dictionary with model keys
        from collections import OrderedDict
        new_weights = OrderedDict()
        for key in model_keys:
            new_weights[key] = weights[key]
        weights = new_weights

    model.load_state_dict(weights)
    return model


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


def postprocessing(warpTensor: torch.Tensor) -> np.ndarray:
    """
    Postprocess the warped tensor to convert it back to an image.

    Args:
        warpTensor (torch.Tensor): The warped tensor.

    Returns:
        np.ndarray: The postprocessed image as a NumPy array.
    """
    denormalize = transforms.Normalize(-1, 2)
    warp = ((np.array(denormalize(warpTensor).cpu().data.numpy().squeeze())) * 255).astype(np.uint8)
    warp = np.asarray(warp)

    return warp




def get_gaussian(s: tuple, sigma: float = 1.0 / 8) -> torch.Tensor:
    """
    Generate a Gaussian map with the given shape and standard deviation.

    Args:
        s (tuple): The shape of the Gaussian map (height, width).
        sigma (float): The standard deviation as a fraction of the shape dimensions.

    Returns:
        torch.Tensor: The generated Gaussian map.
    """
    y = torch.arange(s[0])
    x = torch.arange(s[1])
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    center_y = (s[0] - 1) / 2.0
    center_x = (s[1] - 1) / 2.0
    
    sigma_y = s[0] * sigma
    sigma_x = s[1] * sigma

    gaussian_map = torch.exp(-(((yy - center_y) ** 2) / (2 * sigma_y ** 2) +
                               ((xx - center_x) ** 2) / (2 * sigma_x ** 2)))

    gaussian_map /= torch.max(gaussian_map)
    
    return gaussian_map

def flow_merge(flow_list: list, idx_list: list, patch_size: int, h: int, w: int, method: str = 'average', sigma: float = 1.0) -> torch.Tensor:
    """
    Merge a list of flow fields using specified weighting methods.
    这三种融合策略，只有average没有改变非重叠区域的位移值，看看实验效果再说吧
    """
    flow = torch.zeros((1, 2, h, w), dtype=torch.float32)
    weight_sum = torch.zeros((1, 1, h, w), dtype=torch.float32)

    if method == 'average':
        weights = torch.ones((1, 1, patch_size, patch_size), dtype=torch.float32)

    elif method == 'linear':
        weight_x = torch.linspace(1, 0, patch_size).unsqueeze(0).repeat(patch_size, 1)
        weight_y = torch.linspace(1, 0, patch_size).unsqueeze(1).repeat(1, patch_size)
        weights = torch.min(weight_x, weight_y).unsqueeze(0).unsqueeze(0)

    elif method == 'gaussian':
        x = np.arange(patch_size)
        y = np.arange(patch_size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        gaussian_weights = np.exp(-((xx - patch_size / 2) ** 2 + (yy - patch_size / 2) ** 2) / (2 * sigma ** 2))
        weights = torch.from_numpy(gaussian_weights).unsqueeze(0).unsqueeze(0).float()

    else:
        raise ValueError(f"Unknown merging method: {method}")

    for i, flow_patch in enumerate(flow_list):
        idx1, idx2, idy1, idy2 = idx_list[i]
        flow_patch = flow_patch.to(dtype=torch.float32)
        flow[:, :, idx1:idx2, idy1:idy2] += flow_patch * weights
        weight_sum[:, :, idx1:idx2, idy1:idy2] += weights

    # 处理非重叠部分，防止被无效化
    merged_flow = torch.where(weight_sum > 0, flow / weight_sum, flow)
    # weight_sum[weight_sum == 0] = 1
    # merged_flow = flow / weight_sum
    
    return merged_flow


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
            # print('preprocessing ',ed-st)
            st = time.time()
            _, _, _, patch_flow = model(tensor_ref, tensor_mov)
            ed = time.time()
            # print('model ',ed-st)
            st = time.time()
            flow_list.append(patch_flow.cpu())
            idx_list.append(idx_mov)
            ed = time.time()
            # print('cpu ',ed-st)
            # Clear memory
            tensor_ref, tensor_mov, patch_flow = None, None, None
            torch.cuda.empty_cache()
        ed0 = time.time()
        # print('Total time  ',ed0-st0)
    flow = flow_merge(flow_list, idx_list, args.patch_sz, args.height, args.width, method=args.merge_method)
    return flow


def img_update(
    img_orgin: "BigImage", flow: torch.Tensor, args: argparse.Namespace, input_transform=None
) -> "BigImage":
    """
    Update the image based on the estimated flow.

    Args:
        img_orgin (BigImage): The original image encapsulated in the BigImage class.
        flow (torch.Tensor): The flow tensor.
        args (argparse.Namespace): Configuration arguments.
        input_transform (callable, optional): Transformation function to apply to the image.

    Returns:
        BigImage: The updated image encapsulated in the BigImage class.
    """
    tensor_mov = input_transform(img_orgin.merge()).unsqueeze(0)
    tensor_warp = flow_warp(tensor_mov, flow)

    # Denormalize and convert to uint8
    denormalize = transforms.Normalize(-1, 2)
    warp = (np.array(denormalize(tensor_warp).data.numpy().squeeze()) * 255).astype(np.uint8)
    
    warpBigImage = BigImage(img=warp, patch_sz=args.patch_sz, overlap=args.overlap)
    return warpBigImage


def gaussian_weights(sigma: float, r: int) -> np.ndarray:
    """
    Generate Gaussian weights for a given radius and standard deviation.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        r (int): Radius of the Gaussian kernel.

    Returns:
        np.ndarray: Normalized Gaussian weights.
    """
    weights = gaussian(2 * r + 1, sigma)
    normalized_weights = weights / np.sum(weights)
    return normalized_weights


def compute_phi(l, k, r, L, imgs_cur, model, args, input_transform, weights):
    """Compute phi for a specific iteration."""
    w_tmp = (weights[r + l] + weights[r + k] / (k - max(-r, k - L)))
    phi = weights[l + r] * flow_estimation(
        mov=imgs_cur[k + r],
        ref=imgs_cur[l + r],
        model=model,
        args=args,
        input_transform=input_transform
    )
    return phi, w_tmp

def parallel_phi_sum(k, r, L, imgs_cur, model, args, input_transform, weights):
    """Compute the sum of phi in parallel."""
    phi_sum = torch.zeros((1, 2, args.height, args.width), dtype=torch.float32)
    w_tmp_sum = 0.0

    l_values = list(range(max(-r, k - L), k))
    
    with mp.Pool(processes=len(l_values)) as pool:
        results = [
            pool.apply_async(compute_phi, args=(l, k, r, L, imgs_cur, model, args, input_transform, weights))
            for l in l_values
        ]
        
        for result in results:
            phi, w_tmp = result.get()
            phi_sum += phi
            w_tmp_sum += w_tmp
    
    return phi_sum, w_tmp_sum

def compute_flow_estimation(img_tmp, target_img, args, model, input_transform):
    return flow_estimation(
        mov=target_img,
        ref=img_tmp,
        args=args,
        model=model,
        input_transform=input_transform
    )

def gauss_filter(model, r, L, args):
    '''
    Applies Gaussian filtering and optical flow estimation to an image stack.

    Parameters:
    - model: Trained model for optical flow estimation.
    - r: Radius for context images.
    - L: Range of layers to consider.
    - args: Namespace with necessary parameters like input/output directories, image size, etc.

    Updates:
    - Processes and saves images with Gaussian filtering applied.
    '''
    
    SUM_SAVE_IMG_COUNT = 0
    
    def safe_load_image(path):
        '''Safely loads an image and handles errors.'''
        if os.path.exists(path):
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif path[-4:] == 'None':
            return np.zeros((args.height, args.width), np.uint8)
        else:
            raise FileNotFoundError(f"Image at path {path} not found.")
    
    def safe_write_image(path, img):
        '''Safely writes an image, ensuring the directory exists and handles errors.'''
        nonlocal SUM_SAVE_IMG_COUNT
        try:
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            if path[-4:] == 'None':
                # print('none not save ',path)
                return
            else:
                cv2.imwrite(path, img)
                SUM_SAVE_IMG_COUNT += 1
        except Exception as e:
            raise IOError(f"Failed to save image at path {path}: {str(e)}")

    img_names = []
    img_paths = []
    weights = gaussian_weights(args.sigma, r)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    
    all_image_names = get_all_image_names(args)
    # print('all_image_names ',all_image_names)
    if not all_image_names:
        raise ValueError("No images found in the input directory.")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine image paths and apply necessary padding based on direction (reverse or forward)
    if args.reverse == False:
        img_names = all_image_names[args.start:args.end]
        img_paths = [os.path.join(args.input_dir, img_name) for img_name in img_names]
        # print(img_names)
        # Pre-padding
        pre_padding = r - 1
        if args.start == 0:
            img_paths = [img_paths[0]] * pre_padding + img_paths
        else:
            img_paths = [os.path.join(args.input_dir, all_image_names[args.start - i - 1]) for i in range(pre_padding)] + img_paths

        # Post-padding
        post_padding = r
        if args.end == len(all_image_names):
            img_paths += [img_paths[-1]] * post_padding
        else:
            img_paths += [os.path.join(args.input_dir, all_image_names[args.end + i]) for i in range(post_padding)]

        # print('img_paths ',img_paths)
        first_img = safe_load_image(os.path.join(args.input_dir, img_names[0]))
        
        safe_write_image(os.path.join(args.output_dir, img_names[0]), first_img)
    
    else:
        img_names = all_image_names[args.start:args.end][::-1]
        # print(img_names)
        img_paths = [os.path.join(args.output_dir, img_name) for img_name in img_names]

        # Post-padding
        post_padding = r
        if args.start == 0:
            img_paths += [img_paths[-1]] * post_padding
        else:
            img_paths += [os.path.join(args.output_dir, all_image_names[args.start - i - 1]) for i in range(post_padding)]

        # Pre-padding
        pre_padding = r - 1
        if args.end == len(all_image_names):
            img_paths = [img_paths[0]] * pre_padding + img_paths
        else:
            img_paths = [os.path.join(args.output_dir, all_image_names[args.end + i]) for i in range(r)] + img_paths
        
        # print('img_paths ',img_paths)


    start_i=1
    start_i+=(r-1)
    
    imgs_cur = []
    
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    with tqdm(range(start_i, len(img_names) + r - 1), bar_format=bar_format, desc=f"PID {os.getpid()}") as pbar:
        for i in pbar:
            imgs_tmp = []
            ws_tmp = []
            
            if len(imgs_cur)==0:
                imgs_cur = [BigImage(img=safe_load_image(img_paths[i + k]), 
                                    patch_sz=args.patch_sz, overlap=args.overlap) for k in range(-r, r + 1)]
            else:
                imgs_cur = imgs_cur[1:] 
                imgs_cur.append(BigImage(img=safe_load_image(img_paths[i + r]), 
                                        patch_sz=args.patch_sz, overlap=args.overlap))
            
            target_idx = len(imgs_cur) // 2
            
            if not np.all( imgs_cur[target_idx].merge()==0 ):# 查看是不是全0
                for k in range(-r + 1, r + 1):
                    phi_sum = torch.zeros((1, 2, args.height, args.width), dtype=torch.float32)
                    w_tmp = 0.0
                    for l in range(max(-r, k - L), k):
                        w_tmp += (weights[r + l] + weights[r + k] / (k - max(-r, k - L)))
                        phi = weights[l + r] * flow_estimation(mov=imgs_cur[k + r], ref=imgs_cur[l + r], 
                                                            model=model, args=args, input_transform=input_transform)
                        phi_sum += phi

                    img_tmp = img_update(imgs_cur[target_idx + k], phi_sum, args=args, input_transform=input_transform)
                    ws_tmp.append(w_tmp)
                    imgs_tmp.append(img_tmp)
                    imgs_cur[target_idx+k]=img_tmp # 2024/9/3 根据gauss_filer3.py修改添加这一行
                    
                
                ws_tmp = np.array(ws_tmp) / float(np.sum(ws_tmp))
                phi_1 = flow_estimation(mov=imgs_cur[target_idx], ref=imgs_cur[target_idx - 1], args=args, model=model, input_transform=input_transform)
                phi_2 = torch.zeros((1, 2, args.height, args.width), dtype=torch.float32)
                
                # st = time.time()
                for j, img_tmp in enumerate(imgs_tmp):# 这里如果GPU运行，可以考虑并行
                    phi_2 += ws_tmp[j] * flow_estimation(mov=imgs_cur[target_idx], ref=img_tmp, args=args, model=model, input_transform=input_transform)
                # ed = time.time()
                # print('results ',ed-st)

                img_i_warp = img_update(imgs_cur[target_idx], (phi_1 + phi_2) / 2.0, args=args, input_transform=input_transform)
                imgs_cur[target_idx] = img_i_warp

                # print('not empty ',img_names[i - r + 1])
            else:
                # print('all empty ',img_names[i - r + 1])
                img_i_warp = imgs_cur[target_idx] # zeros

            out_path = os.path.join(args.output_dir, img_names[i - r + 1])
            safe_write_image(out_path, img_i_warp.merge())
            img_paths[i] = out_path
            
            elapsed_time = pbar.format_dict['elapsed']
            rate_per_it = elapsed_time / (pbar.n or 1)
            pbar.set_postfix_str(f"Time: {get_current_time()}, {rate_per_it:.2f}s/it")
        
    return SUM_SAVE_IMG_COUNT
        
def stack_reg(params: dict) -> None:
    """
    Perform stack registration using a specified deep learning model.

    Args:
        params (dict): A dictionary of parameters needed for registration.
                       Required keys include:
                       - gpu_id (int): The ID of the GPU to use.
                       - input_dir (str): Path to the input directory containing images.
                       - output_dir (str): Path to the output directory where results will be saved.
                       - model_path (str): Path to the model checkpoint file.
                       - r (int): Radius parameter for the Gaussian filter.
                       - L (int): Parameter for the Gaussian filter.
                       - Other keys as required by the registration framework.

    Returns:
        None
    """
    
    
    # Set the GPU device
    torch.cuda.set_device(params['gpu_id'])
    
    # Print the process ID and current parameters for debugging
    process_id = os.getpid()
    print(f'PID: {process_id} ', params)
    
    # Convert the params dictionary to an argparse.Namespace object for compatibility
    args = argparse.Namespace()
    for key, value in params.items():
        setattr(args, key, value)
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and load the deep learning model
    aligner = Framework(args).to('cuda')
    aligner = restore_model(aligner, args.model_path)
    aligner.eval()

    # Perform the registration using the Gaussian filter
    with torch.no_grad():
       SUM_SAVE_IMG_COUNT = gauss_filter(model=aligner, r=args.r, L=args.L, args=args)
       print('SUM_SAVE_IMG_COUNT ',SUM_SAVE_IMG_COUNT)


                                    