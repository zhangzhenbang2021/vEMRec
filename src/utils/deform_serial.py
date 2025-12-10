# -*- coding:utf-8 -*-
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt  
import os
import re

def get_ij_from_filename(filename):
    match = re.match(r'img_(\d+)_(\d+).png', filename)
    if match:
        i_value = int(match.group(1))
        j_value = int(match.group(2))
        return i_value, j_value
    else:
        return None, None

def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    imageC = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
    return imageC

import os
import cv2

def load_image(folder, filename):
    image_path = os.path.join(folder, filename)
    return cv2.imread(image_path, 0)  # 以灰度图方式读取图像

def save_image(folder, filename, image):
    save_path = os.path.join(folder, filename)
    cv2.imwrite(save_path, image)

def process_train(folder1,output_folder1,alpha,sigma):
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    files1 = np.sort(os.listdir(folder1))
    for idx,filename1 in enumerate(files1):
        image1 = load_image(folder1, filename1)
        if filename1.endswith("3.png"):#idx>2
            image1 = elastic_transform(image1, alpha=image1.shape[1]*alpha,
                                            sigma=image1.shape[1] * sigma) 
        save_image(output_folder1, filename1, image1)

def process_test(folder1,output_folder1,alpha,sigma):
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    files1 = np.sort(os.listdir(folder1))

    for idx,filename1 in enumerate(files1):
        image1 = load_image(folder1, filename1)
        if idx>2:
            image1 = elastic_transform(image1, alpha=image1.shape[1]*alpha,
                                            sigma=image1.shape[1] * sigma) 
        save_image(output_folder1, f'img{idx:04d}.png', image1)
        
def merge_folders(folder1, folder2, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    for i, filename in enumerate(os.listdir(folder1)):
        if filename.endswith(".png"):
            img_path = os.path.join(folder1, filename)
            img = cv2.imread(img_path)
            new_img_path = os.path.join(output_folder, filename)
            cv2.imwrite(new_img_path, img)
    pre = (i+1)//6
    for _, filename in enumerate(os.listdir(folder2)):
        if filename.endswith(".png"):
            num1,num2 = get_ij_from_filename(filename)
            img_path = os.path.join(folder2, filename)
            img = cv2.imread(img_path)
            new_filename = f"img_{num1+pre}_{num2}.png"
            new_img_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(new_img_path, img)
            

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Data Processing Script")
    parser.add_argument("--input_dir", type=str, help="Input folder for training data")
    parser.add_argument("--output_dir", type=str, help="Output folder for processed data")
    parser.add_argument("--alpha", type=float, help="Alpha parameter for processing")
    parser.add_argument("--sigma", type=float, help="Sigma parameter for processing")

    args = parser.parse_args()
    process_train(args.input_dir, args.output_dir, args.alpha, args.sigma)
