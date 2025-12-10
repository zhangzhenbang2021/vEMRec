import os
import numpy as np
import cv2
import tensorstore as ts
import pymp
import json

def center_crop(img, crop_size=1024):
    height, width = img.shape
    center_x, center_y = width // 2, height // 2
    start_x = max(center_x - crop_size // 2, 0)
    start_y = max(center_y - crop_size // 2, 0)
    end_x = min(center_x + crop_size // 2, width)
    end_y = min(center_y + crop_size // 2, height)

    if end_x - start_x < crop_size:
        start_x = max(end_x - crop_size, 0)
    if end_y - start_y < crop_size:
        start_y = max(end_y - crop_size, 0)

    cropped_img = img[start_y:end_y, start_x:end_x]

    if cropped_img.shape != (crop_size, crop_size):
        padded_img = np.zeros((crop_size, crop_size), dtype=img.dtype)
        padded_img[:cropped_img.shape[0], :cropped_img.shape[1]] = cropped_img
        cropped_img = padded_img

    return cropped_img

def unitTo8(data):
    data-=data.min()
    data=data/float(data.max()-data.min())
    data*=255.0
    return np.uint8(data)

def n5png(inpath,output,sz=None,len=None):
    input_dataset = ts.open({
    'driver': 'n5',
    'kvstore': {
        'driver': 'file',
        'path': inpath,
        },
    }).result()
    
    x,y,z=input_dataset.shape
    
    if not os.path.exists(output):
      os.makedirs(output)
    
    if len==None:
        len=z
    print(input_dataset.domain)
    for i in range(len):
        img = input_dataset[:,:,i]
        if sz == None:
            img = unitTo8(np.array(img))
        else:
            img = unitTo8(center_crop(np.array(img),sz))
        cv2.imwrite(os.path.join(output,f'{i:06d}.png'),img)
    
                
def pngn5(attributes_path,png_path,output_path,store_type='n5'):
    with open(attributes_path, 'r') as f:
        input_metadata = json.load(f)
        
    output_dataset = ts.open({
    'driver': store_type,
    'kvstore': {
        'driver': 'file',
        'path': output_path,
        },
        'metadata':input_metadata,
        'create': True,
    }).result()
    
    # dataType
    
    img_list = np.sort(os.listdir(png_path))
    for i,_ in enumerate(img_list):
        img = cv2.imread(os.path.join(png_path,img_list[i]),0)
        output_dataset[:,:,i] = img.astype(input_metadata['dataType'])
        
import os
import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter   
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from shutil import copyfile

def elastic_transform(image, alpha, sigma, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    imageC = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
    return imageC,dx,dy,x,y


def add_black_border(image, border_size=400):
    (h, w) = image.shape[:2]
    new_h = h + 2 * border_size
    new_w = w + 2 * border_size
    bordered_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    bordered_image[border_size:border_size + h, border_size:border_size + w] = image
    return bordered_image

def process_border(input_folder, output_folder, border_size=40,height=1400,width=1400):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i,filename in enumerate(sorted(os.listdir(input_folder))):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            bordered_image = add_black_border(image, border_size)
            bordered_image = cv2.resize(bordered_image,(height,width))
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, bordered_image)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Failed to load image: {image_path}")

        
def process_elastic(input_folder, output_folder, sigma=0.08, alpha=4.0,seed=42):
    random_state = np.random.RandomState(seed)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i,filename in enumerate(sorted(os.listdir(input_folder))):
        if filename.endswith('3.png'):
            image_path = os.path.join(input_folder, filename)
            print(filename)
            image = cv2.imread(image_path,0)
            transformed_image,dx,dy,x,y = elastic_transform(image, alpha=image.shape[1]*alpha,sigma=image.shape[1]*sigma,random_state=random_state)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, transformed_image)


def split_images(src_folder, dest_folder, k):
    files = sorted([f for f in os.listdir(src_folder) if f.endswith('.png')])
    os.makedirs(dest_folder, exist_ok=True)

    num_files = len(files)
    for i in range(num_files - k + 1):
        for j in range(k):
            src_file = os.path.join(src_folder, files[i + j])
            dest_file = os.path.join(dest_folder, f"img_{i}_{j}.png")
            copyfile(src_file, dest_file)
            

if __name__ == '__main__':
    
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    n5png(f'{current_dir}/../../raw_data/jrc_mus-kidney-2/em/fibsem-uint8/s3/',
        './../../raw_data/jrc_mus-kidney-2-s3',
        sz=None,len=None)
    n5png(f'{current_dir}/../../raw_data/jrc_mus-sc-zp104a/em/fibsem-uint8/s3/',
        './../../raw_data/jrc_mus-sc-zp104a-s3',
        sz=None,len=None)
    n5png(f'{current_dir}/../../raw_data/jrc_mus-liver-2/em/fibsem-uint8/s3/',
        './../../raw_data/jrc_mus-liver-2-s3',
        sz=None,len=None)
    n5png(f'{current_dir}/../../raw_data/jrc_mus-sc-zp105a/em/fibsem-uint8/s2/',
        './../../raw_data/jrc_mus-sc-zp105a-s2',
        sz=None,len=None)
    

    names = ['jrc_mus-liver-2-s3','jrc_mus-sc-zp104a-s3','jrc_mus-sc-zp105a-s2','jrc_mus-kidney-2-s3']
    for name in names:
        inpath = f'./../../raw_data/{name}'
        border_sz = 25
        height,width = 1184,1184
        outpath = f'./../../raw_data/{name}-border-{border_sz}-height-{height}-width-{width}'
        process_border(inpath, outpath,border_size=border_sz,height=height,width=width)
        
        outpath2 = f'./../../train_data/{name}'
        split_images(outpath,outpath2,k=6)
        
        alpha = 4.0
        sigma = 0.08
        process_elastic(outpath2,outpath2,alpha=alpha,sigma=sigma,seed=42)
