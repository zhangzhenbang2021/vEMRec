import os
import numpy as np
import random
import sys
import os
import torch
import glob
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import argparse
import cv2

def split_train_val(image_groups, split_ratio=0.9):
    all_groups = list(image_groups.keys())
    random.shuffle(all_groups)  # 随机打乱顺序

    split_index = int(len(all_groups) * split_ratio)
    train_groups = all_groups[:split_index]
    val_groups = all_groups[split_index:]

    train_set = {key: image_groups[key] for key in train_groups}
    val_set = {key: image_groups[key] for key in val_groups}

    return train_set, val_set

# def get_image_groups(folder_path):
#     image_groups = {}
    
#     for file_name in np.sort(os.listdir(folder_path)):
#         if file_name.endswith('.png'):
#             parts = file_name.split('_') # img_i_j.png
#             if len(parts) == 3 and parts[0] == 'img' and parts[2].endswith('.png'):
#                 i = int(parts[1])
#                 j = int(parts[2][0])  
#                 key = f'img_{i}'
#                 if key not in image_groups:
#                     image_groups[key] = []
#                 image_groups[key].append(file_name)
#     return image_groups


def get_image_groups(folder_path,args):
    image_groups = {}

    
    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if args.dataset == 'cremi':
            if os.path.isdir(subfolder_path) and subfolder_path.endswith('warp'):
                for file_name in np.sort(os.listdir(subfolder_path)):
                    if file_name.endswith('.png'):
                        parts = file_name.split('_') # img_i_j.png
                        if len(parts) == 3 and parts[0] == 'img' and parts[2].endswith('.png'):
                            i = int(parts[1])
                            key = f'{subfolder}/img_{i}'
                            if key not in image_groups:
                                image_groups[key] = []
                            
                            image_groups[key].append(f'{subfolder}/{file_name}')
        elif args.dataset == 'openog':
            print(subfolder_path)
            if os.path.isdir(subfolder_path) and 'jrc' in subfolder_path:
                print(subfolder_path)
                for file_name in np.sort(os.listdir(subfolder_path)):
                    if file_name.endswith('.png'):
                        parts = file_name.split('_') # img_i_j.png
                        if len(parts) == 3 and parts[0] == 'img' and parts[2].endswith('.png'):
                            i = int(parts[1])
                            key = f'{subfolder}/img_{i}'
                            if key not in image_groups:
                                image_groups[key] = []
                            
                            image_groups[key].append(f'{subfolder}/{file_name}')
        else:
            raise ValueError(f'Error: Invalid dataset {args.dataset}')
                        
    return image_groups

class createDataset(Dataset):
    def __init__(self, base_path, img_list, transform=None,size=1024):
        super(createDataset,self).__init__()
        self.base_path=base_path
        self.input_transform = transform
        self.size=size
        self.img_list=img_list
        
    def __getitem__(self,index):
        imgs = self.img_list[index]
        # for img_path in imgs:
        #     print(os.path.join(self.base_path,img_path))
        # print('over')
        img_forward = cv2.imread(os.path.join(self.base_path,imgs[2]),0)
        img_backward = cv2.imread(os.path.join(self.base_path,imgs[4]),0) 
        img_moving = cv2.imread(os.path.join(self.base_path,imgs[3]),0)
        
        img_forward_resized = cv2.resize(img_forward, (self.size,self.size)) 
        img_backward_resized = cv2.resize(img_backward, (self.size,self.size)) 
        img_moving_resized = cv2.resize(img_moving, (self.size,self.size))

        if self.input_transform is not None:
            tensor_forward = self.input_transform(img_forward_resized) 
            tensor_backward = self.input_transform(img_backward_resized) 
            tensor_moving = self.input_transform(img_moving_resized)
        # expanded_forward = tensor_forward.unsqueeze(1)
        # tensor_forward = torch.cat(expanded_forward, dim=1)
        # expanded_backward = [img.unsqueeze(1) for img in tensor_backward]
        # tensor_backward = torch.cat(expanded_backward, dim=1)
        data = {'forward':tensor_forward,'moving':tensor_moving,'backward':tensor_backward}
        return data


    
    def __len__(self):
        return len(self.img_list)
    


        

        
def LoadData(args):
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])
    
    image_groups = get_image_groups(args.root_dataset,args)
    train_set, val_set = split_train_val(image_groups, split_ratio=0.9)
    
    train_list = list(train_set.values())
    val_list = list(val_set.values())
    # print(train_list[:1000])
    # print(val_list[:1000])
    train_dataset=createDataset(base_path=args.root_dataset,img_list=train_list,transform=input_transform,size=args.size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    
    val_dataset=createDataset(base_path=args.root_dataset,img_list=val_list,transform=input_transform,size=args.size)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    
    return train_loader,val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1184)
    parser.add_argument('--batch', type=int, default=4, help='number of image pairs per batch on single gpu')
    parser.add_argument("--root_dataset", type=str, default='/home/zhangzhenbang/storage/Cremi/openog_data/train_data')
    parser.add_argument('--base_path', type=str, default='./')
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.batch = args.batch
    print(args)

    train_loader,valid_loader=LoadData(args)
    
    for i_batch, data_blob in enumerate(train_loader):
        
        image1, image2,image3 = data_blob['forward'], data_blob['moving'],data_blob['backward']
        print(image1.size(),image2.size(),image3.size())
        break




