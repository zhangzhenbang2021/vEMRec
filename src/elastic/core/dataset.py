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

def get_image_groups(folder_path):
    image_groups = {}
    
    for file_name in np.sort(os.listdir(folder_path)):
        if file_name.endswith('.png'):
            parts = file_name.split('_') # img_i_j.png
            if len(parts) == 3 and parts[0] == 'img' and parts[2].endswith('.png'):
                i = int(parts[1])
                j = int(parts[2][0])  
                key = f'img_{i}'
                if key not in image_groups:
                    image_groups[key] = []
                
                image_groups[key].append(file_name)

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
        img_forward = [cv2.imread(os.path.join(self.base_path,img_path),0) for img_path in imgs[:3]]
        img_backward = [cv2.imread(os.path.join(self.base_path,img_path),0) for img_path in imgs[4:]]
        img_moving = cv2.imread(os.path.join(self.base_path,imgs[3]),0)
        
        img_forward_resized = [cv2.resize(img, (self.size,self.size)) for img in img_forward]
        img_backward_resized = [cv2.resize(img, (self.size,self.size)) for img in img_backward]
        img_moving_resized = cv2.resize(img_moving, (self.size,self.size))

        if self.input_transform is not None:
            tensor_forward = [self.input_transform(i) for i in img_forward_resized]
            tensor_backward = [self.input_transform(i) for i in img_backward_resized]
            tensor_moving = self.input_transform(img_moving_resized)
        expanded_forward = [img.unsqueeze(1) for img in tensor_forward]
        tensor_forward = torch.cat(expanded_forward, dim=1)
        expanded_backward = [img.unsqueeze(1) for img in tensor_backward]
        tensor_backward = torch.cat(expanded_backward, dim=1)
        data = {'forward':tensor_forward,'moving':tensor_moving,'backward':tensor_backward}
        return data


    
    def __len__(self):
        return len(self.img_list)
    


        

        
def LoadData(args):
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])
    
    image_groups = get_image_groups(args.root_dataset)
    train_set, val_set = split_train_val(image_groups, split_ratio=0.9)
    
    train_list = list(train_set.values())
    val_list = list(val_set.values())
    print(val_list)
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
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--name', type=str, default='EMNet3d', help='name your experiment')
    parser.add_argument('--dataset', type=str, default='brain', help='which dataset to use for training')
    parser.add_argument("--epoch_num", type=int, default=1500)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--batch', type=int, default=4, help='number of image pairs per batch on single gpu')
    parser.add_argument("--root_dataset", type=str, default='/home/zhangzhenbang/storage/Cremi/bigData/aug_data/abc_padded_warp')
    parser.add_argument('--base_path', type=str, default='./')
    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tmp", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # args.model_path = args.base_path + args.name + '/output/checkpoints_' + args.dataset
    # args.eval_path = args.base_path + args.name + '/output/eval_' + args.dataset
    # if args.tmp:
    #     args.tmp_path = args.base_path + args.name + '/output/tmp_' + args.dataset



    # os.makedirs(args.model_path, exist_ok=True)
    # os.makedirs(args.eval_path, exist_ok=True)
    # os.makedirs(args.tmp_path, exist_ok=True)
    
    args.nums_gpu = torch.cuda.device_count()
    args.batch = args.batch
    
    print(args)


    train_loader,valid_loader=LoadData(args)
    
    for i_batch, data_blob in enumerate(train_loader):
        
        image1, image2,image3 = data_blob['forward'], data_blob['moving'],data_blob['backward']
        break




