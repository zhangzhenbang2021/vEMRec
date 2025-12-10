# from model import trainfunc
import numpy as np
import torch
import torch.utils
import h5py
import matplotlib.pyplot as plt
import cv2, copy
import os
import argparse

def augment(x, y, sz,aug = True):
    new_x = torch.zeros(x.shape[0] * 10, *x.shape[1:3], sz, sz)
    new_y = torch.zeros(y.shape[0] * 10, *y.shape[1:3], sz, sz)
    
    for i in range(x.shape[0]):
        new_anchors = np.random.randint(0, 3072-sz-1, (10, 2))
        # print(new_anchors)
        for j, anchor in enumerate(new_anchors):
            new_x[i * 10 + j, :, :, :, :] = x[i, :, :, 
                                              anchor[0]:anchor[0] + sz, 
                                              anchor[1]:anchor[1] + sz]
            new_y[i * 10 + j, :, :, :, :] = y[i, :, :, 
                                              anchor[0]:anchor[0] + sz, 
                                              anchor[1]:anchor[1] + sz]
        if aug:
            new_x[i * 10 + 4, :, :, :, :] = torch.flip(new_x[i * 10 + 4, :, :, :, :], [2])
            new_x[i * 10 + 5, :, :, :, :] = torch.flip(new_x[i * 10 + 5, :, :, :, :], [3])
            new_x[i * 10 + 6, :, :, :, :] = torch.flip(new_x[i * 10 + 6, :, :, :, :], [2, 3])
            new_y[i * 10 + 4, :, :, :, :] = torch.flip(new_y[i * 10 + 4, :, :, :, :], [2])
            new_y[i * 10 + 5, :, :, :, :] = torch.flip(new_y[i * 10 + 5, :, :, :, :], [3])
            new_y[i * 10 + 6, :, :, :, :] = torch.flip(new_y[i * 10 + 6, :, :, :, :], [2, 3])
            
            new_x[i * 10 + 7, :, :, :, :] = torch.rot90(new_x[i * 10 + 7, :, :, :, :],1, [2, 3])
            new_x[i * 10 + 8, :, :, :, :] = torch.rot90(new_x[i * 10 + 8, :, :, :, :],-1, [2, 3])
            new_y[i * 10 + 7, :, :, :, :] = torch.rot90(new_y[i * 10 + 7, :, :, :, :],1, [2, 3])
            new_y[i * 10 + 8, :, :, :, :] = torch.rot90(new_y[i * 10 + 8, :, :, :, :],-1, [2, 3])

    return new_x, new_y


def aug_data(ipath,opath,sz,border_size,random_seed = None):
    
    if random_seed != None:
        np.random.seed(random_seed)

    f = h5py.File(ipath, 'r')
    output_folder1 = opath

    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    len =  f['/volumes/raw'].shape[0] - 5
    x_sequences = torch.zeros((len,1,5,3072,3072))
    y_label = torch.zeros((len,1,1,3072,3072))
    for i in range(len):
        x_sequences[i, 0, 0,:,:] =  torch.from_numpy(cv2.resize(f['/volumes/raw'][i],(3072, 3072)))
        x_sequences[i, 0, 1,:,:] =  torch.from_numpy(cv2.resize(f['/volumes/raw'][i + 1],(3072, 3072)))
        x_sequences[i, 0, 2,:,:] =  torch.from_numpy(cv2.resize(f['/volumes/raw'][i + 2],(3072, 3072)))
        x_sequences[i, 0, 3,:,:] =  torch.from_numpy(cv2.resize(f['/volumes/raw'][i + 4],(3072, 3072)))
        x_sequences[i, 0, 4,:,:] =  torch.from_numpy(cv2.resize(f['/volumes/raw'][i + 5],(3072, 3072)))
        y_label[i, 0, 0,:,:] = torch.from_numpy(cv2.resize(f['/volumes/raw'][i + 3],(3072, 3072)))
        

    indices_to_remove = []
    for i in range(x_sequences.size(0)):
        if torch.all(x_sequences[i, 0, 0, :, :] == 0) or torch.all(x_sequences[i, 0, 1, :, :] == 0) or \
            torch.all(x_sequences[i, 0, 2, :, :] == 0) or torch.all(x_sequences[i, 0, 3, :, :] == 0) or \
            torch.all(x_sequences[i, 0, 4, :, :] == 0) or torch.all(y_label[i, 0, 0, :, :] == 0):
            indices_to_remove.append(i)
    # print(indices_to_remove)
    X_filtered = torch.cat([x_sequences[i:i+1] for i in range(x_sequences.size(0)) if i not in indices_to_remove], dim=0)
    Y_filtered = torch.cat([y_label[i:i+1] for i in range(y_label.size(0)) if i not in indices_to_remove], dim=0)


    aug_x, aug_y = augment(X_filtered, Y_filtered,sz=sz)# [1, 2, 3, 4, 5, 6, 162, 163, 164, 165, 166, 167]
    for i in range(aug_x.shape[0]):
        img1 = cv2.copyMakeBorder(aug_x[i,0,0,:,:].numpy(), border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        img2 = cv2.copyMakeBorder(aug_x[i,0,1,:,:].numpy(), border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        img3 = cv2.copyMakeBorder(aug_x[i,0,2,:,:].numpy(), border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        img4 = cv2.copyMakeBorder(aug_y[i,0,0,:,:].numpy(), border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        img5 = cv2.copyMakeBorder(aug_x[i,0,3,:,:].numpy(), border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        img6 = cv2.copyMakeBorder(aug_x[i,0,4,:,:].numpy(), border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        
        cv2.imwrite(os.path.join(output_folder1, f'img_{i}_0.png'),img1)
        cv2.imwrite(os.path.join(output_folder1, f'img_{i}_1.png'),img2)
        cv2.imwrite(os.path.join(output_folder1, f'img_{i}_2.png'),img3)
        cv2.imwrite(os.path.join(output_folder1, f'img_{i}_3.png'),img4)
        cv2.imwrite(os.path.join(output_folder1, f'img_{i}_4.png'),img5)
        cv2.imwrite(os.path.join(output_folder1, f'img_{i}_5.png'),img6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation Script")
    parser.add_argument("--input_file", type=str, help="Path to the input HDF file")
    parser.add_argument("--output_dir", type=str, help="Directory for saving augmented data")
    parser.add_argument("--size", type=int, help="Size of the augmented data")
    parser.add_argument("--border", type=int)
    parser.add_argument("--random_seed", type=int, default=999, help="Random seed for reproducibility")

    args = parser.parse_args()
    aug_data(args.input_file, args.output_dir, args.size, args.border, args.random_seed)
    
    