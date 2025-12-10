import argparse
import glob
import numpy as np
import os
import time
import csv
import cv2
from LPM import LPM_filter
import random
from scipy import linalg
import multiprocessing
from top_superPoint import *



def SIFT(im1, mask1=None):
    sift = cv2.SIFT_create()
    kp1, dsp1 = sift.detectAndCompute(im1, mask1)  # None --> mask
    return kp1, dsp1


def flann_match(dsp1,dsp2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(dsp1, dsp2, k=2)
    good = np.zeros((len(dsp1),len(dsp2)))
    for m, n in matches:
        good[m.queryIdx, m.trainIdx]=1
    return good

def bi_match(kp1,kp2,dsp1,dsp2):
    match1=flann_match(dsp1,dsp2)
    match2=flann_match(dsp2,dsp1)
    match2=match2.T
    row_indices, col_indices = np.where((match1+match2) == 2)
    srcdsp = kp1[row_indices].astype(np.float32)
    tgtdsp = kp2[col_indices].astype(np.float32)

    return srcdsp,tgtdsp

def calculate_H(X_ok,Y_ok):
    point_num = X_ok.shape[0]
    centroid_1 = np.mean(X_ok, axis=0)
    centroid_2 = np.mean(Y_ok, axis=0)
    XX = X_ok - np.tile(centroid_1, (point_num, 1))
    YY = Y_ok - np.tile(centroid_2, (point_num, 1))
    H_ = np.dot(XX.T, YY)
    U, S, VT = np.linalg.svd(H_)
    R = np.dot(VT.T, U.T)
    if np.linalg.det(R) < 0:
        VT[1, :] *= -1
        R = np.dot(VT.T, U.T)
    t = -np.dot(R, centroid_1) + centroid_2
    H = np.zeros((3, 3))
    H[:2, 2] = t
    H[:2, :2] = R
    H[2,2] = 1
    H_inv=np.linalg.inv(H)
    
    return H_inv
            
def load_image(path, gray=True):
    if gray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return cv2.imread(path)
    
def load_mask(path,th,gray=True):
    if gray:
        mask = cv2.imread(path)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask = cv2.imread(path)
    mask[mask>=th]=255
    mask[mask<th]=0
    return mask

def filter_by_mask(pts, dsp, mask):
    col = pts[:, 0].astype(int)
    row = pts[:, 1].astype(int)
    mask_values = mask[row, col]
    mask_condition = (mask_values != 0)
    valid_condition = mask_condition
    edge_pts = pts[valid_condition]
    edge_dsp = dsp[valid_condition]

    return edge_pts, edge_dsp
    
def apply_homography(points, H):
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.concatenate((points, ones), axis=1)
    transformed_points_homogeneous = np.dot(H, points_homogeneous.T).T
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]
    return transformed_points

def calculate_distance_error(X, Y, H):
    transformed_Y = apply_homography(Y, H)
    distance_error = np.sqrt(np.sum((X - transformed_Y)**2, axis=1)).mean()
    return distance_error



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
        help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--skip', type=int, default=1,
        help='Images to skip if input is movie or directory (default: 1).')
    parser.add_argument('--min_length', type=int, default=2,
        help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=5,
        help='Maximum length of point tracks (default: 5).')
    parser.add_argument('--nms_dist', type=int, default=4,
        help='Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0.015,
        help='Detector confidence threshold (default: 0.015).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
        help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--cuda', type=bool,default=True)
    
    parser.add_argument('--use_mask', type=int, default=1)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--input_dir', type=str,default='./data/serial')
    parser.add_argument('--input_mask', type=str, default='')
    parser.add_argument('--output_dir', type=str,default='./data/serial_result')
    parser.add_argument('--use_ransac', type=int,default=0)
    
    
    args = parser.parse_args()
    
    
    if not os.path.exists(args.output_dir):            
        os.makedirs(args.output_dir) 

    fe = SuperPointFrontend(weights_path=args.weights_path,
                            nms_dist=args.nms_dist,
                            conf_thresh=args.conf_thresh,
                            nn_thresh=args.nn_thresh,
                            cuda=args.cuda)
    
    img_list = np.sort(os.listdir(args.input_dir))
    
    if args.use_mask:
        mask_list = np.sort(os.listdir(args.input_mask))
        LastMask=None
    
    for i,_ in enumerate(img_list[:-1]):
        if i==0:
            img1 = os.path.join(args.input_dir,img_list[i])
            img2 = os.path.join(args.input_dir,img_list[i+1])
            if args.use_mask:
                mask_name1 = os.path.join(args.input_mask, mask_list[i])
                mask_name2 = os.path.join(args.input_mask, mask_list[i+1])
                mask1=load_mask(mask_name1,20)
                mask2=load_mask(mask_name2,20)  
        else:
            img1 = os.path.join(args.output_dir,img_list[i])
            img2 = os.path.join(args.input_dir,img_list[i+1])
            if args.use_mask:
                mask_name2 = os.path.join(args.input_mask, mask_list[i+1])  
                mask1=LastMask
                mask2=load_mask(mask_name2,20)
        img1 = load_image(img1,gray=True)
        img2 = load_image(img2,gray=True)
        if i==0 :
            cv2.imwrite(os.path.join(args.output_dir,img_list[i]),img1)
        
        
        pts1, desc1, _ = fe.run(img1.astype('float32') / 255.)
        pts2, desc2, _ = fe.run(img2.astype('float32') / 255.)
        kp1 = pts1.T[:,:2]
        kp2 = pts2.T[:,:2]
        dsp1 = desc1.T
        dsp2 = desc2.T
        
        if args.use_mask:
            edge_kp1,edge_dsp1=filter_by_mask(kp1,dsp1,mask1)
            edge_kp2,edge_dsp2=filter_by_mask(kp2,dsp2,mask2)
        else:
            edge_kp1,edge_dsp1=kp1,dsp1
            edge_kp2,edge_dsp2=kp2,dsp2
        
        X,Y=bi_match(edge_kp1,edge_kp2,edge_dsp1,edge_dsp2)
        X_clone,Y_clone = X.copy(),Y.copy()
        
        if len(X)>10:
            ok=LPM_filter(X,Y)
            X_ok=X[ok,:]
            Y_ok=Y[ok,:]
            if len(X_ok)<4:
                X_ok,Y_ok = X,Y
        else:
            X_ok,Y_ok = X,Y
        
        if args.use_ransac and len(X_ok)<8:
            X_ok,Y_ok = X,Y

        H_sum = np.eye(3)
        H_i = calculate_H(X_ok,Y_ok)
        H_sum = np.dot(H_i, H_sum)
        errors = calculate_distance_error(X_ok,Y_ok,H_i)

        error_list,H_list = [],[]
        error_list.append(errors)
        H_list.append(H_sum)
        for iter in range(args.iters):
            Y = apply_homography(Y, H_i)
            try:
                if args.use_ransac:
                    _,ok = cv2.findHomography(X,Y,cv2.RANSAC,6.0)
                    ok = ok.reshape(-1)
                    if np.sum(ok)<6:
                        ok = np.ones_like(ok)
                    X_r = X[ok, :]
                    Y_r = Y[ok, :]
                    H_r = calculate_H(X_r, Y_r)
                    H_sum = np.dot(H_r, H_sum)
                    Y = apply_homography(Y, H_r)
                ok = LPM_filter(X, Y)
            except ValueError as e:
                print(f"Iteration {iter}: Error in LPM_filter - {e}")
                break
            X_ok = X[ok, :]
            Y_ok = Y[ok, :]
            H_i = calculate_H(X_ok, Y_ok)
            H_sum = np.dot(H_i, H_sum)
            errors = calculate_distance_error(X_ok, Y_ok, H_i)

            H_list.append(H_sum)
            error_list.append(errors)
  
        error_array = np.array(error_list)
        non_nan_indices = ~np.isnan(error_array)
        filtered_errors = error_array[non_nan_indices]
        idx_min_filtered = np.argmin(filtered_errors)
        idx = np.where(non_nan_indices)[0][idx_min_filtered]

        H_sum = H_list[idx]
        error = error_list[idx]

        rows,cols=img2.shape
        img2_warp=cv2.warpAffine(img2,H_sum[:2,:],(cols, rows))
        cv2.imwrite(os.path.join(args.output_dir,img_list[i+1]),img2_warp)
        
        if args.use_mask:
            LastMask=cv2.warpAffine(mask2,H_sum[:2,:],(cols,rows))
            
            
        
        
    

    