'''
To detect the landmarks and save into the json dict
'''

from tomlkit import key
import torch 
import torch.nn as nn
from model.keypoint_detection import KeypointDetection
from tqdm import tqdm
import numpy as np 
import pandas as pd 
import os
import argparse
import torch.nn.functional as F
import torchio as tio
from glob import glob 
from skimage.measure import label, regionprops
import time
import json

plane_keypoint_names = [
    "Or R",
    "Or L",
    "Po R",
    "Po L",
    "C-R",
    "C-L",
    "N",
    "S"
]

roi_keypoint_names = [
    'Cornoid-R',
    'Condylar-posterior-R',
    'Cornoid-L',
    'Condylar-posterior-L',
    'Go-R',
    'Go-L'
]

def crop_3d_centroid_move(img, centroid, patch_size):
    
    H, W, D = img.shape

    # obtain the crop size
    h_crop_1 = centroid[0] - patch_size[0] // 2
    h_crop_2 = centroid[0] + patch_size[0] - patch_size[0] // 2

    w_crop_1 = centroid[1] - patch_size[1] // 2
    w_crop_2 = centroid[1] + patch_size[1] - patch_size[1] // 2

    d_crop_1 = centroid[2] - patch_size[2] // 2
    d_crop_2 = centroid[2] + patch_size[2] - patch_size[2] // 2 


    # correct the crop size
    if h_crop_1 < 0:
        h_crop_2 = h_crop_2 + h_crop_1 
        h_crop_1 = 0
    
    if h_crop_2 > H:
        h_crop_1 = h_crop_1 - (h_crop_2 - H)
        h_crop_2 = H
        
    if w_crop_1 < 0:
        w_crop_2 = w_crop_2 + w_crop_1 
        w_crop_1 = 0
    
    if w_crop_2 > W:
        w_crop_1 = w_crop_1 - (w_crop_2 - W)
        w_crop_2 = W

    if d_crop_1 < 0:
        d_crop_2 = d_crop_2 + d_crop_1 
        d_crop_1 = 0
    
    if d_crop_2 > D:
        d_crop_1 = d_crop_1 - (d_crop_2 - D)
        d_crop_2 = D
    
    crop_img = img[h_crop_1:h_crop_2, w_crop_1:w_crop_2, d_crop_1:d_crop_2]
    h_pad = 64 + h_crop_1 - h_crop_2
    w_pad = 64 + w_crop_1 - w_crop_2
    d_pad = 64 + d_crop_1 - d_crop_2
    
    pad_img = np.pad(crop_img, ((0, h_pad), (0, w_pad), (0, d_pad)), 'constant')
    
    return pad_img


def keypoint_detection(path1, path2, img_path, out_path, normalize=(0, 2500)):
    
    # stage 1 for plane points
    model = KeypointDetection(1, 8)
    model.load_state_dict(torch.load(path1))
    model = model.cuda()
    model.eval()
    
    # load data of stage 1
    start = time.time()
    img = tio.ScalarImage(img_path)
    target_spacing = np.array([0.48883, 0.48883, 1.])
    ori_size = np.array(img.data.shape[1:])
    
    target_size = (ori_size * np.array(img.spacing) / target_spacing).astype(int)
    img_int = F.interpolate(img.data.unsqueeze(0).float(), size=(256, 256, 128), mode='trilinear').float(
    img_int_clamp = torch.clamp(img_int, normalize[0], normalize[1])
    img_int_norm = (img_int_clamp - img_int_clamp.min()) / (img_int_clamp.max() - img_int_clamp.min())
    

    
    # stage 1 prediction
    start = time.time()
    with torch.no_grad():
        pred = model(img_int_norm.cuda()) # (1, 8, target_size)

    stage1_centroid = {}

    pred_data = F.interpolate(pred.cpu(), size=tuple(int(x) for x in target_size), mode='trilinear').squeeze().numpy()
    pred_Or = pred_data[0] + pred_data[1]
    gate_Or = np.percentile(pred_Or, 99.995)
    pred_Or_seg = np.where(pred_Or > gate_Or, 1, 0)

    pred_Or_label = label(pred_Or_seg)
    Or_regions = [item for item in sorted(regionprops(pred_Or_label), key=lambda x: x.area, reverse=True)]
    Or_2regions = sorted(Or_regions[:2], key=lambda x: x.centroid[0])
    c1, c2 = Or_2regions[0].centroid, Or_2regions[1].centroid
    stage1_centroid[plane_keypoint_names[0]] = c1 
    stage1_centroid[plane_keypoint_names[1]] = c2

    pred_Po = pred_data[2] + pred_data[3]
    gate_Po = np.percentile(pred_Po, 99.995)
    pred_Po_seg = np.where(pred_Po > gate_Po, 1, 0)
    pred_Po_label = label(pred_Po_seg)
    Po_regions = [item for item in sorted(regionprops(pred_Po_label), key=lambda x: x.area, reverse=True)]
    Po_2regions = sorted(Po_regions[:2], key=lambda x: x.centroid[0])
    c3, c4 = Po_2regions[0].centroid, Po_2regions[1].centroid
    stage1_centroid[plane_keypoint_names[2]] = c3
    stage1_centroid[plane_keypoint_names[3]] = c4

    pred_C = pred_data[4] + pred_data[5]
    gate_C = np.percentile(pred_C, 99.995)
    pred_C_seg = np.where(pred_C > gate_C, 1, 0)

    pred_C_label = label(pred_C_seg)
    C_regions = [item for item in sorted(regionprops(pred_C_label), key=lambda x: x.area, reverse=True)]
    C_2regions = sorted(C_regions[:2], key=lambda x: x.centroid[0])
    c5, c6 = C_2regions[0].centroid, C_2regions[1].centroid
    stage1_centroid[plane_keypoint_names[4]] = c5
    stage1_centroid[plane_keypoint_names[5]] = c6

    pred_N = pred_data[6]
    gate_N = np.percentile(pred_N, 99.995)
    pred_N_seg = np.where(pred_N > gate_N, 1, 0)
    pred_N_label = label(pred_N_seg)
    N_regions = [item for item in sorted(regionprops(pred_N_label), key=lambda x: x.area, reverse=True)]
    c7 = N_regions[0].centroid
    stage1_centroid[plane_keypoint_names[6]] = c7

    pred_S = pred_data[7]
    gate_S = np.percentile(pred_S, 99.995)
    pred_S_seg = np.where(pred_S > gate_S, 1, 0)
    pred_S_label = label(pred_S_seg)
    S_regions = [item for item in sorted(regionprops(pred_S_label), key=lambda x: x.area, reverse=True)]    
    
    cs = [c1, c2, c3, c4, c5, c6, c7]
    for region in S_regions:
        ps = True 
        for c in cs:
            if np.linalg.norm(np.array(region.centroid) - np.array(c)) < 100:
                ps = False 
                break 
        
        if ps:
            stage1_centroid[plane_keypoint_names[7]] = region.centroid 
            break         

    print('------------Stage 1 Postprocess Time {:.2f}------------'.format(time.time() - start))
    
    # stage 2 preprocess data
    stage2_centroids = {}
    
    # stage 1 for plane points
    model = KeypointDetection(1, 4)
    model.load_state_dict(torch.load(path2))
    model = model.cuda()
    model.eval()          

    with torch.no_grad():
        pred = model(img_int_norm.cuda()) # (1, 4, target_size)      
    
    
    start = time.time()
    pred_data = F.interpolate(pred.cpu(), size=tuple(int(x) for x in target_size), mode='trilinear').squeeze().numpy()


    # process points of Cornoid-R, Condylar-posterior-R
    pred_L = pred_data[0]
    gate = np.percentile(pred_L, 99.995)
    pred_seg = np.where(pred_L > gate, 1, 0)

    pred_label = label(pred_seg)
    regions = [item for item in sorted(regionprops(pred_label), key=lambda x: x.area, reverse=True)]
    fb_regions = sorted(regions[:2], key=lambda x:x.centroid[1])
    c1, c2 = fb_regions[0].centroid, fb_regions[1].centroid
        
    stage2_centroids[roi_keypoint_names[0]] = c1 
    stage2_centroids[roi_keypoint_names[1]] = c2

    # process points of C-R and C-L
    pred_R = pred_data[1]
    gate = np.percentile(pred_R, 99.995)
    pred_seg = np.where(pred_R > gate, 1, 0)

    pred_label = label(pred_seg)
    regions = [item for item in sorted(regionprops(pred_label), key=lambda x: x.area, reverse=True)]
    fb_regions = sorted(regions[:2], key=lambda x:x.centroid[1])
    c3, c4 = fb_regions[0].centroid, fb_regions[1].centroid
        
    stage2_centroids[roi_keypoint_names[2]] = c3
    stage2_centroids[roi_keypoint_names[3]] = c4    

    
    pred_Go = pred_data[2] + pred_data[3]
    gate = np.percentile(pred_Go, 99.995)
    pred_seg = np.where(pred_Go > gate, 1, 0)

    pred_label = label(pred_seg)
    regions = [item for item in sorted(regionprops(pred_label), key=lambda x: x.area, reverse=True)]
    fb_regions = sorted(regions[:2], key=lambda x:x.centroid[0])
    c5, c6 = fb_regions[0].centroid, fb_regions[1].centroid

    stage2_centroids[roi_keypoint_names[4]] = c5
    stage2_centroids[roi_keypoint_names[5]] = c6            
        
    # final postprocess

    out_dict = {}
    for k, v in stage1_centroid.items():
        centroid = np.array(v)
        ori_centroid = centroid * ori_size / target_size 
        out_dict[k] = (ori_centroid[0], ori_centroid[1], ori_centroid[2])

    for k, v in stage2_centroids.items():
        centroid = np.array(v)
        ori_centroid = centroid * ori_size / target_size 
        try:
            out_dict[k] = (ori_centroid[0], ori_centroid[1], ori_centroid[2])       
            # print(k,ori_centroid)
        except:
            pass


    predict_time = time.time() - start
    print('------------Stage 2 Postprocess Time {:.2f}------------'.format(time.time() - start))

    with open(out_path, 'w') as f: 
        json.dump(out_dict, f, indent=4)



path1 = r'code\checkpoints\keypoint_detection\stage1.ckpt'
path2 = r'code\checkpoints\keypoint_detection\stage2.ckpt'


time_points = ['T1', 'T2']
dir_path = r'code\data'
IDs = os.listdir(dir_path)
total_time = []
total_case = 0

for ID in IDs:
    for time_point in time_points:
        img_path = os.path.join(dir_path, ID, f'{time_point}_img.nii.gz')
        img = tio.ScalarImage(img_path).data.numpy()
        normalize = (1024, 3524)
        if np.max(img) < 3500:
            normalize = (0, 2500)
        else:
            normalize = (1024, 3524)
        out_path = f'code/data/{ID}'
        out_file_path = os.path.join(out_path, f'{time_point}_keypoints.json')
        keypoint_detection(path1, path2, img_path, out_file_path, normalize=normalize)








            

        
    
    
    
    
