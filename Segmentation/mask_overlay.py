from matplotlib import pylab
from pylab import *
import cv2
from dataset import load_image
import torch
from utils import cuda
from generate_masks_mengya import get_model
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import img_to_tensor
import numpy as np
from PIL import Image
import os
import tqdm
import pathlib

rcParams['figure.figsize'] = 10, 10

factor = 1

color  = {
        0*factor: [0, 0, 0],
        1*factor: [144, 238, 144], # light green
        # 1*factor: [0,255,0], #green color
        2*factor: [153, 255, 255], # light blue
        3*factor: [0, 102, 255], # dark blue
        4*factor: [255, 55, 0], # red
        5*factor: [0, 153, 51], # dark green
        6*factor: [187, 155, 25], # khaki
        7*factor: [255, 204, 255], # pink
        8*factor: [255, 255, 125], # light yellow
        9*factor: [123, 15, 175], # purple
        10*factor: [124, 155, 5],
        11*factor: [125, 255, 12],
        12*factor: [218,112,214],#orchid
        13*factor: [100, 149, 237], #cornflowblue
        14*factor: [255,228,196], #bisque
        15*factor: [250,128,114], #salmon
        16*factor: [0,206,209], #darkturquoise
}


def mask_overlay(image_src_path, src_path, dst_path):
    """
    Helper function to visualize mask on the top of the car
    """
    image = load_image(image_src_path)
    mask = Image.open(src_path)
    mask = np.asarray(mask) 
    print('mask', mask.shape) # (224, 224)  # mask (1024, 1280)
    print('unique', np.unique(mask)) #.astype(np.uint8)
    # mask = np.dstack((mask, mask, mask)) * np.array(color)
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color_mask[i, j] = color[mask[i, j]]
    
    color_mask = Image.fromarray(color_mask.astype(np.uint8))
    newsize = (1920,1080) # (1280, 1024), (1024, 1280)
    color_mask = color_mask.resize(newsize)
    print('---color_mask',color_mask)
    print('---image.shape',image.shape)
    color_mask = np.asarray(color_mask)
    weighted_sum = cv2.addWeighted(color_mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = color_mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind] 
    img = Image.fromarray(img.astype(np.uint8))
    img.save(dst_path)
    return img




# # # 18 predict
# mask_src_base_path = ['/mnt/disk1_ssd/mengya/new-semi-robot-surgery-segmentation/predict_mask_jieming/semisyn_withReal_dataset/DeepLabv3_plus/val/',]
# image_src_base_path = ['/mnt/disk2_hdd/mengya/2018_RoboticSceneSegmentation/ISINet_Train_Val/val/images/',]
# dst_base_path = ['/mnt/disk1_ssd/mengya/new-semi-robot-surgery-segmentation/predict_mask_jieming/semisyn_withReal_dataset/DeepLabv3_plus/overlay/',]



# # 18 predict
mask_src_base_path = ['/mnt/disk1_ssd/mengya/3DTool-robot-surgery-segmentation/predict_mask_jieming/binary_syn_dataset/DeepLabv3_plus/video15_demo_masks/',]
image_src_base_path = ['/mnt/disk2_hdd/mengya/surgical_blender/simulation_dataset_3DTool/video15/image_demo/',]
dst_base_path = ['/mnt/disk1_ssd/mengya/3DTool-robot-surgery-segmentation/predict_mask_jieming/binary_syn_dataset/DeepLabv3_plus/video15_mask_overlay/',]


# # # 18 predict
# src_base_path = ['/mnt/disk1_ssd/mengya/2D-robot-surgery-segmentation/predict_mask/SwinUnet/val/',]
# dst_base_path = ['/mnt/disk1_ssd/mengya/2D-robot-surgery-segmentation/predict_mask/SwinUnet/predict_mask_RGB/',]

print("Start🙂")
for i in range(len(dst_base_path)):
    if not os.path.isdir(dst_base_path[i]):
        pathlib.Path(dst_base_path[i]).mkdir(parents=True, exist_ok=True)
    mask_names = os.listdir(mask_src_base_path[i])
    print(len(mask_names))
    for j in range(len(mask_names)):
        image = image_src_base_path[i] + mask_names[j]
        src_mask = mask_src_base_path[i] + mask_names[j]
        dst_mask = dst_base_path[i] + mask_names[j]
        mask_overlay(image,src_mask, dst_mask)
print("Done😀")


