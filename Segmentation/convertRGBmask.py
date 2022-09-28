import PIL.Image
import numpy as np
from skimage import io, data, color
from skimage import img_as_ubyte
from tqdm import tqdm
import os
import cv2
import pathlib

factor = 1

# color = {0*factor: [0, 0, 0],
#          1*factor: [255, 182, 193],
#          2*factor: [220, 20, 60],
#          3*factor: [0, 0, 255],
#          4*factor: [0, 255, 255],
#          5*factor: [0, 128, 0],
#          6*factor: [255, 165, 0],
#          7*factor: [128, 0, 128], }  # 对应标签的颜色编码


color  = {
        0*factor: [0, 0, 0],
        1*factor: [144, 238, 144], # light green
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


def visual_mask(src_path, dst_path):
    mask = PIL.Image.open(src_path)
    mask = np.asarray(mask)
    print('mask', mask.shape) # (224, 224)  # mask (1024, 1280)
    print('unique', np.unique(mask))
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color_mask[i, j] = color[mask[i, j]] # if the mask does not have  RGB channels
            # color_mask[i, j] = color[mask[i, j, 0]] # if the mask has RGB channelss
            
            # if mask[i, j,0] == 11*factor:
            #     print("predict the tumor")

    mask = PIL.Image.fromarray(color_mask.astype(np.uint8))
    # ==== Resize it into 1024*1280 === #
    newsize = (1280, 1024) # (1280, 1024), (1024, 1280)
    mask = mask.resize(newsize)
    # ================================= #
    mask.save(dst_path)


# # # 18 predict
# src_base_path = ['/mnt/disk1_ssd/mengya/2D-robot-surgery-segmentation/predict_mask/LinkNet34/val/',]
# dst_base_path = ['/mnt/disk1_ssd/mengya/2D-robot-surgery-segmentation/predict_mask/LinkNet34/predict_mask_RGB/',]



# # 18 predict
src_base_path = ['/mnt/disk1_ssd/mengya/new-semi-robot-surgery-segmentation/predict_mask_jieming/semisyn_withReal_dataset/DeepLabv3_plus/val/',]
dst_base_path = ['/mnt/disk1_ssd/mengya/new-semi-robot-surgery-segmentation/predict_mask_jieming/semisyn_withReal_dataset/DeepLabv3_plus/predict_mask_RGB/',]


# # # 18 predict
# src_base_path = ['/mnt/disk1_ssd/mengya/2D-robot-surgery-segmentation/predict_mask/SwinUnet/val/',]
# dst_base_path = ['/mnt/disk1_ssd/mengya/2D-robot-surgery-segmentation/predict_mask/SwinUnet/predict_mask_RGB/',]

print("Start🙂")
for i in range(len(dst_base_path)):
    if not os.path.isdir(dst_base_path[i]):
        pathlib.Path(dst_base_path[i]).mkdir(parents=True, exist_ok=True)
    mask_names = os.listdir(src_base_path[i])
    print(len(mask_names))
    for j in tqdm(range(len(mask_names))):
        src_mask = src_base_path[i] + mask_names[j]
        dst_mask = dst_base_path[i] + mask_names[j]
        visual_mask(src_mask, dst_mask)
print("Done😀")