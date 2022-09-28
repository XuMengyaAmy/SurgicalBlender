import torch
import numpy as np
import cv2
for i in range(1,2882,6):
    mask_path = '/mnt/disk2_hdd/surgical_blender/simulation_dataset_3DTool/video1/mask_no_blood/{0:0>4}.png'.format(i)
    num = '{0:0>4}'.format(i)
    mask = cv2.imread(mask_path, 0)
    print('unique label', np.unique(mask),'--',num)
