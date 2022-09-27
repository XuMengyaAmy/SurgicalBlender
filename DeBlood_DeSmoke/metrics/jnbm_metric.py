'''
This code is simply a python version of C code based implementation of the JNBM metric by David Roberts : https://github.com/davidatroberts/No-Reference-Sharpness-Metric
'''


import cv2
import math

def _local_extrema(row,x_pos):
    x_value = row[x_pos]
    last_x_value = x_value
    right_inc_width = 0
    for x in range(x_pos,len(row)):
        if x != x_pos:
            current_value = row[x]
            if current_value > last_x_value:
                right_inc_width+=1
                last_x_value=current_value
            else:
                break
    last_x_value = x_value
    right_dec_width = 0
    for x in range(x_pos,len(row)):
        if x!= x_pos:
            current_value = row[x]
            if current_value < last_x_value:
                right_dec_width+=1
                last_x_value = current_value
            else:
                break
    last_x_value = x_value
    left_inc_width = 0
    for x in range(x_pos,-1,-1):
        if x!= x_pos:
            current_value = row[x]
            if current_value > last_x_value:
                left_inc_width+=1
                last_x_value=current_value
            else:
                break
    last_x_value = x_value
    left_dec_width = 0
    for x in range(x_pos,-1,-1):
        if x!= x_pos:
            current_value = row[x]
            if current_value < last_x_value:
                left_dec_width+=1
                last_x_value=current_value
            else:
                break
    right_width = 0
    left_width = 0
    if right_inc_width > right_dec_width:
        right_width = right_inc_width
        left_width = left_dec_width
    elif right_inc_width < left_inc_width:
        right_width = right_dec_width
        left_width = left_inc_width
    else:
        right_width = right_inc_width
        left_width = left_inc_width if left_inc_width > left_dec_width else left_dec_width

    return left_width+right_width

def _get_block_distortion(edges, w_jnb, beta):
    distortion = 0
    for i in range(len(edges)):
        width = edges[i]
        ratio = width/w_jnb
        edge_distortion = math.pow(ratio, beta)
        distortion += edge_distortion
    distortion = math.pow(distortion, 1/beta)
    return distortion

def _get_image_distortion(block_distortions, beta):
    image_distortion = 0
    for i in range(len(block_distortions)):
        dist = block_distortions[i]
        single_dist = math.pow(math.fabs(dist),beta)
        image_distortion += single_dist
    image_distortion = math.pow(image_distortion,1/beta)
    return image_distortion

def _get_sharpness_measure(distortion, processed_blocks):
    return processed_blocks/distortion #if distortion != 0 else 0

def _get_bluriness_measure(distortion, processed_blocks):
    return distortion/processed_blocks #if processed_blocks != 0 else 0

# def _probability_detecting_blur(distortion, beta):
#     return (1 - math.exp(math.pow(-distortion, beta)))

def _local_contrast(chunk):
    min_val,max_val,_,_ = cv2.minMaxLoc(chunk)
    return max_val-min_val

def _jnb_edge_width(contrast):
    return 5 if contrast <= 50 else 3


def JNBM(image_path):
    T = 0.002
    CHUNK_SIZE = 64
    BETA = 3.6
    src_img = cv2.imread(image_path, 1)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    lum_8 = cv2.convertScaleAbs(src_img)
    sobel = cv2.Sobel(lum_8,cv2.CV_16S,dx=1,dy=0,ksize=3,scale=1,delta=0)
    sobel_abs = cv2.convertScaleAbs(sobel)
    processed_blocks = 0
    threshold = T *CHUNK_SIZE*CHUNK_SIZE
    im_size = sobel.shape
    x_chunks = math.ceil(im_size[1]/CHUNK_SIZE)
    y_chunks = math.ceil(im_size[0]/CHUNK_SIZE)

    block_distortions = []
    for x in range(0, x_chunks):
        cx = x*CHUNK_SIZE
        for y in range(0, y_chunks):
            cy = y*CHUNK_SIZE
            edge_chunk = sobel_abs[cy:cy+CHUNK_SIZE, cx:cx+CHUNK_SIZE]
            edge_ct = 0
            for i in range(edge_chunk.shape[0]):
                for j in range(edge_chunk.shape[1]):
                    if edge_chunk[i,j] > 0:
                        edge_ct+=1
            if edge_ct > threshold:
                processed_blocks+=1
                lum_chunk = lum_8[cy:cy+CHUNK_SIZE,cx:cx+CHUNK_SIZE]
                contrast = _local_contrast(lum_chunk)
                jnb_width = _jnb_edge_width(contrast)
                edge_widths = []
                for i in range(edge_chunk.shape[0]):
                    row = edge_chunk[i]
                    for j in range(edge_chunk.shape[1]):
                        if edge_chunk[i,j] > 0:
                            edge_width = _local_extrema(row,j)
                            edge_widths.append(edge_width)
                block_dist = _get_block_distortion(edge_widths,jnb_width, BETA)
                block_distortions.append(block_dist)

    total_distortion = _get_image_distortion(block_distortions, BETA)
    sharp_distortion = _get_sharpness_measure(total_distortion,processed_blocks)
    blur_distortion = _get_bluriness_measure(total_distortion,processed_blocks)
    return total_distortion, blur_distortion, sharp_distortion