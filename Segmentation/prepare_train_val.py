import random
from prepare_data import real_data_path,semisyn_train_data_path,fullysyn_train_data_path
import os
from pathlib import Path
import utils

def get_split(dataset, fold, add_real='False', change_test_set='False'):
    print('change_test_set',change_test_set)
    train_file_names = []
    val_file_names = []
    

    ###instrumennt segmentation
    
    # Real dataset
    if dataset == 'real':
        train_file_names = list((real_data_path/'train/images').glob('*.png'))
        val_file_names = list((real_data_path/'val/images').glob('*.png'))

    # Semi-Part dataset
    if dataset == 'semi_part':
        for instrument_id in range(1,7):
            train_file_names += list((semisyn_train_data_path/('video'+str(instrument_id))/'images').glob('*.png'))
        val_file_names += list((real_data_path/'val/images').glob('*.png'))

    # Semi-Full dataset
    elif dataset == 'semi_full':
        for instrument_id in range(1,8):
            train_file_names += list((semisyn_train_data_path/('video'+str(instrument_id))/'images').glob('*.png'))
        val_file_names += list((real_data_path/'val/images').glob('*.png'))

    # Add 50 real images to train set
    if add_real == 'True':
        utils.seed_everything(3047)
        real_image_names = list((real_data_path/'train/images').glob('*.png'))
        real_images_added = random.sample(real_image_names,50)
        train_file_names += real_images_added

    ###bleeding source segmentation

    # Fully-Synthetic dataset
    if dataset == 'fully_synthetic':
        folds = {0: [7,8,15,17]}
        for instrument_id in range(1,18):
            if instrument_id in folds[fold]:
                val_file_names += list((fullysyn_train_data_path/('video'+str(instrument_id))/'image').glob('*.png'))
            else:
                train_file_names += list((fullysyn_train_data_path/('video'+str(instrument_id))/'image').glob('*.png'))
    
    # Change Fully-synthetic test set with real test set
    if change_test_set == 'True':
        val_file_names = []
        val_data_path = real_data_path
        val_file_names += list((val_data_path/'image').glob('*jpg'))


    return train_file_names, val_file_names

if __name__ == '__main__':
    train_file_names, val_file_names = get_split('fully_synthetic',0)
    print('train_file_names', len(train_file_names))
    print('val_file_names', len(val_file_names))
    # train_file_names 6253
    # val_file_names 1924
