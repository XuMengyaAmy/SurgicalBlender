"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_split
from dataset import RoboticsDataset
import cv2
from models import UNet16, LinkNet34, UNet11, UNet, AlbuNet, DeepLabv3_plus
# =============================== #
from networks.vision_transformer import SwinUnet 
from config import get_config
# =============================== #
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
import prepare_data
from torch.utils.data import DataLoader
from torch.nn import functional as F
# from prepare_data import (original_height,
#                           original_width,
#                           h_start, w_start
#                           )
from albumentations import Compose, Normalize


# def img_transform(p=1):
#     return Compose([
#         Normalize(p=1)
#     ], p=p)

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    Resize
)
# def img_transform(p=1):
#     return Compose([
#         PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
#         CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
#         Normalize(p=1)
#     ], p=p)

# def img_transform(p=1):
#     return Compose([
#         Resize(args.val_crop_height, args.val_crop_width, always_apply=True, p=1),
#         Normalize(p=1)
#     ], p=p)

def img_transform(p=1):
    return Compose([
        Resize(args.val_crop_height, args.val_crop_width, always_apply=True, p=1),
        Normalize(p=1)
    ], p=p)




def get_model(model_path, model_type='UNet11', problem_type='binary'):
    """

    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34', 'AlbuNet'
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """
    if problem_type == 'binary':
        num_classes = 1
    elif problem_type == 'parts':
        num_classes = 4
    elif problem_type == 'instruments':
        num_classes = 8

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)
    elif model_type == 'DeepLabv3_plus':
        model = DeepLabv3_plus(num_classes=num_classes, pretrained=True)
    elif model_type == 'SwinUnet':
        config = get_config(args)
        model = SwinUnet(config, img_size=args.img_size, num_classes=num_classes)
        model.load_from(config)


    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model

def predict(model, from_file_names, batch_size, to_path, problem_type, img_transform):
    loader = DataLoader(
        dataset=RoboticsDataset(from_file_names, transform=img_transform, mode='predict', problem_type=problem_type),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)

            outputs = model(inputs)

            for i, image_name in enumerate(paths):
                if problem_type == 'binary':
                    factor = prepare_data.binary_factor
                    t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(np.uint8)
                elif problem_type == 'parts':
                    factor = prepare_data.parts_factor
                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)
                elif problem_type == 'instruments':
                    factor = prepare_data.instrument_factor
                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)

                h, w = t_mask.shape

                # full_mask = np.zeros((original_height, original_width))
                # full_mask[h_start:h_start + h, w_start:w_start + w] = t_mask

                instrument_folder = Path(paths[i]).parent.parent.name

                (to_path / instrument_folder).mkdir(exist_ok=True, parents=True)

                cv2.imwrite(str(to_path / instrument_folder / (Path(paths[i]).stem + '.png')), t_mask)#full_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # arg('--model_path', type=str, default='data/models/UNet', help='path to model folder')
    arg('--model_path', type=str, default='./runs/LinkNet34/', help='path to model folder')
    arg('--model_type', type=str, default='LinkNet34', help='network architecture',
        choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34', 'AlbuNet','SwinUnet', 'DeepLabv3_plus'])
    # arg('--output_path', type=str, help='path to save images', default='1')
    arg('--output_path', type=str, help='path to save images', default='predict_mask/LinkNet34')

    arg('--batch-size', type=int, default=4)
    # arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, -1], help='-1: all folds')
    arg('--fold', type=int, default=0, choices=[0, 1, 2, 3, -1], help='-1: all folds')
    arg('--problem_type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=12)

    # For our models
    arg('--train_crop_height', type=int, default=224) 
    arg('--train_crop_width', type=int, default=224) 
    arg('--val_crop_height', type=int, default=224)
    arg('--val_crop_width', type=int, default=224)
    arg('--img_size', type=int, default=224, help='input patch size of network input')
    arg('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )

    args = parser.parse_args()
    print('================')
    print('================')
    arg('--add_real', type=str, default='False')

    if args.fold == -1:
        for fold in [0, 1, 2, 3]:
            _, file_names = get_split(fold)
            model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=fold))),
                              model_type=args.model_type, problem_type=args.problem_type)

            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path)
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type,
                    img_transform=img_transform(p=1))
    else:
        _, file_names = get_split(args.fold)
        # model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=args.fold))),
        #                   model_type=args.model_type, problem_type=args.problem_type)

        model = get_model(str(Path(args.model_path).joinpath('best_model.pt')),
                          model_type=args.model_type, problem_type=args.problem_type)

        print('num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type,
                img_transform=img_transform(p=1))
