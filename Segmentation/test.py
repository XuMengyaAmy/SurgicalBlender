import argparse
import json
from pathlib import Path
from validation import validation_binary, validation_multi

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16, AlbuNet, DeepLabv3_plus
#========================================#
from networks.vision_transformer import SwinUnet 
from config import get_config
#========================================#
from loss import LossBinary, LossMulti
from dataset import RoboticsDataset
import utils
import sys
from prepare_train_val import get_split

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,
    Resize,
)

moddel_list = {'UNet11': UNet11,#best model for instrument mode
               'UNet16': UNet16,#best model for binary mode
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34,
               'DeepLabv3_plus': DeepLabv3_plus,
               'SwinUnet': SwinUnet}


def main():
    utils.seed_everything(3047)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--device-ids', type=str, default='0, 1', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/mask_all/UNet16', help='checkpoint root')
    arg('--batch-size', type=int, default=64)
    arg('--n-epochs', type=int, default=60)
    arg('--lr', type=float, default=0.001)
    arg('--workers', type=int, default=12)

    # arg('--train_crop_height', type=int, default=1024) # 1024, 512
    # arg('--train_crop_width', type=int, default=1280) # 640
    # arg('--val_crop_height', type=int, default=1024)
    # arg('--val_crop_width', type=int, default=1280)

    # arg('--train_crop_height', type=int, default=512)
    # arg('--train_crop_width', type=int, default=640)
    # arg('--val_crop_height', type=int, default=512)
    # arg('--val_crop_width', type=int, default=640)

    arg('--train_crop_height', type=int, default=224)
    arg('--train_crop_width', type=int, default=224)
    arg('--val_crop_height', type=int, default=224)
    arg('--val_crop_width', type=int, default=224)

    arg('--type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--model', type=str, default='UNet16', choices=moddel_list.keys())


    # ========= Swin-Unet ========== #
    arg('--img_size', type=int, default=224, help='input patch size of network input')
    arg('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
    # ============================== #

    # ======== for test ======= #
    arg('--checkpoint', type=str, required=True)
    arg('--add_real', type=str, default='False')
    arg('--dataset', type=str, default='semi_part')
    arg('--change_test_set', type=str,default='False')
    # /mnt/disk1_ssd/mengya/robot-surgery-segmentation/runs/mask_blood/UNet16/best_model.pt
    # ========================= #

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    if not utils.check_crop_size(args.train_crop_height, args.train_crop_width):
        print('Input image sizes should be divisible by 32, but train '
              'crop sizes ({train_crop_height} and {train_crop_width}) '
              'are not.'.format(train_crop_height=args.train_crop_height, train_crop_width=args.train_crop_width))
        sys.exit(0)

    if not utils.check_crop_size(args.val_crop_height, args.val_crop_width):
        print('Input image sizes should be divisible by 32, but validation '
              'crop sizes ({val_crop_height} and {val_crop_width}) '
              'are not.'.format(val_crop_height=args.val_crop_height, val_crop_width=args.val_crop_width))
        sys.exit(0)

    if args.type == 'parts':
        num_classes = 4
    elif args.type == 'instruments':
        num_classes = 8   ############### fit to our dataset inclduing background class 8+1
    else:
        num_classes = 1

    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'SwinUnet':
        config = get_config(args)
        model = SwinUnet(config, img_size=args.img_size, num_classes=num_classes)
        model.load_from(config)
    elif args.model == 'DeepLabv3_plus':
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True)
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    if args.type == 'binary':
        loss = LossBinary(jaccard_weight=args.jaccard_weight)
    else:
        loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)

    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.dataset, args.fold,args.add_real,args.change_test_set)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))
    # num train = 3848, num_val = 483

    def train_transform(p=1):
        return Compose([
            Resize(args.train_crop_height, args.train_crop_width, always_apply=True, p=1),
            PadIfNeeded(min_height=args.train_crop_height, min_width=args.train_crop_width, p=1),
            RandomCrop(height=args.train_crop_height, width=args.train_crop_width, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            Resize(args.train_crop_height, args.train_crop_width, always_apply=True, p=1),
            PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
            CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=args.type,
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type,
                               batch_size=len(device_ids))

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    if args.type == 'binary':
        valid = validation_binary
    else:
        valid = validation_multi

    # utils.train(
    #     init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
    #     args=args,
    #     model=model,
    #     criterion=loss,
    #     train_loader=train_loader,
    #     valid_loader=valid_loader,
    #     validation=valid,
    #     fold=args.fold,
    #     num_classes=num_classes,
    #     type =args.type
    # )

    utils.test(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes,
        type=args.type,
        checkpoint=args.checkpoint
    ) 


if __name__ == '__main__':
    main()
