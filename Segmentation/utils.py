import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
import tqdm


import random
import numpy as np
import os

from models import UNet16, LinkNet34, UNet11, UNet, AlbuNet, DeepLabv3_plus
# =============================== #
from networks.vision_transformer import SwinUnet 
from config import get_config
# =============================== #

def seed_everything(seed=3047):
    print('Seed everything')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x #cuda(async=True)


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None, type=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold) # folder 0 leads to model_0
    best_model_path = root / 'best_model.pt'

    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    save_best = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(best_model_path))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []

    # ======== Added by mengya =========== #
    best_average_iou = 0.0
    best_jaccard = 0.0
    best_epoch = 1
    
    # ==================================== #

    for epoch in range(epoch, n_epochs + 1):
        model.train()
        # random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)

                # print('inputs', inputs.shape)
                # print('targets', targets.shape)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            
            
            save(epoch + 1) # original one: save the last model and resume the training

            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            
            # ======================= Added by mengya ================ #
            if type == 'binary':
                jaccard = valid_metrics['jaccard_loss']
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_epoch = epoch
                    print('Saving the best model !!!!, best epoch:', best_epoch, ', best_jaccard:', best_jaccard)
                    save_best(best_epoch)
                print('best epoch so far:', best_epoch, 'best_jaccard so far:', best_jaccard)
            elif type == 'instruments':
                average_iou = valid_metrics['iou']
                if average_iou > best_average_iou:
                    best_average_iou = average_iou
                    best_epoch = epoch
                    print('Saving the best model !!!!, best epoch:', best_epoch, ', best_average_iou', best_average_iou)
                    save_best(best_epoch)
            
                print('best epoch so far:', best_epoch, 'best_average_iou so far:', best_average_iou)
            # ====================================================== #

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

    
def get_model(args,model_path, model_type='LinkNet34', problem_type='instruments'):
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

    # print('num_classes in get model', num_classes)
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

def test(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None, type=None, checkpoint=None): 

    # checkpoint = '/mnt/disk1_ssd/mengya/robot-surgery-segmentation/runs/mask_blood/UNet16/best_model.pt'
    model = get_model(args,str(Path(args.checkpoint)),
                          model_type=args.model, problem_type=args.type)
    print('load the model successful !')

    

    valid_losses = []
    tq = tqdm.tqdm(total=(len(valid_loader) * args.batch_size))
    # print('num_classes in loss', num_classes)
    valid_metrics = validation(model, criterion, valid_loader, num_classes)
    valid_loss = valid_metrics['valid_loss']
    valid_losses.append(valid_loss)

    # ======================= Added by mengya ================ #
    if type == 'binary':
        jaccard = valid_metrics['jaccard_loss']
        print('jaccard', jaccard)

    elif type == 'instruments':
        average_iou = valid_metrics['iou']
        print('average_iou', average_iou)

    # ====================================================== #
   