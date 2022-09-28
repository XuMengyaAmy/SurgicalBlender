# SurgicalBlender
Project page is available from [here](https://sites.google.com/view/surgicalblender/home) for video demonstration.

# Investigation of the ability to remove smoke and blood in 3D surgical images using unpaired and paired data

<img src = "DeBlood_DeSmoke/imgs/VisualComparison.png" width=961>

## Model train/test
### Install
```pip install -r requirements.txt```
### Datasets

Use our <a href="">dataset</a> own dataset by creating the appropriate folders and adding in the images.
- Create a dataset folder under `/dataset` for your dataset.
- Create subfolders `testA`, `testB`, `trainA`, and `trainB` under your dataset's folder. For example, place clear images in the `trainA` folder, hazy images in the `trainB` folder, and do the same for the `testA` and `testB` folders.

### Pretrained models

Add the provided model under `/pretrained/[NAME]` to `./checkpoints/[NAME]/latest_net_G.pt` (Pretrained models can be found [here](https://drive.google.com/drive/folders/1BScWxXajVd1JmN6TxylAscsOfZN8pgtM?usp=sharing).)

Or add your own pretrained model to `./checkpoints/{NAME}/latest_net_G.pt`

### Training


Change the `--dataroot` and `--name` to your own dataset's path and model's name. Use `--gpu_ids 0,1,..` to train on multiple GPUs and `--batch_size` to change the batch size. I've found that a batch size of 16 with total images of around 6.5k fits onto 2 RTX 3090 and can finish training an epoch in ~300s.

Once your model has trained, copy over the last checkpoint to a format that the testing model can automatically detect

<!-- Use `cp ./checkpoints/deblood_iter1/latest_net_G_A.pth ./checkpoints/deblood_iter1/latest_net_G.pth` if you want to transform images from class A to class B and `cp ./checkpoints/deblood_iter1/latest_net_G_B.pth ./checkpoints/deblood_iter1/latest_net_G.pth` if you want to transform images from class B to class A. -->

Below are different commands with the configs as used in our training & testing:
- ```python train.py --dataroot /mnt/disk2_hdd/User/"DeSmoke-LAP dataset"/Simul_3D --n_epochs 75 --n_epochs_decay 50 --name deblood_iter1 --model cycle_gan --batch_size 16 --display_id -1 --gpu_ids 0,1 --preprocess scale_width_and_crop```
- ```!python train.py --dataroot /mnt/disk2_hdd/User/"DeSmoke-LAP dataset"/Simul_3D/pix --n_epochs 75 --n_epochs_decay 50 --name deblood_iter3_paired --model template --batch_size 128 --display_id -1 --gpu_ids 0,1 --continue_train --epoch_count 35 --dataset_mode template```

### Testing

Change the `--dataroot` and `--name` to be consistent with your trained model's configuration.

- ```python test.py --dataroot /mnt/disk2_hdd/User/"DeSmoke-LAP dataset"/Simul_3D/testA --name deblood_iter1 --model test --no_dropout --eval --num_test 1000```
- ``` python test.py --dataroot /mnt/disk2_hdd/User/"DeSmoke-LAP dataset"/Simul_3D/pix --name deblood_iter3_paired_epoch115 --dataset_mode template --model template --no_dropout --eval --num_test 1000```

> from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix:
> The option --model test is used for generating results of CycleGAN only for one side. This option will automatically set --dataset_mode single, which only loads the images from one set. On the contrary, using --model cycle_gan requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at ./results/. Use --results_dir {directory_path_to_save_result} to specify the results directory.

> For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.


## Model Architecture & Losses

The model is built based on the architecture of <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix">CycleGAN</a> network. IC(Inter Channel) & DC(Dark Channel) loss implementations are adapted from this <a href="https://github.com/yiroup20/DeSmoke-LAP"> repo</a>


## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [DeSmoke-LAP](https://github.com/yiroup20/DeSmoke-LAP)
