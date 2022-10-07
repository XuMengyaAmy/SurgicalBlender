import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import prepare_data
from albumentations.pytorch.transforms import img_to_tensor
from prepare_train_val import get_split

# id_to_trainid = {
#     0:   0,

#     2:   1,
#     3:   2,
#     4:   3,
#     5:   4,
#     6:   5,
#     7:   6,
#     8:   7,
#     9:   8,

# }

class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type
        # self.target_transform = tv.transforms.Lambda(lambda t: t.
        #                                             apply_(lambda x: id_to_trainid.get(x))
        #                                             )

    def __len__(self):
        # print(self.file_names,'12345')
        return len(self.file_names)
    
    def __getitem__(self, idx):
        # print(self.file_names)
        img_file_name = self.file_names[idx]
        # print(img_file_name)
        image = load_image(img_file_name)
        # print('image', image.shape) # (1080, 1920, 3)
        mask = load_mask(img_file_name, self.problem_type)
        # print('mask 1', mask.shape) # mask (1080, 1920
        # print('unique label 1', np.unique(mask))

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]
        # print('mask 2', mask.shape)
        # print('unique label 2', np.unique(mask))
        # target = self.target_transform(target)  
        # print('unique label 3', np.unique(mask))

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type, change_test_set='False'):
    
    if problem_type == 'binary':
        mask_folder = 'mask_blood'
        factor = prepare_data.binary_factor
        if change_test_set == 'True':
            mask = cv2.imread(str(path).replace('image', mask_folder).replace('jpg','png'), 0) #for valiidation on real dataset
        else:
            mask = cv2.imread(str(path).replace('image', mask_folder), 0) #for trainingg on  Fully-synthetic dataset 
            #print(str(path).replace('image',mask_folder))
            
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = prepare_data.parts_factor

    elif problem_type == 'instruments':
        factor = prepare_data.instrument_factor
        mask_folder = 'annotations/instrument' # mask_all (8+1+1=10 classes), mask_no_blood (8+1=9 classes, class index is wrong)
        mask = cv2.imread(str(path).replace('images', mask_folder), 0)
    
    # print(str(path).replace('images', mask_folder))
    # print('unique label', np.unique(mask))
    return (mask / factor).astype(np.uint8)





