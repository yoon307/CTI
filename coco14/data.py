import pdb
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import PIL.Image
import cv2
import os.path
import scipy.misc
import random

import tools.imutils as imutils

##########################################################
from scipy.spatial.distance import cdist
import scipy.ndimage
import numpy as np
from matplotlib import pyplot as plt
# from skimage.segmentation import mark_boundaries
# from fast_slic import Slic
##########################################################

# img_root: coco/train2014 or coco/val2014
# img_id_list_file: coco/train.txt or coco/val.txt
SEG_FOLDER_NAME = "SegmentationClass"
COCO_VERSION = '2014'
NUM_CAT = 80

def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()

def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load('coco/cls_labels.npy',allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]

def get_img_path(img_name, img_root, img_subdir=None, ext='.jpg'):
    if img_subdir is not None:
        return os.path.join(img_root, img_subdir, img_name + ext)
    else:
        return os.path.join(img_root, img_name + ext)

def load_image_label_list_from_segmentation(img_name_list, segment_root):
    from tqdm import tqdm

    img_list = []
    img_append = img_list.append
    onehot = np.eye(NUM_CAT+1)[:,1:]
    print(onehot.shape)
    for img_name in tqdm(img_name_list):
        label = np.unique(np.array(PIL.Image.open(get_img_path(img_name, segment_root, img_name.split('_')[1], ext='.png'))))
        label = label[~((label == 255) + (label == 0))]
        img_append(np.sum(onehot[label], axis=0))
    return img_list


class COCOImageDataset(Dataset):

    def __init__(self, img_id_list_file, img_root, task, transform=None):
        assert task in ['train', 'val']
        
        self.img_name_list = load_img_id_list(img_id_list_file)

        #
        # self.img_name_list = ['./coco/'+img_name[8:] for img_name in self.img_name_list]
        #

        self.img_root = img_root
        self.img_subdir = task + COCO_VERSION
        self.task = task
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.img_root, self.img_subdir)).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        return name, img

class COCOClsDataset(COCOImageDataset):

    def __init__(self, img_id_list_file, coco_root, task, transform=None):
        super().__init__(img_id_list_file, coco_root, task, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
    
    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label


class COCOMocoDataset(COCOClsDataset):
    
    def __init__(self, args, img_id_list_file, img_root, task = 'train', transform=None):
        assert task in ['train', 'val']
        super().__init__(img_id_list_file, img_root, task, transform)

        # Get augmentation parameters from args
        self.crf_flag = False #args.use_crf
        self.resize_range = args.resize # Default : 0.2, 1.0
        self.crop_size = args.crop # Default : 224
        self.cj = args.cj # Default : 0.4, 0.4, 0.4, 0.1

        if self.crf_flag:
            # TODO: Should I make crf? - Yes
            self.crf_list = ['./crf/'+ name + '.npy' for name in self.img_name_list]

        augmentation = [transforms.Resize((self.crop_size, self.crop_size)),
                        torchvision.transforms.RandomApply([transforms.ColorJitter(self.cj[0], self.cj[1], self.cj[2], self.cj[3])], p=0.8),
                        transforms.RandomGrayscale(p=0.01),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                    ]

        self.hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.tf_moco = torchvision.transforms.Compose(augmentation)

    def tf_at_once(self, img, crf, seg=None):
        rrcrop = transforms.RandomResizedCrop(self.crop_size, scale=(self.resize_range[0], self.resize_range[1]), interpolation=transforms.InterpolationMode.NEAREST)
        crop_params = rrcrop.get_params(img, scale=(self.resize_range[0], self.resize_range[1]), ratio=(3.0/4.0, 4.0/3.0))
        isflip =random.randint(0,1)

        img_ = TF.crop(img, *crop_params)
        crf_ = crf[crop_params[0]:crop_params[0]+crop_params[2], crop_params[1]:crop_params[1]+crop_params[3]]
        crf_ = cv2.resize(crf_, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

        if isflip==0:
            img_ = self.hflip(img_)
            crf_ = cv2.flip(crf_, 1)

        img_ = self.tf_moco(img_)
        crf_mc_ = np.zeros((80, self.crop_size, self.crop_size))

        for i in range(80):
            crf_mc_[i][np.where(crf_==i+1)] = 1

        return img_, crf_mc_
    
    def tf_img_only(self, img):
        rrcrop = transforms.RandomResizedCrop(self.crop_size, scale=(self.resize_range[0], self.resize_range[1]), interpolation=transforms.InterpolationMode.NEAREST)
        crop_params = rrcrop.get_params(img, scale=(self.resize_range[0], self.resize_range[1]), ratio=(3.0/4.0, 4.0/3.0))
        isflip = random.randint(0,1)

        img = TF.crop(img, *crop_params)
        if isflip==0:
            img = self.hflip(img)
        img = self.tf_moco(img)
        
        return img

    def __getitem__(self, idx):
        retval = super().__getitem__(idx)

        out = {}
        name, img, label = retval

        if self.crf_flag:
            crf = np.load(self.crf_list[idx]) # H x W

            img_a, crf_a = self.tf_at_once(img, crf)
            img_p, crf_p = self.tf_at_once(img, crf)

            out['img_a'] = img_a
            out['crf_a'] = crf_a
            out['img_p'] = img_p
            out['crf_p'] = crf_p
            out['name'] = name
            out['label'] = label

            return out
        
        else:
            img_a = self.tf_img_only(img)
            img_p = self.tf_img_only(img)

            out['img_a'] = img_a
            out['img_p'] = img_p
            out['name'] = name
            out['label'] = label

            return out 

class COCOClsDatasetMSF(COCOClsDataset):
    '''
        MSF : Multi-Scale and Flip
    '''
    def __init__(self, img_name_list_path, img_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, img_root, 'train', transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class COCOImageDataset_SelfSup(Dataset):

    def __init__(self, img_id_list_file, img_root, task):
        self.img_name_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root
        self.img_subdir = task + COCO_VERSION
        self.task = task

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.img_root, self.img_subdir)).convert("RGB")

        return name, img

class COCOClsDataset_SelfSup(COCOImageDataset_SelfSup):

    def __init__(self, img_id_list_file, img_root, crop, resize, cj):
        super().__init__(img_id_list_file, img_root, task='train')
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        
        self.crop = crop
        self.flag_asym_crop = self.crop[0]==self.crop[1]
        self.resize = resize
        self.cj = cj
        
        self.set_tf()

    def set_tf(self):

        self.tf_rr = imutils.random_resize(self.resize[0], self.resize[1])

        if self.flag_asym_crop:
            self.tf_rc_A = imutils.random_crop(self.crop[0])
            self.tf_rc_P = imutils.random_crop(self.crop[1])
            
        else:
            self.tf_rc = imutils.random_crop(self.crop[0])
        
        self.tf_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.tf_cj = transforms.RandomApply([transforms.ColorJitter(self.cj[0], self.cj[1], self.cj[2], self.cj[3])], p=0.8)
        self.tf_gray = transforms.RandomGrayscale(p=0.1)
        self.tf_norm = imutils.normalize()

        self.tf_list = []
        self.tf_list.append(imutils.HWC_to_CHW)
        self.tf_list.append(imutils.torch.from_numpy)
        self.tf_final = transforms.Compose(self.tf_list)

    def apply_tf(self, img, rc=None):

        img = self.tf_rr(img)
        img = self.tf_flip(img)
        img_ori = img
        # img = self.tf_cj(img)
        img = self.tf_gray(img)

        img = np.asarray(img)
        img = self.tf_norm(img)

        img_ori = np.asarray(img_ori)
        img_ori = self.tf_norm(img_ori)
        img_ori2= img_ori

        if rc==None:
            img = self.tf_rc(img)
        elif rc=='A':
            img,xy = self.tf_rc_A(img,get_xy=True,seg=False)
            img_ori = self.tf_rc_A(img_ori,xy=xy,seg=False)

        elif rc=='P':
            img,xy = self.tf_rc_P(img,get_xy=True,seg=False)
            img_ori2 = self.tf_rc_P(img_ori2, xy=xy, seg=False)

        img = self.tf_final(img)
        img_ori = self.tf_final(img_ori)
        img_ori2 = self.tf_final(img_ori2)

        return img,img_ori,img_ori2

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])
        
        if self.flag_asym_crop:
            img_A,img_ori,_ = self.apply_tf(img, rc='A')
            img_P,_,img_ori2 = self.apply_tf(img, rc='P')
        else:
            img_A,img_ori,_ = self.apply_tf(img)
            img_P,_,img_ori2 = self.apply_tf(img)
        
        pack = {}
        pack['img_ori'] = img_ori
        pack['img_oriP'] = img_ori2
        pack['img_a'] = img_A
        pack['img_p'] = img_P
        pack['label'] = label
        pack['name'] = name

        return pack


    def __len__(self):
        return len(self.img_name_list)


def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    # img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]
    # img_name_list = [img_gt_name[:-4] for img_gt_name in img_gt_name_list] #remove .jpg

    return img_gt_name_list


class COCOImageSegDataset(Dataset):

    def __init__(self, gt_path, img_name_list_path, coco_root, val_flag=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.coco_root = coco_root
        self.gt_path = gt_path


    def set_tf(self, phase):

        self.tf_rr = imutils.random_resize(256, 768)
        self.tf_rc = imutils.random_crop(385)

        # self.tf_rr = imutils.random_resize(320, 650)
        # self.tf_rc = imutils.random_crop(321)

        self.tf_cj = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        self.tf_norm = imutils.normalize()

        self.tf_permute = imutils.HWC_to_CHW

        self.tf_list = []
        self.tf_list.append(imutils.HWC_to_CHW)

        if phase == 'train':
            self.tf_list.append(imutils.torch.from_numpy)

        self.tf = transforms.Compose(self.tf_list)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx, val_flag=False):
        name = self.img_name_list[idx]
        
        # print(self.coco_root)

        # if 'val' in name: ##must be fixed shyoon
        if val_flag:
            # name = name[-25:]
            img_dir = os.path.join(self.coco_root, 'val2014', "COCO_val2014_"+name + '.jpg')
        else:
            # name = name[-27:]
            img_dir = os.path.join(self.coco_root, 'train2014', "COCO_train2014_"+name + '.jpg')

        # print(img_dir)

        img = PIL.Image.open(img_dir).convert("RGB")

        if not val_flag: #can be change according to your pGT format
            label = PIL.Image.open(os.path.join(self.gt_path, name + '.png'))
            # a=np.load(os.path.join(self.gt_path, name + '.npy'),allow_pickle=True).astype(np.uint8)
            
            # label = PIL.Image.fromarray(np.load(os.path.join(self.gt_path, name + '.npy'),allow_pickle=True).astype(np.uint8))
        else:
            label = PIL.Image.open(os.path.join(self.gt_path,'val2014',"COCO_val2014_"+name + '.png'))

        # label = PIL.Image.open(os.path.join(self.gt_path,'val2014', name + '.png'))

        # label = np.zeros_like(img,dtype=np.uint8)
        
        # if not val_flag:
        #     temp = np.load(self.gt_path + name + '.npy',allow_pickle=True)
        #     label_dict = temp.item()
        #     h, w = list(label_dict.values())[0].shape
        #     label = np.zeros((81,h, w), np.float32)
        #     for key in label_dict.keys():
        #         label[key] = label_dict[key]
        #     label = np.argmax(label,0).astype(np.uint8)
        #     label = PIL.Image.fromarray(np.array(label))
        # else:
        #     label = np.load(self.gt_path + name + '.npy',allow_pickle=True)
        #     label = PIL.Image.fromarray(np.array(label))



        # print('img',np.shape(img)) #h w
        # print('label',np.shape(label))

        if not val_flag:

            img, xy = self.tf_rr(img, get_xy=True)
            label = self.tf_rr(label, xy=xy)

            if random.random() < 0.5:
                transforms.functional.hflip(img)
                transforms.functional.hflip(label)

            img = self.tf_cj(img)

            img = np.asarray(img)
            label = np.expand_dims(np.asarray(label), axis=2)

            img = self.tf_norm(img)

            img, xy = self.tf_rc(img, get_xy=True)
            label = self.tf_rc(label, xy=xy)

            img = self.tf(img)
            label = self.tf(label)
       

        return name, img, label


class COCOImageSegDatasetMSF(COCOImageSegDataset):

    def __init__(self, gt_path, img_name_list_path, coco_root, scales, inter_transform=None, unit=1):
        super().__init__(gt_path, img_name_list_path, coco_root, val_flag=True)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def set_tf(self):

        self.tf_norm = imutils.normalize()
        self.tf_permute = imutils.HWC_to_CHW

    def __getitem__(self, idx):

        name, img, label = super().__getitem__(idx, val_flag=True)

        rounded_size = (
        int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        ms_label_list = []

        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            s_label = label.resize(target_size, resample=PIL.Image.NEAREST)

            ms_img_list.append(s_img)
            ms_label_list.append(s_label)

        for i in range(len(ms_img_list)):
            ms_img_list[i] = np.asarray(ms_img_list[i])
            ms_img_list[i] = self.tf_norm(ms_img_list[i])
            ms_img_list[i] = self.tf_permute(ms_img_list[i])

            ms_label_list[i] = np.expand_dims(np.asarray(ms_label_list[i]), axis=2)
            ms_label_list[i] = self.tf_permute(ms_label_list[i])

        msf_img_list = []
        msf_label_list = []

        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_label_list.append(ms_label_list[i])
            msf_label_list.append(np.flip(ms_label_list[i], -1).copy())

        msf_img_list_t = []
        msf_label_list_t = []

        for i in range(len(msf_img_list)):
            msf_img_list_t.append(torch.from_numpy(msf_img_list[i]))
            msf_label_list_t.append(torch.from_numpy(msf_label_list[i].copy()))

        return name, msf_img_list_t, msf_label_list_t

#######AFF################







class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)

from PIL import Image

class COCOAffDataset(COCOImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None,task=None):
        super().__init__(img_name_list_path, voc12_root,task='train', transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.png')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.png')

        label_la = np.expand_dims(np.array(Image.open(label_la_path)),axis=0)#np.load(label_la_path,allow_pickle=True).item()
        label_ha = np.expand_dims(np.array(Image.open(label_ha_path)),axis=0) #np.load(label_ha_path,allow_pickle=True).item()

        label = np.array(list(label_la) + list(label_ha))
        label = np.transpose(label, (1, 2, 0)) # H x W x (2 x num_cls)

        # print(label_la)


        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        no_score_region = np.max(label, -1) < 1e-5
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        # label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        # label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label_la = label_la.astype(np.uint8)
        label_ha = label_ha.astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255 # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label










