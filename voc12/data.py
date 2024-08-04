import pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image
import os.path
import scipy.misc
import random

import tools.imutils as imutils

##########################################################
from scipy.spatial.distance import cdist
import scipy.ndimage
import numpy as np
from matplotlib import pyplot as plt
##########################################################


IMG_FOLDER_NAME = "JPEGImages" ##################caution
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def save_img(x, path):
    plt.imshow(x)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path)
    plt.close()

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list,coco = False):

    if coco :
        cls_labels_dict = np.load('coco14/cls_labels_coco.npy',allow_pickle=True).item()
        cls_labels_dict_new = {}
        for k,v in cls_labels_dict.items():
            new_key = "%012d"%k
            cls_labels_dict_new[new_key]=v

        return [cls_labels_dict_new[img_name] for img_name in img_name_list]
    else:
        cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()
        return [cls_labels_dict[img_name] for img_name in img_name_list]
    
    # cls_labels_dict = np.load('voc12/cls_labels_custom2.npy', allow_pickle=True).item()


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path, coco=False):
    if coco:
        img_name_list = open(dataset_path).read().splitlines()
        # pdb.set_trace()
        # img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    else:
        img_gt_name_list = open(dataset_path).read().splitlines()
        img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    return img_name_list

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img

class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label



class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
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

class VOC12ImageDataset_SelfSup(Dataset):

    def __init__(self, img_name_list_path, voc12_root):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        return name, img

import torchvision.transforms.functional as TF
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import RandomResizedCropAndInterpolation
import torch.nn.functional as F


class VOC12ClsDataset_SelfSup(VOC12ImageDataset_SelfSup):

    def __init__(self, img_name_list_path, voc12_root, crop, resize, cj,step):
        super().__init__(img_name_list_path, voc12_root)
        
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        
        self.crop = crop
        self.flag_asym_crop = True 
        self.resize = resize
        self.cj = cj
        self.step = step

        assert resize>150
        
        self.set_tf()

    def set_tf(self):

        self.tf_rr = transforms.Resize(size=(int(self.resize), int(self.resize)))
        
        self.tf_cj = transforms.RandomApply([transforms.ColorJitter(self.cj[0], self.cj[1], self.cj[2], self.cj[3])], p=0.8)

        self.tf_list = []
        self.tf_list.append(transforms.ToTensor())
        self.tf_list.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        self.tf_final = transforms.Compose(self.tf_list)

    def apply_tf(self, img, img_diff, rc=None):

        img = self.tf_rr(img)
        img_diff = self.tf_rr(img_diff)
        
        if random.random()>0.5:
            img = TF.hflip(img)
            img_diff = TF.hflip(img_diff)
        
        img = self.tf_cj(img) 
        img_diff = self.tf_cj(img_diff)

        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.crop, self.crop))
        
        img = TF.crop(img, i, j, h, w)
        img_diff = TF.crop(img_diff, i, j, h, w)

        img = self.tf_final(img)
        img_diff = self.tf_final(img_diff)

        ##############
        # i, j, h, w = RandomResizedCropAndInterpolation.get_params(img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
        # img = TF.resized_crop(img, i, j, h, w, (self.crop,self.crop), TF.InterpolationMode.BICUBIC)
        # img_diff = TF.resized_crop(img_diff, i, j, h, w, (self.crop,self.crop), TF.InterpolationMode.BICUBIC)

        # if random.random()>0.5:
        #     img = TF.hflip(img)
        #     img_diff = TF.hflip(img_diff)
        
        # img = self.tf_cj(img) 
        # img_diff = self.tf_cj(img_diff)

        # img = self.tf_final(img)
        # img_diff = self.tf_final(img_diff)


        return img,img_diff

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)
        img_diff = PIL.Image.open(os.path.join('/mnt/shyoon4tb/denoising-diffusion-pytorch/images_step%s'%self.step, name + '.jpg')).convert("RGB")

        label = torch.from_numpy(self.label_list[idx])
        
 
        img_A,img_P = self.apply_tf(img,img_diff)
        
        # pack = {}
        # pack['img_a'] = img_A
        # pack['img_p'] = img_P
        # pack['label'] = label
        # pack['name'] = name

        # return pack
        return img_A, img_P, label, name


    def __len__(self):
        return len(self.img_name_list)

import torchvision
import cv2
import torchvision.transforms.functional as TF

class VOC12MocoDataset(VOC12ClsDataset):

    def __init__(self, args, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)

        # Get augmentation parameters from args
        self.crf_flag = False #args.use_crf
        self.resize_range = args.resize  # Default : 0.2, 1.0
        self.crop_size = args.crop  # Default : 224
        self.cj = args.cj  # Default : 0.4, 0.4, 0.4, 0.1

        assert args.resize[0]<1.0

        if self.crf_flag:
            self.crf_list = ['./crf/' + name + '.npy' for name in self.img_name_list]

        augmentation = [transforms.Resize((self.crop_size, self.crop_size)),
                        torchvision.transforms.RandomApply(
                            [transforms.ColorJitter(self.cj[0], self.cj[1], self.cj[2], self.cj[3])], p=0.8),
                        transforms.RandomGrayscale(p=0.2), #0.2 original
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                        ]
        
        augmentation_ori = [transforms.Resize((self.crop_size, self.crop_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                        ]


        self.hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.tf_moco = torchvision.transforms.Compose(augmentation)
        self.tf_moco_ori = torchvision.transforms.Compose(augmentation_ori)

    def tf_at_once(self, img, crf):
        rrcrop = transforms.RandomResizedCrop(self.crop_size, scale=(self.resize_range[0], self.resize_range[1]))
        crop_params = rrcrop.get_params(img, scale=(self.resize_range[0], self.resize_range[1]),
                                        ratio=(3.0 / 4.0, 4.0 / 3.0))

        img_ = TF.crop(img, *crop_params)
        crf_ = crf[crop_params[0]:crop_params[0] + crop_params[2], crop_params[1]:crop_params[1] + crop_params[3]]
        crf_ = cv2.resize(crf_, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)

        if random.randint(0, 1) == 0:
            img_ = self.hflip(img_)
            crf_ = cv2.flip(crf_, 1)

        img_ = self.tf_moco(img_)
        crf_mc_ = np.zeros((20, self.crop_size, self.crop_size))

        for i in range(20):
            crf_mc_[i][np.where(crf_ == i + 1)] = 1

        return img_, crf_mc_

    def tf_img_only(self, img):
        rrcrop = transforms.RandomResizedCrop(self.crop_size, scale=(self.resize_range[0], self.resize_range[1]))
        crop_params = rrcrop.get_params(img, scale=(self.resize_range[0], self.resize_range[1]),
                                        ratio=(3.0 / 4.0, 4.0 / 3.0))

        img = TF.crop(img, *crop_params)

        if random.randint(0, 1) == 0:
            img = self.hflip(img)

        img_ori = self.tf_moco_ori(img)
        img = self.tf_moco(img)

        return img,img_ori

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        if self.crf_flag:

            crf = np.load(self.crf_list[idx])  # H x W
            img_a, crf_a = self.tf_at_once(img, crf)
            img_p, crf_p = self.tf_at_once(img, crf)

            out = {}
            out['img_a'] = img_a
            out['crf_a'] = crf_a
            out['img_p'] = img_p
            out['crf_p'] = crf_p
            out['name'] = name
            out['label'] = label

            return out

        else:
            img_a,img_ori = self.tf_img_only(img)
            img_p,_ = self.tf_img_only(img)

            out = {}
            out['img_ori']=img_ori
            out['img_a'] = img_a
            out['img_p'] = img_p
            out['name'] = name
            out['label'] = label

            return out


class VOC12ImageSegDataset(Dataset):

    def __init__(self, gt_path, img_name_list_path, voc12_root, crop=384, val_flag=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.gt_path = gt_path
        self.crop = crop

    def set_tf(self, phase):

        # self.tf_rr = imutils.random_resize(256, 768) original
        # self.tf_rc = imutils.random_crop(384)
        self.tf_rr = imutils.random_resize(256, 768)
        self.tf_rc = imutils.random_crop(self.crop)

        self.tf_cj = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        self.tf_norm = imutils.normalize()

        self.tf_permute = imutils.HWC_to_CHW

        self.tf_list = []
        self.tf_list.append(imutils.HWC_to_CHW)
        
        if phase=='train':
            self.tf_list.append(imutils.torch.from_numpy)
            
        self.tf = transforms.Compose(self.tf_list)   

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx, val_flag=False):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        label = PIL.Image.open(self.gt_path + '/' + name + '.png')
        cls_label = torch.from_numpy(self.label_list[idx])

        # longer = max(img.size(0), img.size(1))
        # self.tf_rr = imutils.random_resize(longer*0.5, longer*2)

        if not val_flag:

            img, xy = self.tf_rr(img, get_xy=True)
            label = self.tf_rr(label, xy=xy)

            if random.random()<0.5:
                img = transforms.functional.hflip(img)
                label = transforms.functional.hflip(label)

            img = self.tf_cj(img)

            img = np.asarray(img)
            label = np.expand_dims(np.asarray(label), axis=2)

            img = self.tf_norm(img)

            img, xy = self.tf_rc(img, get_xy=True)
            label = self.tf_rc(label, xy=xy)

            img = self.tf(img)
            label = self.tf(label)

        return name, img, label, cls_label


class VOC12ImageSegDataset_test(Dataset):

    def __init__(self, gt_path, img_name_list_path, voc12_root, val_flag=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.gt_path = gt_path

    def set_tf(self, phase):

        self.tf_rr = imutils.random_resize(256, 768)
        self.tf_rc = imutils.random_crop(448)

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

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        # label = PIL.Image.open(self.gt_path + '/' + name + '.png')

        # longer = max(img.size(0), img.size(1))
        # self.tf_rr = imutils.random_resize(longer*0.5, longer*2)

        if not val_flag:

            img, xy = self.tf_rr(img, get_xy=True)
            # label = self.tf_rr(label, xy=xy)

            if random.random() < 0.25:
                img = transforms.functional.hflip(img)
                # label = transforms.functional.hflip(label)

            img = self.tf_cj(img)

            img = np.asarray(img)
            # label = np.expand_dims(np.asarray(label), axis=2)

            img = self.tf_norm(img)

            img, xy = self.tf_rc(img, get_xy=True)
            # label = self.tf_rc(label, xy=xy)

            img = self.tf(img)
            # label = self.tf(label)

        return name, img #, label


class VOC12ImageSegDatasetMSF(VOC12ImageSegDataset):

    def __init__(self, gt_path, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(gt_path, img_name_list_path, voc12_root, val_flag=True)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def set_tf(self):

        self.tf_norm = imutils.normalize()
        self.tf_permute = imutils.HWC_to_CHW

    def __getitem__(self, idx):
        name, img, label,_ = super().__getitem__(idx, val_flag=True)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        ms_label_list = []

        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
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

        for i in range(len(msf_img_list)):
            msf_img_list[i] = torch.from_numpy(msf_img_list[i])
            msf_label_list[i] = torch.from_numpy(msf_label_list[i])

        return name, msf_img_list, msf_label_list


class VOC12ImageSegDatasetMSF_test(VOC12ImageSegDataset_test):

    def __init__(self, gt_path, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(gt_path, img_name_list_path, voc12_root, val_flag=True)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def set_tf(self):

        self.tf_norm = imutils.normalize()
        self.tf_permute = imutils.HWC_to_CHW

    def __getitem__(self, idx):
        name, img  = super().__getitem__(idx, val_flag=True)

        rounded_size = (
        int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        # ms_label_list = []

        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            # s_label = label.resize(target_size, resample=PIL.Image.NEAREST)

            ms_img_list.append(s_img)
            # ms_label_list.append(s_label)

        for i in range(len(ms_img_list)):
            ms_img_list[i] = np.asarray(ms_img_list[i])
            ms_img_list[i] = self.tf_norm(ms_img_list[i])
            ms_img_list[i] = self.tf_permute(ms_img_list[i])

            # ms_label_list[i] = np.expand_dims(np.asarray(ms_label_list[i]), axis=2)
            # ms_label_list[i] = self.tf_permute(ms_label_list[i])

        msf_img_list = []
        msf_label_list = []

        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            # msf_label_list.append(ms_label_list[i])
            # msf_label_list.append(np.flip(ms_label_list[i], -1).copy())

        for i in range(len(msf_img_list)):
            msf_img_list[i] = torch.from_numpy(msf_img_list[i])
            # msf_label_list[i] = torch.from_numpy(msf_label_list[i])

        return name, msf_img_list









#########################MCTformer######################################


class COCOClsDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, transform2 = None, gen_attn=False):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path, coco=True)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, coco=True)
        self.coco_root = coco_root
        self.transform = transform
        self.transform2 = transform2
        self.train = train
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train or self.gen_attn :
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014',"COCO_train2014_"+name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014',"COCO_val2014_"+name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        img_diff = img

        if self.transform:
            img = self.transform(img)
         
        if self.transform2:
            img_diff = self.transform2(img_diff)
        else:
            img_diff = self.transform(img_diff)

        return img, img_diff, label, name

    def __len__(self):
        return len(self.img_name_list)

class COCOClsDatasetMS(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, scales, train=True, transform=None, gen_attn=False, unit=1):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.unit = unit
        self.scales = scales
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        if self.train or self.gen_attn:
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])

            # msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label

    def __len__(self):
        return len(self.img_name_list)
    
    
class COCOClsDataset_fft(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, transform2 = None, gen_attn=False):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path, coco=True)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, coco=True)
        self.coco_root = coco_root
        self.transform = transform
        self.transform2 = transform2
        self.train = train
        self.gen_attn = gen_attn

        if 'train' in img_name_list_path:
            self.dataset_train = True

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        # if self.train or self.gen_attn:
        if self.dataset_train:
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014',"COCO_train2014_"+name + '.jpg')).convert("RGB")
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014',"COCO_val2014_"+name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        img_diff = img

        if self.transform:
            img = self.transform(img)
         
        if self.transform2:
            img_diff = self.transform2(img_diff)
        else:
            img_diff = self.transform(img_diff)

        return img, img_diff, label, name

    def __len__(self):
        return len(self.img_name_list)

class VOC12Dataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, train=True, transform=None, transform2=None, gen_attn=False):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        self.transform2 = transform2

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        # img_diff400 = PIL.Image.open(os.path.join('/mnt/shyoon4tb/denoising-diffusion-pytorch/images_step400', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])
        img_diff = img

        if self.transform:
            img = self.transform(img)
        
        if self.transform2:
            img_diff = self.transform2(img_diff)
        else:
            img_diff = self.transform(img_diff)

        return img, img_diff, label, name

    def __len__(self):
        return len(self.img_name_list)



class VOC12DatasetMS(Dataset):
    def __init__(self, img_name_list_path, voc12_root, scales, train=True, transform=None, gen_attn=False, unit=1):
        # img_name_list_path = os.path.join(img_name_list_path, f'{"train_aug" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.transform = transform
        self.unit = unit
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        label = torch.from_numpy(self.label_list[idx])

        rounded_size = (int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(torch.flip(ms_img_list[i], [-1]))
        return msf_img_list, label, name

    def __len__(self):
        return len(self.img_name_list)














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


class VOC12AffDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

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

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')

        label_la = np.load(label_la_path,allow_pickle=True).item()
        label_ha = np.load(label_ha_path,allow_pickle=True).item()

        label = np.array(list(label_la.values()) + list(label_ha.values()))
        label = np.transpose(label, (1, 2, 0)) # H x W x (2 x num_cls)

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
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255 # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label


class VOC12AffGtDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_dir = label_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_path = os.path.join(self.label_dir, name + '.png')

        label = scipy.misc.imread(label_path)

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

        label = self.extract_aff_lab_func(label)

        return img, label












class VOC12AffSpDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

        mean_img = [0.485, 0.456, 0.406]
        std_img = [0.229, 0.224, 0.225]

        # transforms.Resize((384, 300)),
        self.custom_tf = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean_img, std=std_img),
                                            ])

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')
        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')
        
        la = np.load(label_la_path,allow_pickle=True).item()  # bg amplified -> confidence fg
        ha = np.load(label_ha_path,allow_pickle=True).item()  # fg amplified -> confidence bg
        cls_list = list(la.keys())
        num_cls = len(cls_list)

        img_ori = img
        img = np.array(img)
        H, W, _ = img.shape
        sp = qs(img, ratio=0.75, kernel_size=3, max_dist=8, sigma=0)
        num_sp = sp.max()+1
        
        rag = graph.rag_mean_color(img, sp)
        edge_index = np.transpose(np.array(rag.edges), (1,0))
        edge_index = np.concatenate((edge_index, edge_index[::-1]), axis=1)  

        # adj = np.zeros((num_sp, num_sp))
        # for i in range(H-1):
        #     for j in range(W-1):
        #         if sp[i,j]!=sp[i+1,j]:
        #             adj[sp[i,j],sp[i+1,j]] = 1
        #         if sp[i,j]!=sp[i,j+1]:
        #             adj[sp[i,j],sp[i,j+1]] = 1

        label_la = np.zeros((H,W,num_cls))
        label_ha = np.zeros((H,W,num_cls))
        
        label_cls = []
        for i in range(num_sp):
            reg = sp==i


            bg = np.average(ha[0][reg])
            flag = True

            fg_max = -100
            fg_cls = 0
            for j in range(num_cls):
                c = cls_list[j]

                if j!=0 and bg<=np.average(ha[c][reg]):
                    flag = False
                fg_temp = np.average(la[c][reg])

                if fg_max<fg_temp: # and 0.7<fg_temp:
                    fg_max = fg_temp
                    fg_cls = c
            
            if flag:
                label_cls.append(0)
            else:
                if fg_cls==0:
                    label_cls.append(-1)
                else:
                    label_cls.append(fg_cls)

        label_cls = np.asarray(label_cls)
        temp = np.copy(sp)

        for i in range(num_sp):
            temp[sp==i]=label_cls[i]
        
        #img = np.transpose(img, (2,0,1))
        img = self.custom_tf(img_ori)
        label_cls = np.array(label_cls)

        
        return img, sp, edge_index, label_cls, temp




class VOC12SpDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

        mean_img = [0.485, 0.456, 0.406]
        std_img = [0.229, 0.224, 0.225]

        # transforms.Resize((384, 300)),
        self.custom_tf = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean_img, std=std_img),
                                            ])

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')
        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')
        
        la = np.load(label_la_path,allow_pickle=True).item()  # bg amplified -> confidence fg
        ha = np.load(label_ha_path,allow_pickle=True).item()  # fg amplified -> confidence bg
        cls_list = list(la.keys())
        num_cls = len(cls_list)

        img_ori = img
        img = np.array(img)
        H, W, _ = img.shape
        sp = qs(img, ratio=0.75, kernel_size=3, max_dist=8, sigma=0)
        num_sp = sp.max()+1
        
        rag = graph.rag_mean_color(img, sp)
        edge_index = np.transpose(np.array(rag.edges), (1,0))
        edge_index = np.concatenate((edge_index, edge_index[::-1]), axis=1)  

        # adj = np.zeros((num_sp, num_sp))
        # for i in range(H-1):
        #     for j in range(W-1):
        #         if sp[i,j]!=sp[i+1,j]:
        #             adj[sp[i,j],sp[i+1,j]] = 1
        #         if sp[i,j]!=sp[i,j+1]:
        #             adj[sp[i,j],sp[i,j+1]] = 1

        label_la = np.zeros((H,W,num_cls))
        label_ha = np.zeros((H,W,num_cls))
        
        label_cls = []
        for i in range(num_sp):
            reg = sp==i


            bg = np.average(ha[0][reg])
            flag = True

            fg_max = -100
            fg_cls = 0
            for j in range(num_cls):
                c = cls_list[j]

                if j!=0 and bg<=np.average(ha[c][reg]):
                    flag = False
                fg_temp = np.average(la[c][reg])

                if fg_max<fg_temp: # and 0.7<fg_temp:
                    fg_max = fg_temp
                    fg_cls = c
            
            if flag:
                label_cls.append(0)
            else:
                if fg_cls==0:
                    label_cls.append(-1)
                else:
                    label_cls.append(fg_cls)

        label_cls = np.asarray(label_cls)
        temp = np.copy(sp)

        for i in range(num_sp):
            temp[sp==i]=label_cls[i]
        
        #img = np.transpose(img, (2,0,1))
        img = self.custom_tf(img_ori)
        label_cls = np.array(label_cls)

        
        return img, sp, edge_index, label_cls, temp
