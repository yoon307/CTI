import os
import os.path as osp
import random

import PIL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import voc12.data
import coco14.data
import tools.imutils as imutils
from timm.data import create_transform

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

class PolyOptimizer_adam(torch.optim.Adam):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class PolyOptimizer_cls(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                if i == 4:
                    self.param_groups[i]['lr'] = self.__initial_lr[i]
                else:
                    self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def make_path(args):

    exp_path = osp.join('./experiments', args.name)
    ckpt_path = osp.join(exp_path, 'ckpt')
    train_path = osp.join(exp_path, 'train')
    val_path = osp.join(exp_path, 'val')
    infer_path = osp.join(exp_path, 'infer')
    dict_path = osp.join(exp_path, 'dict')
    crf_path = osp.join(exp_path, 'crf')
        
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.makedirs(ckpt_path)
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(infer_path)
        os.makedirs(dict_path)
        os.makedirs(crf_path)
        print(exp_path + ' is built.')
    else:
        print(exp_path + ' already exsits.')

    for alpha in args.alphas:
        crf_alpha_path = osp.join(crf_path, str(alpha).zfill(2))
        if not os.path.exists(crf_alpha_path):
            os.makedirs(crf_alpha_path)

    return exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path


def make_path_with_log(args):

    exp_path = osp.join('./experiments', args.name)
    ckpt_path = osp.join(exp_path, 'ckpt')
    train_path = osp.join(exp_path, 'train')
    val_path = osp.join(exp_path, 'val')
    infer_path = osp.join(exp_path, 'infer')
    dict_path = osp.join(exp_path, 'dict')
    crf_path = osp.join(exp_path, 'crf')
    log_path = osp.join(exp_path, 'log.txt')
        
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.makedirs(ckpt_path)
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(infer_path)
        os.makedirs(dict_path)
        os.makedirs(crf_path)
        print(exp_path + ' is built.')
    else:
        print(exp_path + ' already exsits.')

    for alpha in args.alphas:
        crf_alpha_path = osp.join(crf_path, str(alpha).zfill(2))
        if not os.path.exists(crf_alpha_path):
            os.makedirs(crf_alpha_path)

    return exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path


def build_dataset(phase='train', path="voc12/train_aug.txt", root='./data/VOC2012',is_vgg=False,crop=384,resize=[256,448]):

    tf_list = []

    if phase=='train':
        tf_list.append(imutils.random_resize(resize[0], resize[1]))
        tf_list.append(transforms.RandomHorizontalFlip())
        tf_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

    tf_list.append(np.asarray)

    if is_vgg:
        tf_list.append(imutils.normalize_vgg())
        print("VGG")
    else:
        tf_list.append(imutils.normalize())
        print("ResNet38d")

    if phase=='train':
        tf_list.append(imutils.random_crop(crop))

    tf_list.append(imutils.HWC_to_CHW)

    if phase=='train':
        tf_list.append(imutils.torch.from_numpy)

    tf = transforms.Compose(tf_list)

    if phase=='train':
        dataset = voc12.data.VOC12ClsDataset(path, voc12_root=root, transform=tf)
    elif phase=='val':
        # MSF dataset augments an image to 8 images with multi-scale & flip 
        dataset = voc12.data.VOC12ClsDatasetMSF(path, voc12_root=root, scales=[0.5,1.0,1.5, 2.0], inter_transform=tf)
    
    return dataset


def build_dataset_mix(phase='train', path="voc12/train_aug.txt", root='./data/VOC2012', is_vgg=False):

    tf_list = []

    if phase == 'train':
        tf_list.append(imutils.random_resize(384, 512))
        tf_list.append(transforms.RandomHorizontalFlip())
        tf_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

    tf_list.append(np.asarray)

    if is_vgg:
        tf_list.append(imutils.normalize_vgg())
        print("VGG")
    else:
        tf_list.append(imutils.normalize())
        print("ResNet38d")

    if phase == 'train':
        tf_list.append(imutils.random_crop(448))

    tf_list.append(imutils.HWC_to_CHW)

    if phase == 'train':
        tf_list.append(imutils.torch.from_numpy)

    tf = transforms.Compose(tf_list)

    if phase == 'train':
        dataset = voc12.data.VOC12ClsDataset(path, voc12_root=root, transform=tf)
    elif phase == 'val':
        # MSF dataset augments an image to 8 images with multi-scale & flip
        dataset = voc12.data.VOC12ClsDatasetMSF(path, voc12_root=root, scales=[0.5, 1.0, 1.5, 2.0], inter_transform=tf)

    return dataset

def build_dataset_dl(phase='train', path="voc12/train.txt", gt_path='', root='./data/VOC2012',crop=384):
    
    
    if phase=='train':
        dataset = voc12.data.VOC12ImageSegDataset(gt_path, path, voc12_root=root,crop=crop)
        dataset.set_tf(phase)
        
    if phase=='val':
        dataset = voc12.data.VOC12ImageSegDatasetMSF(gt_path, path, voc12_root=root,scales=[0.5,0.75,1.0,1.25,1.5])#scales=[0.5, 0.75, 1, 1.25, 1.5])#
        dataset.set_tf()
    if phase=='test':
        dataset = voc12.data.VOC12ImageSegDatasetMSF_test(gt_path,path,voc12_root=root,scales=[0.5,0.75,1.0,1.25,1.5])
        dataset.set_tf()

    return dataset

def build_dataset_coco_dl(phase='train', path='coco/img_name_train2014.txt', gt_path='', root='./data/COCO2014'):

    if phase=='train':
        dataset = coco14.data.COCOImageSegDataset(gt_path, path, coco_root=root)
        dataset.set_tf(phase)
        
    if phase=='val':
        dataset = coco14.data.COCOImageSegDatasetMSF(gt_path, path, coco_root=root, scales=[0.5,0.75, 1.0, 1.25, 1.5])
        dataset.set_tf()

    return dataset

def build_dataset_moco(args, phase='train', path="voc12/train_aug.txt", root='./data/VOC2012'):

    tf_list = []
    tf_list.append(np.asarray)
    tf_list.append(imutils.normalize())
    tf_list.append(imutils.HWC_to_CHW)
    tf = transforms.Compose(tf_list)

    crop = args.crop
    resize = args.resize
    cj = args.cj
    step = args.step

    if phase == 'train':
        dataset = voc12.data.VOC12ClsDataset_SelfSup(path, voc12_root=root, crop=crop, resize=resize, cj=cj, step=step)
        # dataset = voc12.data.VOC12MocoDataset(args, path, voc12_root=root)

    elif phase == 'val':
        # MSF dataset augments an image to 8 images with multi-scale & flip
        dataset = voc12.data.VOC12ClsDatasetMSF(path, voc12_root=root, scales=[0.5, 1.0, 1.5, 2.0], inter_transform=tf)

    return dataset




def build_dataset_trm(is_train, args, gen_attn=False):
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        if gen_attn:
            # args.gen_attention_maps = False
            args.gen_attention_maps = True
            transform = build_transform(is_train, args)
            dataset = voc12.data.VOC12Dataset(img_name_list_path=args.val_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        else: 
            args.gen_attention_maps = False

            args.aa = None
            args.color_jitter = None
            transform = build_transform(is_train, args)

            args.aa = 'rand-m9-mstd0.5-inc1'
            args.color_jitter = 0.4
            args.scale = args.scale2
            transform2 = build_transform(is_train, args)

            dataset = voc12.data.VOC12Dataset(img_name_list_path=args.train_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform, transform2=transform2)
        nb_classes = 20

    elif args.data_set == 'VOC12MS':
        
        args.gen_attention_maps = True
        transform = build_transform(is_train, args)
        dataset = voc12.data.VOC12DatasetMS(img_name_list_path=args.val_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20


    elif args.data_set == 'COCO':
        if gen_attn:
            # args.gen_attention_maps = False
            args.gen_attention_maps = True
            transform = build_transform(is_train, args)
            dataset = voc12.data.COCOClsDataset(img_name_list_path=args.val_list, coco_root=args.data_path,label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        else: 
            args.gen_attention_maps = False
         
            args.color_jitter = None
            args.aa = None
            transform = build_transform(is_train, args)

            args.aa = 'rand-m9-mstd0.5-inc1'
            args.color_jitter = 0.4
            args.scale = args.scale2
            transform2 = build_transform(is_train, args)

            dataset = voc12.data.COCOClsDataset(img_name_list_path=args.train_list, coco_root=args.data_path,label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform, transform2=transform2)
       
        nb_classes = 80


    return dataset, nb_classes

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms.functional import InterpolationMode

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train

        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=args.scale
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
            # transform.transforms[0] = transforms.CenterCrop(
                # args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)



def build_dataset_trm_fft(is_train, args, gen_attn=False):
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        if gen_attn:
            # args.gen_attention_maps = False
            args.gen_attention_maps = True
            transform = build_transform(is_train, args)
            dataset = voc12.data.VOC12Dataset(img_name_list_path=args.val_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        else: 
            if is_train:
                args.gen_attention_maps = False
                args.color_jitter = None
                args.aa = None
                transform = build_transform(is_train, args)

                args.color_jitter = 0.4
                args.scale = args.scale2
                args.aa = 'rand-m9-mstd0.5-inc1'
                transform2 = build_transform(is_train, args)

                dataset = voc12.data.VOC12Dataset(img_name_list_path=args.train_list, voc12_root=args.data_path,
                                train=is_train, gen_attn=gen_attn, transform=transform, transform2=transform2)
            else:
                args.gen_attention_maps = False
                transform = build_transform(is_train, args)
                dataset = voc12.data.VOC12Dataset(img_name_list_path=args.train_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20

    elif args.data_set == 'VOC12MS':
        
        args.gen_attention_maps = True
        transform = build_transform(is_train, args)
        dataset = voc12.data.VOC12DatasetMS(img_name_list_path=args.val_list, voc12_root=args.data_path, scales=tuple(args.scales),
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20


    elif args.data_set == 'COCO':
        if gen_attn:
            # args.gen_attention_maps = False
            args.gen_attention_maps = True
            transform = build_transform(is_train, args)
            dataset = voc12.data.COCOClsDataset_fft(img_name_list_path=args.val_list, coco_root=args.data_path,label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        else:
            if is_train:
                args.gen_attention_maps = False
                args.color_jitter = None
                args.aa = None
                transform = build_transform(is_train, args)

                args.color_jitter = 0.4
                args.scale = args.scale2
                args.aa = 'rand-m9-mstd0.5-inc1'
                transform2 = build_transform(is_train, args)

                dataset = voc12.data.COCOClsDataset_fft(img_name_list_path=args.train_list, coco_root=args.data_path,label_file_path=args.label_file_path,
                                train=is_train, gen_attn=gen_attn, transform=transform, transform2=transform2)
            else:
                args.gen_attention_maps = False
                transform = build_transform(is_train, args)
                dataset = voc12.data.COCOClsDataset_fft(img_name_list_path=args.train_list, coco_root=args.data_path,label_file_path=args.label_file_path,
                                train=is_train, gen_attn=gen_attn, transform=transform)
       
        nb_classes = 80


    return dataset, nb_classes

def build_transform_fft(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train

        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            scale=args.scale
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
            # transform.transforms[0] = transforms.CenterCrop(
                # args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        # size = int((256 / 224) * args.input_size)
        # t.append(
        #     transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        # )
        # t.append(transforms.CenterCrop(args.input_size))

        t.append(
            transforms.Resize((args.input_size, args.input_size))
        )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)