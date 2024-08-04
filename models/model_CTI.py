from cProfile import label
from http.client import NON_AUTHORITATIVE_INFORMATION
import os, time
from statistics import mode
import os.path as osp
import argparse
import glob
import random
import pdb
from turtle import pos

import numpy as np
from numpy.core.fromnumeric import size
# from sklearn.preprocessing import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

# Image tools
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from torchvision import transforms
import torchvision

import voc12.data
from tools import utils, pyutils, trmutils
from tools.imutils import save_img, denorm, _crf_with_alpha, cam_on_image
# import tools.visualizer as visualizer
from networks import mctformer

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler


import sys
sys.path.append("..") 

def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


##########
# CUDA_VISIBLE_DEVICES=0,1 python train_trm.py --name cti --model CTI --W 1 1 1
##########

class model_WSSS():

    def __init__(self, args, logger):

        self.args = args

        if self.args.C == 20:
            self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
            self.categories_withbg = ['bg','aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        elif self.args.C == 80:
            self.categories= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                        'bus', 'train', 'truck', 'boat', 'traffic_light',
                        'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird',
                        'cat', 'dog', 'horise', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
                        'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle',
                        'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                        'bowl', 'banana', 'apple', 'sandwich', 'orange',
                        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                        'cake', 'chair', 'couch', 'potted_plant', 'bed',
                        'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
                        'toaster', 'sink', 'refrigerator', 'book', 'clock',
                        'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
            self.categories_withbg= ['bg','person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                        'bus', 'train', 'truck', 'boat', 'traffic_light',
                        'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird',
                        'cat', 'dog', 'horise', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
                        'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle',
                        'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                        'bowl', 'banana', 'apple', 'sandwich', 'orange',
                        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                        'cake', 'chair', 'couch', 'potted_plant', 'bed',
                        'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
                        'toaster', 'sink', 'refrigerator', 'book', 'clock',
                        'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']


        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        self.bce = nn.BCEWithLogitsLoss()
        self.bs = args.batch_size
        self.logger = logger
        self.writer = args.writer

        # Model attributes
        self.net_names = ['net_trm']
        self.base_names = ['cls','bg','ctk_swap_intra','ctk_swap_cross']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.ebd_memory_t = []
        self.ebd_memory_s = []
        self.ctk_global = [[] for _ in range(self.args.C)]

        self.is_empty_memory = [True for i in range(len(self.categories))]

        self.nets = []
        self.opts = []

        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.num_count = 0
        
        #Tensorboard
        self.global_step = 0

        self.val_wrong = 0
        self.val_right = 0

        self.num_class = args.C


        # Define networks
        self.net_trm = create_model(
            'deit_small_MCTformerV2_CTI',
            pretrained=False,
            num_classes=args.C,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None
        )

            
        if args.finetune:
           
            checkpoint = torch.load("./pretrained/deit_small_imagenet.pth", map_location='cpu')

            try:
                checkpoint_model = checkpoint['model']
            except:
                checkpoint_model = checkpoint
            state_dict = self.net_trm.state_dict()

            if 'head.bias' in state_dict.keys():
                for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
            else:
                for k in ['head.weight', 'head_dist.weight', 'head_dist.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

            
        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = self.net_trm.patch_embed.num_patches
        if args.finetune.startswith('https'):
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.net_trm.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

        if args.finetune.startswith('https') and 'MCTformer' in args.trm:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,self.num_class+1,1)
        else:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = pos_tokens[:, :, :, None, :, None].expand(-1, -1, -1, 1, -1, 1).reshape(pos_tokens.size(0), pos_tokens.size(1), pos_tokens.size(2), pos_tokens.size(3) )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        checkpoint_model['pos_embed_cls'] = extra_tokens
        checkpoint_model['pos_embed_pat'] = pos_tokens

        if args.finetune.startswith('https') and 'MCTformer' in args.trm:
            cls_token_checkpoint = checkpoint_model['cls_token']
            checkpoint_model['cls_token'] = cls_token_checkpoint

        self.net_trm.load_state_dict(checkpoint_model, strict=False)

        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.KD = nn.KLDivLoss(reduction='batchmean')
        
    

    # Save networks
    def save_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        torch.save(self.net_trm.module.state_dict(), ckpt_path + '/' + epo_str + 'net_trm.pth')
    
    def save_model_coco(self, epo, iter, ckpt_path):
        epo_str = str(epo).zfill(3)
        iter_str = str(iter).zfill(5)
        torch.save(self.net_trm.module.state_dict(), ckpt_path + '/' + epo_str +"_"+ iter_str + 'net_trm.pth')

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_trm.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_trm.pth'), strict=True)

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))
    
    def load_pretrained_model(self, pretrained_path):
        self.net_trm.load_state_dict(torch.load(pretrained_path), strict=True)

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))

    # Set networks' phase (train/eval)
    def set_phase(self, phase):

        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
            self.logger.info('Phase : train')

        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
            self.logger.info('Phase : eval')
        # self.net_sup.eval()

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args


        linear_scaled_lr = args.lr * args.batch_size * trmutils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

        self.opt_trm = create_optimizer(args, self.net_trm)
        self.lr_scheduler, _ = create_scheduler(args, self.opt_trm)


        self.logger.info('Poly-optimizer for trm is defined.')

        self.net_trm = torch.nn.DataParallel(self.net_trm.to(self.dev))
        self.logger.info('Networks are uploaded on multi-gpu.')

        self.nets.append(self.net_trm)

    # Unpack data pack from data_loader
    def unpack(self, pack):

        if self.phase == 'train':
            self.img = pack[0].to(self.dev)  # B x 3 x H x W
            self.img_pos = pack[1].to(self.dev)  # B x 3 x H x W
            self.label = pack[2].to(self.dev)  # B x 20
            self.name = pack[3]  # list of image names

        if self.phase == 'eval':
            self.img = pack[0]
            self.img_pos = pack[1]
            self.label = pack[2].to(self.dev)
            self.name = pack[3][0]

            self.label_bg = torch.cat((torch.ones_like(self.label[:,0].unsqueeze(-1)), self.label), dim=1) #FG

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):
        # Tensor dimensions
        B = self.img.shape[0]
        H = self.img.shape[2]
        W = self.img.shape[3]

        h,w = H//16, W//16
        C = self.num_class  # Number of cls

        self.B = B
        self.C = C

        ################################################### Update TRM ###################################################
        self.opt_trm.zero_grad()
        self.net_trm.train()

        swap_idx = 3 
        ######################################################################
        outputs = self.net_trm(self.img,None,swap_idx)

        self.out = outputs['cls']
        self.out_patch = outputs['pcls']
        cams = outputs['cams']
        rcams = outputs['rcams']
        fcams = outputs['fcams']
        mtatt = outputs['mtatt']
        patch_attn = outputs['attn']
        ctk = outputs['ctk']

        ####################################################################
        outputs_pos = self.net_trm(self.img_pos,None,swap_idx)
        self.out_pos = outputs_pos['cls']
        self.out_patch_pos = outputs_pos['pcls']
        ctk_pos = outputs_pos['ctk']
        ####################################################################

        warmup_epoch=-1 if C==20 else 1

        self.loss_cls = 1*(
            F.multilabel_soft_margin_loss(self.out,self.label)
            + F.multilabel_soft_margin_loss(self.out_patch,self.label)   
            + F.multilabel_soft_margin_loss(self.out_pos,self.label)
            + F.multilabel_soft_margin_loss(self.out_patch_pos,self.label) 
            )
        loss_trm = self.loss_cls 

        ######################################
        if C==20:
            fcams_fg = (mtatt*cams)[:,1:,:,:]*self.label.view(B,C,1,1) #for PASCAL
        else:
            fcams_fg = (mtatt*cams.detach())[:,1:,:,:]*self.label.view(B,C,1,1) #for COCO
        fcams_fg = torch.max(fcams_fg,dim=1)[0].view(B,1,h,w)
        fcams_bg = (fcams[:,0,:,:].view(B,1,h,w))

        rcams_fg = torch.matmul(patch_attn.unsqueeze(1).double(), fcams_fg.double().view(fcams_fg.shape[0],fcams_fg.shape[1], -1, 1)).reshape(fcams_fg.shape[0],fcams_fg.shape[1], h, w) #(B 1 N2 N2) * (B,20,N2,1)

        rcams_bg = torch.matmul(patch_attn.unsqueeze(1).double(), fcams_bg.double().view(fcams_bg.shape[0],fcams_bg.shape[1], -1, 1)).reshape(fcams_bg.shape[0],fcams_bg.shape[1], h, w) #(B 1 N2 N2) * (B,20,N2,1)
        
        _rcams_fg = self.max_norm(rcams_fg)
        _rcams_bg = self.max_norm(rcams_bg)
        
        self.rcams_fg = F.interpolate(_rcams_fg,size=self.img.size()[2:],mode='bilinear',align_corners=False)
        self.rcams_bg = F.interpolate(_rcams_bg,size=self.img.size()[2:],mode='bilinear',align_corners=False)

        if self.args.W[0]>0 and epo>warmup_epoch:
            
            self.loss_bg = self.args.W[0]*(
                ((1-_rcams_fg)-_rcams_bg).abs().mean() 
            )

            loss_trm += self.loss_bg
        else:
            self.loss_bg = torch.Tensor([0])[0]

        self.loss_cls_bg = torch.Tensor([0])[0]


        ##################INTRA#########################
        if self.args.W[1]>0 and epo>warmup_epoch:
            swap_ctk_pos = ctk_pos[swap_idx] #BEST

            outputs_swap_INTRA= self.net_trm(self.img, swap_ctk_pos, swap_idx)    

            cams_swap_INTRA = outputs_swap_INTRA['cams']

            self.loss_ctk_swap_intra = self.args.W[1]*(
                ((self.max_norm(cams)-self.max_norm(cams_swap_INTRA))).abs().mean()
            )

            loss_trm += self.loss_ctk_swap_intra
        else:
            self.loss_ctk_swap_intra = torch.Tensor([0])[0]

        
        ###################CROSS###################
        if self.args.W[2]>0 and epo>warmup_epoch: 

            swap_ctk = ctk[swap_idx][:,1:,:].clone().detach()
            
            for _b, _c in torch.nonzero(self.label):
                input = F.layer_norm(ctk[swap_idx][_b,1+_c,:].detach().unsqueeze(0), [384]).squeeze(0)
                self.ctk_global[_c].append(input)

            ctk_global_tensor = torch.zeros((C,384)).cuda()

            mem = 30 # To prevent non-existence of object in random selection
            valid = 0

            for i in range(C):
                if len(self.ctk_global[i])>=mem:
                    self.ctk_global[i] = self.ctk_global[i][-mem:]
                    valid += 1

            for i in range(C):
                if len(self.ctk_global[i])>0:
                    ctk_global_tensor[i] += torch.stack(self.ctk_global[i],dim=0).mean(0)

            swap_ctk = (ctk[swap_idx][:,1:,:]*(1-self.label.view(B,C,1)) +
                        ctk_global_tensor*self.label.view(B,C,1))

            outputs_swap_CROSS = self.net_trm(self.img, swap_ctk.detach(), swap_idx)

            fcams_as_swap_CROSS = outputs_swap_CROSS['fcams_as']

            self.loss_ctk_swap_cross = self.args.W[2]*(
                ((self.max_norm(cams)[:,1:,:,:]-self.max_norm(fcams_as_swap_CROSS)[:,1:,:,:])).abs().mean()
            )

            if valid==self.num_class:
                loss_trm += self.loss_ctk_swap_cross
        else:
            self.loss_ctk_swap_cross = torch.Tensor([0])[0]


        loss_trm.backward()

        self.opt_trm.step()
        
        ################################################### Export ###################################################


        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()

        self.count += 1
        #Tensorboard
        self.global_step +=1

        self.count_rw(self.label, self.out_patch, 0)     # FG get rw except fg

    # Initialization for msf-infer
    def infer_init(self,epo):
        n_gpus = torch.cuda.device_count()
        self.net_trm.eval()
        # self.net_trm_replicas = torch.nn.parallel.replicate(self.net_trm.module, list(range(n_gpus)))

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        gt = self.label_bg[0].cpu().detach().numpy()
        self.gt_cls = np.nonzero(gt)[0]

        
        B, _, H, W = self.img.shape
        n_gpus = torch.cuda.device_count()


        cam = self.net_trm.module.forward(self.img.cuda(),return_att=True,n_layers= 12)
        cam = F.interpolate(cam,[H,W],mode='bilinear',align_corners=False) * self.label_bg.view(B,self.num_class+1,1,1)
        

        cam_flip = self.net_trm.module.forward(torch.flip(self.img,(3,)).cuda(),return_att=True,n_layers= 12)
        cam_flip = F.interpolate(cam_flip,[H,W],mode='bilinear',align_corners=False)*self.label_bg.view(B,self.num_class+1,1,1)
        cam_flip = torch.flip(cam_flip,(3,))
   
        cam = cam+cam_flip
        norm_cam = self.max_norm(cam)[0].detach().cpu().numpy()
 
        self.cam_dict = {}

        for i in range(self.num_class):
            if self.label[0, i] > 1e-5:
                self.cam_dict[i] = norm_cam[i+1]

        if vis:
            input = denorm(self.img[0])
            for c in self.gt_cls:
                temp = cam_on_image(input.cpu().detach().numpy(), norm_cam[c])
                self.writer.add_image(self.name+'/'+self.categories_withbg[c], temp, epo)

        if dict:
            np.save(osp.join(dict_path, self.name + '.npy'), self.cam_dict)

        if crf:
            for a in self.args.alphas:
                crf_dict = _crf_with_alpha(self.cam_dict, self.name, alpha=a)
                np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)

    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter):

        loss_str = ''
        acc_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.loss_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '

        for i in range(len(self.acc_names)):
            if self.right_count[i] != 0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc

        self.logger.info(loss_str[:-2])
        self.logger.info(acc_str[:-2])

        ###Tensorboard###
        for i in range(len(self.loss_names)):
            self.writer.add_scalar("Loss/%s"%self.loss_names[i],(self.running_loss[i] / self.count),self.global_step)

        # for c in self.gt_cls:
        temp = cam_on_image(denorm(self.img[0]).cpu().detach().numpy(), self.rcams_bg[0][0].detach().cpu().numpy())
        self.writer.add_image('000000/bg', temp, epo)
        temp = cam_on_image(denorm(self.img[0]).cpu().detach().numpy(), self.rcams_fg[0][0].detach().cpu().numpy())
        self.writer.add_image('000000/fg', temp, epo)

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0

    def count_rw(self, label, out, idx):
        for b in range(out.size(0)):  # 8
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred = out[b].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count[idx] += 1
                else:
                    self.wrong_count[idx] += 1

    # Max_norm
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp

    def cam_l1(self, cam1, cam2):
        return torch.mean(torch.abs(cam2.detach() - cam1))

