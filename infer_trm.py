from __future__ import division
from __future__ import print_function

import os, time
import os.path as osp
import argparse
import glob
import random
import pdb
import importlib
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import tools.utils as utils

# Custom
from torch.utils.tensorboard import SummaryWriter

from evaluation import eval_in_script, do_python_eval

################################################################################
# Infer CAM image, CAM dict and CRF dict from given experiments/checkpoints.
# All of the result files will be saved under given experiment folder.
#
# If you want to get CAM_dict files...
# python infer.py --name [exp_name] --load_epo [epoch] --dict
#
# Or if you want to get CRF dict files with certain alpha (let, a1 and a2)...
# python infer.py --name [exp_name] --load_epo [epoch] --crf --alphas a1 a2
#
# python infer.py --name [exp_name] --model CTI --dict --crf --load_epo 3 --infer_list voc12/train.txt --alphas
# Of course you can do them at the same time. 
# To get CAM image, simply add --vis.
################################################################################

# CUDA_VISIBLE_DEVICES=5 python infer_trm.py --name [exp_name] --model [model] --load_epo [epo] --dict

if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                  'bus', 'car', 'cat', 'chair', 'cow', 
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data-set', default='VOC12', type=str, help='dataset')
    parser.add_argument('--data-path', default='./data/VOC2012', type=str, help='dataset')
    
    parser.add_argument("--val_list", default="./voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=64, type=int)

    # Augmentation
    parser.add_argument("--resize", default=[256, 512], nargs='+', type=float)
    parser.add_argument("--crop", default=[224, 384], nargs='+', type=int)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs='+', type=float)

   
    parser.add_argument("--W", default=[1.0, 1.0, 1.0], nargs='+', type=float)
    parser.add_argument("--C", default=20, type=int)
    parser.add_argument("--M", default=30, type=int)
    parser.add_argument("--CRF", default=4, type=int)

    # Learning rate
    parser.add_argument("--epochs", default=60, type=int)

    # Experiments
    parser.add_argument("--model", default='CTI', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    # parser.add_argument("--seed", default=5123, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--load_epo", required=True, type=int)
    parser.add_argument("--load_pretrained", default=None)

    ##############TRM
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    parser.set_defaults(repeated_aug=True)
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',  help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',  help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,  help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',       help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',help='LR decay rate (default: 0.1)')

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", action='store_true')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--print_freq", default=60, type=int)
    parser.add_argument("--out_num", default=100, type=int)
    parser.add_argument("--alphas", default=[6, 10, 24], nargs='+', type=int)
    parser.add_argument("--visport", default=5137, type=int)
    
    #############################  MCTFORMER ###############################
    # TRM
    parser.add_argument('--gen_attention_maps', action='store_true')
    parser.add_argument('--trm', default='deit_small_MCTformerV2_CTI', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',  help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth1', help='finetune from checkpoint')                        
    

    args = parser.parse_args()
    args.max_step = 1

    logger = logging.getLogger(__name__)

    print('Infer experiment ' + args.name + '!')
    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path = utils.make_path(args)

    args.writer = SummaryWriter(log_dir=exp_path)

    infer_dataset,_ = utils.build_dataset_trm(is_train=False, args=args, gen_attn=True)
    sampler_val = torch.utils.data.SequentialSampler(infer_dataset)
    
    infer_data_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print('Infer dataset is loaded from ' + args.val_list)
    
    model = getattr(importlib.import_module('models.model_'+args.model), 'model_WSSS')(args, logger)
    if args.load_pretrained is not None:
        model.load_pretrained_model(args.load_pretrained)
    else:
        model.load_model(args.load_epo, ckpt_path)
    model.set_phase('eval')

    print('#'*111)
    print(('#'*46)+' Start infer loop '+('#'*47))
    print('#'*111)

    model.infer_init(args.load_epo)

    for iter, pack in enumerate(tqdm(infer_data_loader)):
        model.unpack(pack)
        model.infer_multi(42, train_path, dict_path, crf_path, vis=args.vis, dict=args.dict, crf=args.crf)    # original


    import pandas as pd
    eval_list='train'
    eval_list = './data/VOC2012/ImageSets/Segmentation/' + eval_list + '.txt'
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values

    for i in range(35,45):
        t = i/100.#+0.05
        loglist = do_python_eval(dict_path, './data/VOC2012/SegmentationClass', name_list, 21, 'cam', t, printlog=False)
        print('%d/60 threshold: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))