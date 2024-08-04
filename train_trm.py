from __future__ import division
from __future__ import print_function

# Base
import os
import os.path as osp
from tqdm import tqdm
import random
import importlib
import argparse
import logging
import pdb

from matplotlib import pyplot as plt

# DL
import numpy as np
import torch
from torch.utils.data import DataLoader

# Custom
import tools.imutils as imutils
import tools.utils as utils
import tools.pyutils as pyutils

import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
import shutil


from evaluation import eval_in_script

if __name__ == '__main__':
    def parse_tuple(arg):
        try:
            # Split the input string by commas and convert the parts to integers
            parts = [float(x) for x in arg.split(',')]
            return tuple(parts)
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid tuple format. Use 'x,y' where x and y are integers.")

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data-set', default='VOC12', type=str, help='dataset')
    parser.add_argument('--data-path', default='./data/VOC2012', type=str, help='dataset')
    
    parser.add_argument("--train_list", default="./voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="./voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=64, type=int)

    # Augmentation
    parser.add_argument("--resize", default=[256, 512], nargs='+', type=float)
    parser.add_argument("--crop", default=[224, 384], nargs='+', type=int)
    parser.add_argument("--cj", default=[0.4, 0.4, 0.4, 0.1], nargs='+', type=float)

   
    parser.add_argument("--W", default=[0.1, 1.0, 1.0], nargs='+', type=float)
    parser.add_argument("--C", default=20, type=int)
    parser.add_argument("--T", default=5, type=float)
    parser.add_argument("--A", default=0.2, type=float)
    parser.add_argument("--M", default=0.01, type=float)
    parser.add_argument("--CRF", default=4, type=int)

    parser.add_argument("--step", default=[0, 150, 15], nargs='+', type=int)
    
    # Learning rate
    parser.add_argument("--epochs", default=60, type=int)

    # Experiments
    parser.add_argument("--model", default='CTI', type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##############TRM
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.add_argument('--scale', default=(0.3,1.2), type=parse_tuple) 
    parser.add_argument('--scale2', default=(0.3,1.2), type=parse_tuple) 

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
    parser.add_argument("--dict", action='store_false')
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

    pyutils.same_seeds(0)

    
    cudnn.benchmark = True
    
    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path = utils.make_path_with_log(args)
    args.val_path = val_path


    args.writer = SummaryWriter(log_dir=exp_path)


    # Logger
    if osp.isfile(log_path):
        os.remove(log_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    logger.info('-'*52 + ' SETUP ' + '-'*52)
    for arg in vars(args):
        logger.info(arg + ' : ' + str(getattr(args, arg)))
    logger.info('-'*111)
    
    logger.info('Start experiment ' + args.name + '!')
    
    train_dataset,_ = utils.build_dataset_trm(is_train=True, args=args)
    
    val_dataset,_ = utils.build_dataset_trm(is_train=False,args=args,gen_attn=True)
    
    logger.info('Train dataset is loaded from ' + args.train_list)
    logger.info('Validation dataset is loaded from ' + args.val_list)
    
    sampler_train = torch.utils.data.RandomSampler(train_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )


    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)
    max_step = train_num_img // args.batch_size * args.epochs
    args.max_step = max_step

    src_file = "./models/model_"+args.model+".py"
    des_file = "./experiments/"+args.name+"/copy_model.py"
    shutil.copy(src_file,des_file)

       
    src_file = "./networks/mctformer.py"
    des_file = "./experiments/"+args.name+"/copy_mctformer.py"
    shutil.copy(src_file,des_file)


    model = getattr(importlib.import_module('models.model_'+args.model), 'model_WSSS')(args, logger)
    model.train_setup()

    logger.info('-' * 111)
    logger.info(('-' * 49) + ' Start Train ' + ('-' * 49))

    miou_list = []
    max_miou = 0
    for epo in range(args.epochs):
        epo_str = str(epo).zfill(3)

        # # Train
        logger.info('-' * 111)
        logger.info('Epoch ' + epo_str + ' train')
        model.set_phase('train')

        for iter, pack in enumerate(tqdm(train_data_loader)):
            model.unpack(pack)
            model.update(epo)
            if iter % args.print_freq == 0 and iter != 0:
                model.print_log(epo + 1, iter / train_num_batch)


        logger.info('Epoch ' + epo_str + ' model is saved!')
        model.save_model(epo, ckpt_path)
      
        model.lr_scheduler.step(epo+1)

        if epo>8:
            # # Validation
            logger.info('-' * 111)
            logger.info('Epoch ' + epo_str + ' validation')
            model.set_phase('eval')
            model.infer_init(epo)
            
            for iter, pack in enumerate(tqdm(val_data_loader)):
                model.unpack(pack)
                model.infer_multi(epo, val_path, dict_path, crf_path, vis=(iter<50), dict=args.dict, crf=args.crf)

            # Evaluate mIoU
            miou_temp, th_temp, Pr, Rc = eval_in_script(logger=logger, eval_list='train', name=args.name, dict_dir='./dict')


            miou_temp_str = str(round(miou_temp,3))
            th_temp_str = str(round(th_temp,3))
            miou_list.append(miou_temp_str)
            logger.info('Epoch ' + epo_str + ' max miou : ' + miou_temp_str + ' at ' + th_temp_str)
            logger.info(miou_list)

            if miou_temp>max_miou:
                max_miou = miou_temp
                logger.info('New record!')

            args.writer.add_scalar("mIoU/current_mIoU",miou_temp,epo)
            args.writer.add_scalar("mIoU/best_mIoU",max_miou,epo)
