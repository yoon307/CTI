import os
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import importlib
import argparse
import tools.utils as utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import pdb

import csv

################################################################################
# Evaluate the performance by computing mIoU.
# It assumes that every CAM or CRF dict file is already infered and saved.  
# For CAM, threshold will be searched in range [0.01, 0.80].
#
# If you want to evaluate CAM performance...
# python evaluation.py --name [exp_name] --task cam --dict_dir dict
#
# Or if you want to evaluate CRF performance of certain alpha (let, a1)...
# python evaluation.py --name [exp_name] --task crf --dict_dir crf/a1
#
# For AFF evaluation, go to evaluation_aff.py
################################################################################


# categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
            #   'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

coco_categories = ['background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 
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

def load_label(predict_folder, name, num_cls):
    predict_file = os.path.join(predict_folder,'%s.npy'%name)
    predict_dict = np.load(predict_file, allow_pickle=True).item()
    h, w = list(predict_dict.values())[0].shape
    tensor = np.zeros((num_cls,h,w),np.float32)
    return tensor, predict_dict

# def load_label(predict_folder, name, num_cls): #RIB
#     predict_file = os.path.join(predict_folder,'%s.npy'%name[15:])
#     predict_dict = np.load(predict_file, allow_pickle=True).item()
    
#     return predict_dict


def do_python_eval(predict_folder, gt_folder, name_list, num_cls, task, threshold, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('L', 0, lock=True))
        P.append(multiprocessing.Value('L', 0, lock=True))
        T.append(multiprocessing.Value('L', 0, lock=True))
    
    def compare(start,step,TP,P,T,task,threshold):
        # for idx in tqdm(range(start,len(name_list),step)):
        for idx in range(start,len(name_list),step):
            name = "%012d"%(name_list[idx])

            if task=='cam':
                tensor, predict_dict = load_label(predict_folder, name, num_cls)
                for key in predict_dict.keys():
                    tensor[key+1] = predict_dict[key]
                tensor[0,:,:] = threshold
                predict = np.argmax(tensor, axis=0).astype(np.uint8)
            
            # if task=='cam':
            #     cam_dict = np.load(os.path.join(predict_folder,'%s.npy'%name[15:]), allow_pickle=True).item()
            #     cams = cam_dict['high_res']
            #     cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            #     keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            #     predict = np.argmax(cams, axis=0)
            #     predict = keys[predict]

            if task=='crf':
                # tensor, predict_dict = load_label(predict_folder, name, num_cls)
                # for key in predict_dict.keys():
                #     tensor[key] = predict_dict[key]
                # predict = np.argmax(tensor, axis=0).astype(np.uint8)
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict = np.load(predict_file).astype(np.uint8)

            if task=='dl':
                tensor, predict_dict = load_label(predict_folder, name, num_cls)
                for key in predict_dict.keys():
                    tensor[key] = predict_dict[key]
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            if task=='png':
                predict_file = os.path.join(predict_folder, '%s.png' % name)
                predict = np.array(Image.open(predict_file))

            if "train" in gt_folder:
                gt_file = os.path.join(gt_folder,'COCO_train2014_%s.png'%name)
            else:
                gt_file = os.path.join(gt_folder,'COCO_val2014_%s.png'%name)

            gt = np.array(Image.open(gt_file))
            # gt_file = os.path.join(gt_folder,'%s.npy'%name)
            # gt = np.load(gt_file)
            cal = gt<255 # Reject object boundary
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(os.cpu_count()//2):
        p = multiprocessing.Process(target=compare, args=(i,os.cpu_count()//2,TP,P,T,task,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[coco_categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(coco_categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(coco_categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
    
    if False:
        f = open('./experiments/dl_recon_48.1/coco_val.csv','w',newline='')
        wr = csv.writer(f)
        class_name =[]
        class_iou = [] 
        for i in range(num_cls):
           class_name.append(coco_categories[i])
        wr.writerow(class_name)
        for i in range(num_cls):
            class_iou.append(IoU[i]*100)
        wr.writerow(class_iou)
        f.close()
    return loglist

def do_python_eval_scales(predict_folder, gt_folder, name_list, num_cls, task, threshold, printlog=False):
    TP = []
    P = []
    T = []
    for j in range(5):
        TP.append([])
        P.append([])
        T.append([])
        for i in range(num_cls):
            TP[j].append(multiprocessing.Value('I', 0, lock=True))
            P[j].append(multiprocessing.Value('I', 0, lock=True))
            T[j].append(multiprocessing.Value('I', 0, lock=True))
    scale_list = ['0.5', '1.0', '1.5', '2.0', 'total']

    def compare(start,step,TP,P,T,task,threshold):
        for idx in range(start,len(name_list),step):
            for j, scale in enumerate(scale_list):
                name = name_list[idx]
                name_scale = name + '_' + scale

                if task=='cam':
                    tensor, predict_dict = load_label(predict_folder, name_scale, num_cls)
                    for key in predict_dict.keys():
                        tensor[key+1] = predict_dict[key]
                    tensor[0,:,:] = threshold
                    predict = np.argmax(tensor, axis=0).astype(np.uint8)

                if task=='crf':
                    tensor, predict_dict = load_label(predict_folder, name_scale, num_cls)
                    for key in predict_dict.keys():
                        tensor[key] = predict_dict[key]
                    predict = np.argmax(tensor, axis=0).astype(np.uint8)

                if task=='dl':
                    tensor, predict_dict = load_label(predict_folder, name_scale, num_cls)
                    for key in predict_dict.keys():
                        tensor[key] = predict_dict[key]
                    predict = np.argmax(tensor, axis=0).astype(np.uint8)

                if task=='png':
                    predict_file = os.path.join(predict_folder, '%s.png' % name)
                    predict = np.array(Image.open(predict_file))

                gt_file = os.path.join(gt_folder,'%s.png'%name)
                gt = np.array(Image.open(gt_file))
                cal = gt<255 # Reject object boundary
                mask = (predict==gt) * cal
        
                for i in range(num_cls):
                    P[j][i].acquire()
                    P[j][i].value += np.sum((predict==i)*cal)
                    P[j][i].release()
                    T[j][i].acquire()
                    T[j][i].value += np.sum((gt==i)*cal)
                    T[j][i].release()
                    TP[j][i].acquire()
                    TP[j][i].value += np.sum((gt==i)*mask)
                    TP[j][i].release()
    p_list = []
    for i in range(8):
        # p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,task,threshold))
        p = multiprocessing.Process(target=compare, args=(i,os.cpu_count()//2,TP,P,T,task,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for j in range(5):
        IoU.append([])
        T_TP.append([])
        P_TP.append([])
        FP_ALL.append([])
        FN_ALL.append([])
        for i in range(num_cls):
            IoU[j].append(TP[j][i].value/(T[j][i].value+P[j][i].value-TP[j][i].value+1e-10))
            T_TP[j].append(T[j][i].value/(TP[j][i].value+1e-10))
            P_TP[j].append(P[j][i].value/(TP[j][i].value+1e-10))
            FP_ALL[j].append((P[j][i].value-TP[j][i].value)/(T[j][i].value + P[j][i].value - TP[j][i].value + 1e-10))
            FN_ALL[j].append((T[j][i].value-TP[j][i].value)/(T[j][i].value + P[j][i].value - TP[j][i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[4][i] * 100

    for j in range(len(scale_list)):
        miou = np.mean(np.array(IoU[j]))
        loglist['mIoU_' + scale_list[j]] = miou * 100

    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[4][i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[4][i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU_total',miou*100))

    
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


# def eval_in_script(logger=None, eval_list='train', task='cam', name=None, dict_dir=None, gt_dir='./data/VOC2012/SegmentationClass'):
    
#     eval_list = './data/VOC2012/ImageSets/Segmentation/' + eval_list + '.txt'
#     df = pd.read_csv(eval_list, names=['filename'])
#     name_list = df['filename'].values
#     pred_dir = osp.join('./experiments', name, dict_dir)  

#     max_miou = 0
#     max_th = 0
#     for i in range(30):
#         t = i/100.+0.05
#         loglist = do_python_eval(pred_dir, gt_dir, name_list, 81, task, t, printlog=False)
#         #loglist = do_python_eval_scales(pred_dir, gt_dir, name_list, 21, task, t, printlog=False)
#         logger.info('%d/60 threshold: %.3f\tmIoU: %.3f%%' %(i, t, loglist['mIoU']))

#         miou_temp = loglist['mIoU']
#         if miou_temp>max_miou:
#             max_miou = miou_temp
#             max_th = t

#     return max_miou, max_th

def eval_in_script_coco(logger=None, eval_list=None, task='cam', name=None, dict_dir=None, gt_dir='./data/COCO2014/SegmentationClass/train2014'):
    global categories
    categories = coco_categories
    # eval_list = eval_list
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values

    pred_dir = osp.join('./experiments', name, dict_dir)  

    max_miou = 0
    max_th = 0
    for i in range(32,36):
        t = i/100.
        loglist = do_python_eval(pred_dir, gt_dir, name_list, 81, task, t, printlog=False)
        logger.info('%d/60 threshold: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
        
        miou_temp = loglist['mIoU']
        if miou_temp>max_miou:
            max_miou = miou_temp
            max_th = t
        else:
            break

    return max_miou, max_th

def eval_in_script_dl(logger=None, eval_list='val', task='png', name=None, dict_dir=None, gt_dir='./data/COCO2014/SegmentationClass/val2014'):
    
    # eval_list = './data/COCO2014/SegmentationClass/' + eval_list + '.txt'
    eval_list = './coco14/' + eval_list + '.txt'
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values
    pred_dir = osp.join('./experiments', name, dict_dir)  

    loglist = do_python_eval(pred_dir, gt_dir, name_list, 81, task, 0, printlog=False)

    return loglist

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--list", default="trainsub", type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--dict_dir", required=True, type=str)
    # parser.add_argument("--gt_dir", default="/home/vilab/ksj/wsss_ssl_coco/coco/val2014_old", type=str)
    # parser.add_argument("--gt_dir", default='./coco/SegmentationClass/val2014', type=str)
    parser.add_argument("--gt_dir", default='./coco/SegmentationClass/train2014', type=str)

    parser.add_argument("--model", default='moco', type=str)
    parser.add_argument("--dict", action='store_false')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--alphas", default=[6, 10, 24], nargs='+', type=int)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--iterative", action='store_true')

    # Hyper-parameters
    parser.add_argument("--D", default=256, type=int)
    parser.add_argument("--M", default=0.997, type=float)
    parser.add_argument("--TH", default=0.2, type=float)
    parser.add_argument("--T", default=2.0, type=float)
    parser.add_argument("--W", default=[1.0, 1.0, 1.0], nargs='+', type=float)
    parser.add_argument("--MEM", default=5, type=int)

    # Learning rate
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--max_epochs", default=40, type=int)
    
    args = parser.parse_args()

    # eval_list = './coco/img_name_' + args.list + '.txt'
    eval_list = './coco14/' + args.list + '.txt'  
    df = pd.read_csv(eval_list, names=['filename'])
    name_list = df['filename'].values
    pred_dir = osp.join('./experiments', args.name, args.dict_dir)  

    print('Evaluate ' + pred_dir + ' with ' + eval_list)

    if args.task=='cam':
        for i in range(30):
            t = i/100.
            loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 81, args.task, t, printlog=False)
            print('%d/60 threshold: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
    
    elif args.task=='crf':
        loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 81, args.task, 0, printlog=True)

    elif args.task=='dl'or args.task=='png':
        loglist = do_python_eval(pred_dir, args.gt_dir, name_list, 81, args.task, 0, printlog=True)
