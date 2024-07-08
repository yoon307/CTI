# CTI
### Official repository for CVPR 2024 paper: [Class Tokens Infusion for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/papers/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf) by Sung-Hoon Yoon, Hoyong Kwon, Hyeonseong Kim, and Kuk-Jin Yoon. 
---

# 1.Prerequisite
## 1.1 Environment
* Tested on Ubuntu 20.04, with Python 3.9, PyTorch 1.8.2, CUDA 11.7, multi gpus(2) - Nvidia RTX 3090.
* If you encounter OOM, try to reduce the batchsize (32) - but not checked.

* You can create conda environment with the provided yaml file.
```
conda env create -f environment.yaml
```

## 1.2 Dataset Preparation
* [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
You need to specify place VOC2012 under ./data folder.
- Download MS COCO images from the official COCO website [here](https://cocodataset.org/#download).
- Download semantic segmentation annotations for the MS COCO dataset [here](https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view?usp=sharing). (Refer [RIB](https://github.com/jbeomlee93/RIB))

- Directory hierarchy 
```
    ./data
    ├── VOC2012       
    └── COCO2014            
            ├── SegmentationClass     # GT dir             
            ├── train2014  # train images downloaded from the official COCO website 
            └── val2014    # val images downloaded from the official COCO website
```




* ImageNet-pretrained weights for ViT are from [deit_small_imagenet.pth](https://drive.google.com/drive/folders/1cX6qk0n9mnf6avE81Dx7JPBJp00E5t_c?usp=drive_link).  
**You need to place the weights as "./pretrained/deit_small_imagenet.pth. "**

# 2. Usage
> With the following code, you can generate CAMs (seeds) to train the segmentation network.
> For the further refinement, refer [RIB](https://github.com/jbeomlee93/RIB). 

> We will also update the RIB (transformer version) soon (July,2024)
>

## 2.1 Training
* Please specify the name of your experiment.
* Training results are saved at ./experiment/[exp_name]

For PASCAL:
```
python train_trm.py --name [exp_name] --exp cti_cvpr24
```
For COCO:
```
python train_trm_coco.py --name [exp_name] --exp cti_coco_cvpr24
```

**Note that the mIoU in COCO training set is evaluated on the subset (5.2k images, not the full set of 80k images) for fast evaluation**

## 2.2 Inference (CAM)
* Pretrained weight (PASCAL, seed: 69.5% mIoU) can be downloaded [here](https://drive.google.com/drive/folders/1cX6qk0n9mnf6avE81Dx7JPBJp00E5t_c) (69.5_pascal.pth).

For pretrained model (69.5%):
```
python infer_trm.py --name [exp_name] --load_pretrained [DIR_of_69.5%_ckpt] --load_epo 100 --dict
```

For model you trained:

```
python infer_trm.py --name [exp_name] --load_epo [EPOCH] --dict
```

## 2.3 Evaluation (CAM)
```
python evaluation.py --name [exp_name] --task cam --dict_dir dict
```


# 3. Additional Information
## 3.1 Paper citation
If our code be useful for you, please consider citing our CVPR 2024 paper using the following BibTeX entry.
```
@inproceedings{yoon2024class,
  title={Class Tokens Infusion for Weakly Supervised Semantic Segmentation},
  author={Yoon, Sung-Hoon and Kwon, Hoyong and Kim, Hyeonseong and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3595--3605},
  year={2024}
}
```
You can also check our earlier works published on ICCV 2021 ([OC-CSE](https://openaccess.thecvf.com/content/ICCV2021/papers/Kweon_Unlocking_the_Potential_of_Ordinary_Classifier_Class-Specific_Adversarial_Erasing_Framework_ICCV_2021_paper.pdf)) , ECCV 2022 ([AEFT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890323.pdf)), CVPR 2023 ([ACR](https://openaccess.thecvf.com/content/CVPR2023/papers/Kweon_Weakly_Supervised_Semantic_Segmentation_via_Adversarial_Learning_of_Classifier_and_CVPR_2023_paper.pdf))

### Beside, in ECCV 24, **"Diffusion-Guided Weakly Supervised Semantic Segmentation"** and **"Phase Concentration and Shortcut Suppression for Weakly Supervised Semantic Segmentation"** will be published. Check our github! :)

## 3.2 References
We heavily borrow the work from [MCTformer](https://github.com/xulianuwa/MCTformer)  and [RIB](https://github.com/jbeomlee93/RIB) repository. Thanks for the excellent codes!
```
[1] Xu, Lian, et al. "Multi-class token transformer for weakly supervised semantic segmentation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
[2] Lee, Jungbeom, et al. "Reducing information bottleneck for weakly supervised semantic segmentation." Advances in neural information processing systems 34 (2021): 27408-27421.
