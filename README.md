## 3D Medical Image Segmentation using Parallel Transformers  
This is the official pytorch implementation of the TransHRNet
![](https://github.com/duweidai/TransHRNet/blob/main/images/network.jpg)

## Requirements
CUDA 11.O
Python 3.7
Pytorch 1.7
Torchvision 0.8.2

## Usage
# 1. Data Preparation
* Download BCV dataset (https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
* Preprocess the BCV dataset use the nnUNet package.
* Training and Testing ID are in data/splits_final.pkl

# Training
cd TransHRNet_package/TransHRNet/run
* Run ``` python run_training.py -gpu='0' ``` for training.
```


### Train and Test

We provide all code related to the network structure, and the training and testing of the model are consistent with CoTr.

Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation (https://github.com/YtongXie/CoTr).

That is to say, as long as the network structure in the CoTr project is replaced with the code we provide, it will take effect.

### Performance on Multi-Atlas Labeling Beyond the Cranial Vault  dataset

![](https://github.com/duweidai/TransHRNet/blob/main/images/performance_1.jpg)

### Performance on Medical Segmentation Decathlon (MSD) dataset for brain tumor  

![](https://github.com/duweidai/TransHRNet/blob/main/images/performance_2.jpg)



