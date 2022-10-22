## 3D Medical Image Segmentation using Parallel Transformers  
This is the official pytorch implementation of the TransHRNet
![](https://github.com/duweidai/TransHRNet/blob/main/images/network.jpg)

## Requirements
CUDA 11.0

Python 3.7

Pytorch 1.7

Torchvision 0.8.2

## Usage
### 1. Data Preparation
* Download BCV dataset (https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
* Preprocess the BCV dataset use the nnUNet package.

      cd nnUNet/nnunet/dataset_conversion 
      Run ```python Task017_BeyondCranialVaultAbdominalOrganSegmentation.py```
* Training and Testing ID are in data/splits_final.pkl

### 2. Training
cd TransHRNet_package/TransHRNet/run
* Run ``` python run_training.py -gpu="0" -outpath="TransHRNet" ``` for training.

### 3. Testing
cd TransHRNet_package/TransHRNet/run
* Run ``` python run_training.py -gpu='0' -outpath="TransHRNet" -val --val_folder='validation_result' ``` for validation.

### 4. Performance on Multi-Atlas Labeling Beyond the Cranial Vault (BCV) dataset

![](https://github.com/duweidai/TransHRNet/blob/main/images/performance_1.jpg)

## Acknowledge 
Part of codes are reused from the CoTr (https://github.com/YtongXie/CoTr). Thanks to Fabian Isensee for the codes of CoTr.


