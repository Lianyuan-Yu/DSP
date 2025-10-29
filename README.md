# A Diffusion and Spatial Prototype Learning Method for Semi-supervised Medical Image Segmentation

## 1. Environment
- pytorch: 2.4.0+cu118
- win11
- GPU: NVIDIA GeForce RTX 4090

## 2. Data Preparation

The file structure should be: 
```
.
├── Datasets
│   ├── LASeg
│   │   ├── 2018LA_Seg_Training Set
│   │   │   ├── 0RZDK210BSMWAA6467LU
│   │   │   │   ├── mri_norm2.h5
│   │   │   ├── 1D7CUD1955YZPGK8XHJX
│   │   │   └── ...
│   │   ├── test.list
│   │   └── train.list
│   ├── BraTS2019
│   │   │   ├── data
│   │   │   │   ├── BraTS19_2013_0_1.h5
│   │   │   │   └── ...
│   │   │   ├── test.txt
│   │   │   ├── train.list
│   │   │   ├── train_lab.txt
│   │   │   ├── train_unlab.list
│   │   │   └── val.list
```

Run preprocess_la.py

For the Brats2019 dataset, just modify preprocess_la.py accordingly.

## 3. Training & Testing & Evaluating

Training: train.py

Parameters:
```
LA_8: -task 'la' -exp 'LA_8/train' -mu 2.0 -tl 50 -gamma1 0.2 -gamma2 0.8 
```
```
BraTS_25: -task 'brats2019' -exp 'BraTS_25/train' -mu 2.0 -tl 70 -gamma1 0.01 -gamma2 0.7 
```

Testing & Evaluating: test.py & evaluate.py

Parameters:
```
LA_8: -task 'la' -exp 'LA_8/train -speed 1'
```
```
BraTS_25: -task 'brats2019' -exp 'BraTS_25/train -speed 2'
```
