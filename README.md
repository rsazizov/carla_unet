# CARLA UNet

This repo contains my implementation of UNet architecture for self-driving car semantic segmentation challenge dataset on [kaggle](https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge).

## Results

![prediction](https://github.com/rsazizov/carla_unet/raw/master/images/prediction.png)

The model achieves a mean IoU score of ~0.74 on a test set. Possible improvements include data augmentation and longer training.

## CARLA

[CARLA](https://carla.org/) is an open source self-driving car simulator. It allows to run SITL simulations and collect artificial data (including semantic segmentation camera).

## Semantic Segmentation

To put it simply, semantic segmentation is image classification performed on individual pixels. Each pixel of an image gets classified as a class (road, car, road sign, etc...). The key to semantic segmentation is FCN (Fully Convolutional Neural Network), that takes an image and outputs another image, whose pixels are assigned classes.


## UNet

Due to pooling and strided convolutions, it is hard for FCNs to restore spatial information from the "bottleneck". U-Net [O. Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597) solves this problem by croppping feature maps from the contracting part and feeding them to the corresponding layers of the expansion part. This enables the expansion part to recover spatial information more easily.

![U-Net](https://github.com/rsazizov/carla_unet/raw/master/images/u-net.png)
