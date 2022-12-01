# Oil-Seep-Detection
Segmenting and Detecting oil seeps from Synthetic Aperture Radar images 

Developed a unique petroleum seep detection technique that involves segmentation and categorisation of oil slicks on the ocean surface contained within Synthetic Aperture Radar (SAR) imagery.

### Background

In normal object detection networks we are interested *what* are the objects so we often use max pooling to reduce the size of the feature map. But with semantic segmentation we are also interested in ther *where* of the objects, so we use up sampling methods, such as transposed convolution.

What is transposed convolution?

It uses as input a low resolution images and outputs high resolution images. To see how it works I invite you to check out this [post](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0).

### UNet Architecture

The [UNet](https://arxiv.org/abs/1505.04597) architecture contains two paths:
1. The first path is the encoder, which its objective is to capture the context of the image. It is a traditional object detection network.
2. The second path is the decoder, which is used to precisely locate the objects.

Input: we are getting $256 \times 256 \times 3$ images.

Our final model contains 1,177,649 trainable parameters.

<p align="center">
  <img src="https://github.com/lavinama/UNet/blob/main/media/unet_arch.jpeg", width=600 />
</p>


### Overfitting


### Parameters
