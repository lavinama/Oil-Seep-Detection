# Oil-Seep-Detection
Segmenting and Detecting oil seeps from Synthetic Aperture Radar images 

Developed a unique petroleum seep detection technique that involves segmentation and categorisation of oil slicks on the ocean surface contained within Synthetic Aperture Radar (SAR) imagery.

### Dataset

<p align="center">
  <img src="https://github.com/lavinama/UNet/blob/main/media/preview_data.png", width=400 />
</p>

There are two possible tasks that can be carried out using this dataset:
* **Segmentation:** we are only interested in separating the pixels which are seeps and those who aren’t. Hence, the type of seep it is becomes redundant. So, we simplify the mask as 0 for non-seep and 1 for seep.
* **Classification:** we are not only interested in classifying pixels as seep or non-seep but also what type of seep they are, so we do not alter the mask.

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

The architecture that we designed is the following (note: you can see that the first dimension is `None` that is because that dimension is equal to the batch size): 
<p align="center">
  <img src="https://github.com/lavinama/Oil-Seep-Detection/blob/main/media/model-seep-focal-mask-0-1.png", width=300 />
</p>

Advantages of UNet:
* The network is image size agnostic since it does not contain fully connected layers, this means that the model is of smaller weight size.
* Can be easily scaled ot multiple classes.
* Works well with small datasets, thanks to the robustness provided with data augmentation

Disadvantages of UNet:
* The size of the UNet should be similar to the size of the features (need context of the images).
* High number of layers means that it takes time to train.

There are other methods for semantic segmentation such as: FCN-VGG16, DeepLab, Deconvnet, U-Net, DialatedNet, GCN, PSPNet, FC-DenseNet103, EncNet, Gated-SCNN. For more information visit this [link](https://arxiv.org/pdf/2001.04074.pdf).

### Training

* Evaluation Metric: Accuracy
* Optimiser: Adam Optimiser
* We reduce the learning rate by a factor of 0.1 if the validation loss does not improve for 5 continues epochs reaching a minimum of 0.00001 `ReduceLROnPlateau`.
* We apply early stopping if the validation loss does not improve for 10 continuous epochs `EarlyStopping`.
* Hyperparameters:
  * Batch size: 32
  * Dropout: 0.05

### Loss functions

In this project we are going to evaluate different loss functions. 

