# Oil-Seep-Detection
Segmenting and Detecting oil seeps from Synthetic Aperture Radar images 

Developed a unique petroleum seep detection technique that involves segmentation and categorisation of oil slicks on the ocean surface contained within Synthetic Aperture Radar (SAR) imagery.

### Table of Contents
1. [Dataset](#dataset)
2. [Background](#background)
3. [UNet Architecture](#unet-architecture)
4. [Training](#training)
5. [Loss functions](#loss-functions)
6. [Results](#results)

### Dataset

<p align="center">
  <img src="https://github.com/lavinama/Oil-Seep-Detection/blob/main/media/example_dataset.png", width=500 />
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
  * We use the Adam optimiser because combines the advantage of the Adagrad and RMSProp:
    * Adagrad: good performance on sparse data e.g. computer vision and natural languange processing
    * RMSprop: good performance on online and non-stationary problems e.g. noisy data
* We reduce the learning rate by a factor of 0.1 if the validation loss does not improve for 5 continues epochs reaching a minimum of 0.00001 `ReduceLROnPlateau`.
* We apply early stopping if the validation loss does not improve for 10 continuous epochs `EarlyStopping`.
* Hyperparameters:
  * Batch size: 32
  * Dropout: 0.05

### Loss functions

In this project we are going to evaluate different loss functions.

As we see from the sample given above the background class (the non-seep) is much larger than the other seep classes (more than 90% of the pixels). We can fall into a trap when using accuracy as our evaluation metric. For example, if we have a model that predicts all the pixels being non-seep we are going to get an accuracy of more than 90% but not a very useful model for segmenting seeps. Hence, we are going different loss functions to overcome this class imbalance.

#### Binary Cross Entropy (BCE)

```math
BCE(\hat{p}, p) = -\left(p \cdot \log(p) + (1 - \hat{p}) \cdot \log(1 - p)\right)
```
This is the simplest loss function when dealing with two classes. The idea is to have a loss function that predicts a high probability for a positive example, and a low probability for a negative example. The problem with BCE is that if we miss all the seeps which only account for a minority, we still receive a low loss.

#### Binary Focal Loss (FL)

```math
FL(\hat{p}) = -\alpha(1 - \hat{p})^{\gamma} \cdot \log(\hat{p}))
```
This loss function generalizes binary cross-entropy by introducing a hyperparameter called the focusing parameter $\gamma$ that allows hard-to-classify examples to be penalized more heavily relative to easy-to-classify examples. The idea is that if a sample is already well-classified, we can significantly decrease its contribution to the loss.

To solve the class imbalance problem, it incorporates a weighting parameter $\alpha$, which is usually the inverse class frequency. $\alpha$ is the weighted term whose value is $\alpha$ for the seep class and $1 - \alpha$ for the non-seep class.

Parameters used: $\gamma = 0.75$

#### Tversky Loss

This loss deals with imbalanced datasets by utilising constants that can adjust how harshly different types of error are penalised in the loss function. The Tversky loss uses the Tversky Index (TI):
```math
TI = \frac{TP}{TP + \alpha FN + \beta FP}
```
The loss function is weighted by the constants $\alpha$ and $\beta$ that penalise false positives (FP) and false negatives (FN) respectively to a higher degree in the loss function as their value is increased. The $\beta$ constant in particular has applications in situations where models can obtain misleadingly positive performance via highly conservative prediction.

Parameters used: $\alpha = 0.7$,  $\beta = 0.3$

#### Focal Tversky Loss

```math
FTL = (1 - TI)^{\gamma}
```
Focal Tversky loss is a combination of the Binary Focal loss and the Tversky loss.

Parameters used: $\gamma = 0.75$

*Extension: For the classification task we would have to use variations of the loss functions for multiclass classification such as Sparse Categorical Cross Entropy or Sparse Categorical Focal Loss.*

### Results

Given that for each pixel we get a value between 0 to 1, where 0 represents no seep and 1 represents seep. We take 0.5 as the threshold to decide whether to classify a pixel as non-seep or seep. This output is called Seep Predicted binary.

| Loss Function             | Accuracy  | Loss  |
| :------------             |:---------:| :----:|
| **Binary Cross Entropy**  | 0.9856    | 0.0515|
| **Binary Focal Loss**     | 0.9855    | 0.0296|
| **Focal Tversky Entropy** | 0.9842    | 0.5287|

From the results the numbers we don’t get very conclusive results but if we compare the outputs, for example: we see that the Focal Tversky Loss is more “generous” when detecting seeps. Given that we want to obtain a high recall (most of the seeps are recognised, i.e., we want a low number of false negatives), our best model is the UNet that uses the Focal Tversky loss function.

One possible extension of this project would be to calculate the Dice coefficient, the precision and the recall of each of the models to support our conclusion.

#### Binary Cross Entropy
<p align="center">
  <img src="https://github.com/lavinama/Oil-Seep-Detection/blob/main/media/results/bce_result.png", width=800 />
</p>

#### Binary Focal Loss
<p align="center">
  <img src="https://github.com/lavinama/Oil-Seep-Detection/blob/main/media/results/focal_result.png", width=800 />
</p>

#### Focal Tversky Entropy
<p align="center">
  <img src="https://github.com/lavinama/Oil-Seep-Detection/blob/main/media/results/tversky_focal_result.png", width=800 />
</p>
