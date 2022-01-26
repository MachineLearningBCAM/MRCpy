# -*- coding: utf-8 -*-
"""
.. _featureextraction:

MRCs with Deep Neural Networks: Part I
===========================================================
In this example we will use a pretrained neural network to extract features
of images in a dataset to train and test MRCs with these features in
:ref:`feature_mrc`.

We are using `ResNet18 <https://pytorch.org/hub/pytorch_vision_resnet/>`_
pretrained model implementation in Pytorch library. Resnet models were proposed
in “Deep Residual Learning for Image Recognition”. Here we are using the
version ResNet18 which contains 18 layers and it is pretrained over
`ImageNet dataset<https://www.image-net.org/index.php>` which has 1000
different classes.

.. seealso:: For more information about ResNet models refer to

                [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016).
                Deep residual learning for image recognition.
                In Proceedings of the IEEE conference on computer
                vision and pattern recognition (pp. 770-778).
"""


##############################################################################
# Introduction to Pretrained models, Transfer Learning and Feature Extraction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Deep convolutional neural network models may take days or even weeks
# to train on very large datasets. A way to short-cut this process is
# to re-use the model weights from *pre-trained models* that were developed
# for standard computer vision benchmark datasets, such as the
# ImageNet image recognition tasks.
# Top performing models can be downloaded and used directly, or integrated
# into a new model for your own computer vision problems.
# *Transfer learning* generally refers to a process where a model trained
# on one problem is used in some way on a second related problem.
# Alternately, the pretrained models may be used as feature extraction models.
# Here, the output of the model from a layer prior to the output layer
# of the model is used as input to a new classifier model.

###########################################################################
# Load pretrained model and preprocess images
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Firstly, we load the pretrained model `torchvision.models.ResNet18` and
# we take out the last layer in order to obtain the features. We call this
# feature extraction model `resnet18_features`.
#
# In the next lines we use `torchvision.transforms.Compose` to compose several
# transforms together. In line [2] the input is resized to match its smaller
# edge to the given size, 256. That is, the image is resized mantaining
# the aspect ratio and `min(height,width)=256`.
# In line [3] we use the function `CenterCrop` to crop the given image
# at the center to `(224,224)`.  If image size is smaller than output
# size along any edge, image is padded with 0 and then center cropped.
#
# If you use smaller images, the kernels might not be able to extract the
# features with the usual size, since they are smaller (ore larger),
# which may result in a difference in performance.
#
# Function in line [4] converts a PIL Image
# or numpy.ndarray to tensor. Finally `Normalize` function (lines [5,6])
# normalizes a tensor image with mean and standard deviation required by
# ResNet18 method (check more in
# `pytorch ResNet18 doc<https://pytorch.org/hub/pytorch_vision_resnet/>`).
# Given mean:
# `(mean[1],...,mean[n])` and std: `(std[1],..,std[n])` for `n` channels,
# this transform will normalize each channel i.e.,
# `output[channel] = (input[channel] - mean[channel]) / std[channel]`
# You can check more about `torchvision.transforms` in
# `pytorch docummentation<https://pytorch.org/vision/stable/transforms.html>`.

import os
from os.path import join

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from MRCpy.datasets import load_yearbook_path

resnet18 = models.resnet18(pretrained=True)
features_resnet18 = nn.Sequential(*(list(resnet18.children())[:-1]))
features_resnet18.eval()

transform = transforms.Compose(                              # [1]
    [transforms.Resize(256),                                 # [2]
     transforms.CenterCrop(224),                             # [3]
     transforms.ToTensor(),                                  # [4]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],        # [5]
                          std=[0.229, 0.224, 0.225])])       # [6]

#####################################################################
# Using tensorflow datasets: MNIST & Cats vs Dogs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# MNIST
# -----
# The MNIST database of handwritten digits, available from
# `this page<http://yann.lecun.com/exdb/mnist/>`,
# has a training set of 60000 examples, and a test set of 10000 examples. All
# images have dimension (28,28,1) and they are greyscale. Tensorflow provides
# with a convenient function to directly load this dataset into the scope
# without the need of downloading and storing the dataset locally, you can
# check more in `tensorflow documentation
# <https://www.tensorflow.org/datasets/catalog/mnist>`_.
# It already provides with the train and test partitions. We load the dataset
# with the function `tensorflow_datasets.load` and we specify
# `as_supervised=True` to indicate that we want to load the labels together
# with the images and `with_info=True` will return the tuple
# `(tf.data.Dataset, tfds.core.DatasetInfo)`,
# the latter containing the info associated with the builder.

[[ds_train, ds_test], ds_info] = tfds.load('mnist', split=['train', 'test'],
                                           as_supervised=True, with_info=True)

df_train = tfds.as_dataframe(ds_train, ds_info)
df_test = tfds.as_dataframe(ds_test, ds_info)

images_train = df_train['image'].to_numpy()
Y_train = df_train['label'].to_numpy()
images_test = df_test['image'].to_numpy()
Y_test = df_test['label'].to_numpy()

X_train = []
X_test = []


for img_array in images_train:
    # We convert the gray scale into RGB because it is what the model expect
    img_array = np.repeat(img_array, 3, axis=-1)
    img = Image.fromarray(img_array, mode='RGB').resize((224, 224))
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    X_train.append(features_resnet18(batch_t).detach().numpy().flatten())

for img_array in images_test:
    # We convert the gray scale into RGB because it is what the model expect
    img_array = np.repeat(img_array, 3, axis=-1)
    img = Image.fromarray(img_array, mode='RGB').resize((224, 224))
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    X_test.append(features_resnet18(batch_t).detach().numpy().flatten())

mnist_features_resnet18_train = np.concatenate(
    (X_train, np.reshape(Y_train, (-1, 1))), axis=1)

mnist_features_resnet18_test = np.concatenate(
    (X_test, np.reshape(Y_test, (-1, 1))), axis=1)

np.savetxt('mnist_features_resnet18_train.csv', mnist_features_resnet18_train,
           delimiter=',')
np.savetxt('mnist_features_resnet18_test.csv', mnist_features_resnet18_test,
           delimiter=',')

#####################################################################
# Cats vs Dogs
# ------------
# Cats vs dogs dataset is a database of 23262 RGB cats
# and dogs images released by Microsoft for the Asirra captcha (`homepage
# <https://www.microsoft.com/en-us/download/details.aspx?id=54765>`_).
# Cats are labeled by 0 and dogs by 1 and there are 11658 and 11604 images
# of each class, respectively.
# It is available in tensorflow datasets, you can check the details `here
# <https://www.tensorflow.org/datasets/catalog/cats_vs_dogs>`_.

[ds, ds_info] = tfds.load('cats_vs_dogs', split='train',
                          as_supervised=True, with_info=True)

df = tfds.as_dataframe(ds, ds_info)
images = df['image'].to_numpy()
labels = df['label'].to_numpy()

X_features = []
for img_array in images:
    img = Image.fromarray(img_array, mode='RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    X_features.append(features_resnet18(batch_t).detach().numpy().flatten())

catsvsdogs_features_resnet18 = np.concatenate((X_features,
                                               np.reshape(labels, (-1, 1))),
                                              axis=1)

np.savetxt('catsvsdogs_features_resnet18.csv', catsvsdogs_features_resnet18,
           delimiter=',')

#####################################################################
# Using a local dataset: Yearbook Dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this example, we are going to extract the features from a local dataset.
# We will be using the Yearbook dataset which is a publicly-available dataset
# of 37,921 frontal-facing American high school yearbook portraits taken from
# 1905 to 2013 labeled by gender.
# We will consider binary classification labels identifying
# whether the person on the image is a man or a woman.
#
# .. seealso:: More information about Yearbook dataset can be found in
#
#               [1] Ginosar, S., Rakelly, K., Sachs, S., Yin, B., & Efros,
#               A. A. (2015). A century of portraits: A visual historical
#               record of american high school yearbooks. In Proceedings of
#               the IEEE International Conference on Computer Vision Workshops
#               (pp. 1-7).
#
#               [2] Kumar, A., Ma, T., & Liang, P. (2020, November).
#               Understanding self-training for gradual domain adaptation.
#               In International Conference on Machine Learning
#               (pp. 5468-5479). PMLR.


##################################################################
# We take paths and names from images from F (female) and M (male)
# folder and merge them in a dataset ordered by date
# (as images name start by the year of the photo). We convert the labels into
# 0 for F and 1 for M.


data_path = load_yearbook_path()
F_path = join(data_path, 'F')
F = os.listdir(F_path)
F = np.concatenate((np.reshape(F, (len(F), 1)), np.zeros((len(F), 1)),
                    np.reshape([F_path + x for x in F], (len(F), 1))), axis=1)
M_path = join(data_path, 'M')
M = os.listdir(M_path)
M = np.concatenate((np.reshape(M, (len(M), 1)), np.ones((len(M), 1)),
                    np.reshape([M_path + x for x in M], (len(M), 1))), axis=1)
data = np.concatenate((F, M), axis=0)
data = data[np.argsort(data[:, 0])]

paths = data[:, 2]
Y = data[:, 1]


###########################################################################
# Next, we load the images, transform them using the function `transform` we
# defined above to make the image compatible with ResNet18. Lastly, we extract
# the image features using `features_resnet18()` and we transform the output
# features to a flat array that will be a new instance of our feature dataset.
# We store this feature dataset extracted with resnet18 in a csv file that
# is available in the `dataset` folder of the MRCpy library.

X_features = []

for img_path in paths:
    img = Image.open(img_path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    X_features.append(features_resnet18(batch_t).detach().numpy().flatten())

yearbook_features_resnet18 = np.concatenate((X_features,
                                             np.reshape(Y, (-1, 1))), axis=1)
np.savetxt('yearbook_features_resnet18.csv',
           yearbook_features_resnet18, delimiter=',')
