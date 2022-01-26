# -*- coding: utf-8 -*-
"""
.. _feature_mrc:

Computer vision: Image Classification using extracted features
===============================================================

In this example we will use a features extracted from different sets of images
using pretrained neural networks, as explained in :ref:`Computer vision: Feature
extraction for image classification <featureextraction>`_

We will use image features correponding to a set of training images to train an
MRC model to then predict the class of a set of test images using their
correponding extracted features.
"""


import numpy as np
from sklearn.model_selection import train_test_split
from MRCpy import MRC
from MRCpy.datasets import load_catsvsdogs_features_resnet18,
                            load_yearbook_features_resnet18


##############################################################################
# Cats vs Dogs Dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Cats vs dogs dataset is a database of 23262 RGB cats
# and dogs images released by Microsoft for the Asirra captcha (`homepage
# <https://www.microsoft.com/en-us/download/details.aspx?id=54765>`_).
# Cats are labeled by 0 and dogs by 1 and there are 11658 and 11604 images
# of each class, respectively. We are using the features extracted using
# a pretrained ResNet18 netowork over ImageNet.
#
# For comparison purposes, in this tutorial they obtain accuracy of 97% for
# this task using a pretrained VGG16 network together with some more deep
# neural layers. 


X, Y = load_catsvsdogs_features_resnet18()

X_train, X_test, Y_train, Y_test = train_test_split(
  X, Y, test_size=0.25, random_state=42)


clf = MRC(phi='linear').fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
error = np.average(Y_pred!=Y_test)
print('Cats vs Dogs accuracy error: ' + str(1 - error))


##############################################################################
# Yearbook Dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Yearbook dataset which is a publicly-available dataset
# of 37,921 frontal-facing American high school yearbook portraits taken from
# 1905 to 2013 labeled by gender.
# We will perform binary classification. We want to predict
# whether the person on the image is a man or a woman.
#
# We wil train an MRC with two different settings: training with the first 2000
# images and training with the first 16000 images, testing in both cases over
# images from 16000 to 18000. Note that images are ordered chronologically.
#
# For coparison purposes, in Kumar, Ma, and Liang (2020)[2], they report
# accuraccies of 75.3±1.6 when
# training with "source" images (2000 first ones), 76.9±2.1 when training with
# "target" images (14000 next ones), 78.9±3.0 when training with both and
# 83.8±0.8 when applying their method "Gradual Self-Training.l"
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


X, Y = load_yearbook_features_resnet18()

X_train = X[:2000,:]
Y_train = Y[:2000]
X_test = X[16000:18000,:]
Y_test = Y[16000:18000]

clf = MRC(phi='linear').fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
error = np.average(Y_pred!=Y_test)
print('Yearbook prediction accuracy (2000 training instances): ' +
    str(1 - error))

X_train = X[:16000,:]
Y_train = Y[:16000]
X_test = X[16000:18000,:]
Y_test = Y[16000:18000]

clf = MRC(phi='linear').fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
error = np.average(Y_pred!=Y_test)
print('Yearbook prediction accuracy (16000 training instances): ' +
    str(1 - error))

