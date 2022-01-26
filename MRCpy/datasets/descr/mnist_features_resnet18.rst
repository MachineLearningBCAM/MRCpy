1. Title: MNIST Features ResNet18 Dataset

2. Sources:
   https://www.tensorflow.org/datasets/catalog/mnist

3. Past Usage:

4. Relevant Information:
   This dataset is derived from the MNIST Dataset consisting in
   grayscale 28x28 images (http://yann.lecun.com/exdb/mnist/)
   of labeled handwritten digits.
   It contains the features of these images extracted using a pretrained
   ResNet18 network. First 60000 correspond to train set and 10000 last to
   test set.

5. Number of Instances: 70000
   Train 60000
   Test  10000

6. Number of Attributes: 512 + class atribute

7. Attribute Information:
   The 512 attributes correspond to the second last layer of a ResNet18 model
   pretrained on ImageNet Data Set used to predict the class of each image
   on MNIST dataset.

8. Missing Attribute Values: None
