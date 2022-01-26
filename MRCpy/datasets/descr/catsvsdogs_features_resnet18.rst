1. Title: Cats vs Dogs Features ResNet18 Dataset

2. Sources:
   https://www.tensorflow.org/datasets/catalog/cats_vs_dogs

3. Past Usage:
   Elson, J., Douceur, J. R., Howell, J., & Saul, J. (2007).
   Asirra: a CAPTCHA that exploits interest-aligned manual
   image categorization. CCS, 7, 366-374.

4. Relevant Information:
   This dataset is derived from the Cats vs Dogs Dataset consisting in
   RGB images
   (https://www.microsoft.com/en-us/download/details.aspx?id=54765).
   It contains the features of these images extracted using a pretrained
   ResNet18 network.
   Cats vs Dogs is a large set of images of cats and dogs. Cats are labeled 0
   and dogs are labeled 1.

5. Number of Instances: 23262
   Class 0 (Cats) 11658
   Class 1 (Dogs) 11604

6. Number of Attributes: 512 + class atribute

7. Attribute Information:
   The 512 attributes correspond to the second last layer of a ResNet18 model
   pretrained on ImageNet Data Set used to predict the class of each image
   on Cats vs Dogs dataset.

8. Missing Attribute Values: None
