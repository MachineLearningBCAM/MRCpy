1. Title: Yearbook-Features ResNet18 Dataset

2. Sources:
   https://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html

3. Past Usage:


4. Relevant Information:
   This dataset is derived from the Yearbook Dataset consisting in grayscale
   images
   (https://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html).
   It contains the features of these images extracted using a pretrained
   ResNet18 network.
   This part of the dataset contains 37,921 vectors of features extracted from
   frontal-facing American high school yearbook portraits
   taken from 1905 to 2013.

5. Number of Instances: 37921
   Class 0 (F) 20248
   Class 1 (M) 17673

6. Number of Attributes: 512 + 4 info attributes (year, state, city, school)
                             + class atribute

7. Attribute Information:
   The 512 attributes correspond to the second last layer of a ResNet18 model
   pretrained on ImageNet Data Set used to predict the class of each image
   on Yearbook dataset.
   The 4 attributes giving additional information are available in the file
   yearbook_attributes.csv

8. Missing Attribute Values: None
