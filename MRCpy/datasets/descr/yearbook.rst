1. Title: Yearbook Dataset

2. Sources:
   https://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html

3. Past Usage:
   1. Ginosar, S., Rakelly, K., Sachs, S., Yin, B., & Efros, A. A. (2015).
      A century of portraits: A visual historical record of american high
      school yearbooks. In Proceedings of the IEEE International Conference
      on Computer Vision Workshops (pp. 1-7).
   2. Kumar, A., Ma, T., & Liang, P. (2020, November). Understanding
      self-training for gradual domain adaptation. In International Conference
      on Machine Learning (pp. 5468-5479). PMLR.

4. Relevant Information:
   This part of the dataset contains 37,921 frontal-facing
   American high school yearbook portraits taken from 1905 to 2013.

5. Number of Instances: 37921
   Class 0 (F) 20248
   Class 1 (M) 17673

6. Number of Attributes: Grayscale PNG images

7. Attribute Information:
   The 512 attributes correspond to the second last layer of a ResNet18 model
   pretrained on ImageNet Data Set used to predict the class of each image
   on Yearbook dataset.

8. Missing Attribute Values: None

This is a copy of Portraits Data Set. The original dataset can
be download from this adress:
https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0