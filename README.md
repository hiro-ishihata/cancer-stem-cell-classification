# cancer-stem-cell-classification
## Overview
Perform the classification of Cancer Stem Cell (CSC) culture 1-day and 2-day as described in the book.

## Requirement

* python 3.9.6
* pytorch 1.10.0+cuda11.3.1


## Installation

* pytorch (https://pytorch.org/get-started/locally/)
* pytorch-gradcam (https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjXg_TKwdz5AhUGPXAKHaiEAL8QFnoECDQQAQ&url=https%3A%2F%2Fpypi.org%2Fproject%2Fpytorch-gradcam%2F&usg=AOvVaw3JJwW8Mb_wMYC1uwkvDAX7)

## Sample data

Download from (https://drive.google.com/file/d/1EXsWd8inxnQWs-6wxmOIVfrWRE75qVGS/view?usp=sharing})
(413.5MB)

* sampledata/day01/*.png  Image #1 to #3357
* sampledata/day02/*.png  Image #3358 to #6620

## Usage

* python train.py --model MODEL [--pretrained]
* python gcam.py --imgs n1[,n2,....] --model MODEL [--pretrained]

## Note

## Reference
* Aida, S., Okugawa, J., Fujisaka, S., Kasai, T., Kameda, H., Sugiyama, T.: Deep learning of cancer stem cell morphology using conditional generative adversarial networks. Biomolecules 10(6), 931 (2020)

* Hanai, Y., Ishihata, H., Zhang, Z., Maruyama, R., Kasai, T., Kameda, H., Sugiyama, T.: Temporal and locational values of images affecting the deep learning of cancer stem cell morphology. Biomedicines 10(5), 941 (2022)

## Author

* Hiro Ishihata
* Tokyo University of Technology
* ishihata@stf.teu.ac.jp

## Licence

[MIT license](http://choosealicense.com/licenses/mit/)