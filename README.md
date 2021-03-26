# multi-label-image-classification
A Baseline for Multi-Label Image Classification Using Ensemble Deep CNN
## Code description
1. Code tested with PyTorch 0.4.

2. Model2 (M2) and model3 (M3) appearing in the paper could be adapted from model1 code by uncommenting corresponding lines for randomcropping and mixup.

3. To run a script using: python resnet101_model1fc.py 1 512 16 (three arguments are trial index, patch size, batch size)

4. The evaluation metrics for VOC2007 are slightly different from those for NUS-WIDE and MS-COCO since there are "difficult examples" in the annotations which are ignored when evaluating.

5. We use all training data to train the model and a fixed criterion for training stop.

## Data
To run the code you might need to download images for three datasets from their official websites.

## Reference
Qian Wang, Ning Jia, Toby P. Breckon, A Baseline for Multi-Label Image Classification Using Ensemble Deep CNN, IEEE International Conference on Image Processing 2019, Taipei.
## Contact
qian.wang173@hotmail.com
