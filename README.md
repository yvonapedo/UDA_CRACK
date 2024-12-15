# [Unsupervised Domain Adaptation for Road Crack Segmentation](https://)

Authors: *Yvon Apedo*, *Huanjie Tao*

---
## Abstract

Surface structural defects, such as cracks, are a common challenge in infrastructure maintenance and operation. Intelligent inspection of these defects using computer vision and deep learning is crucial for early detection and timely intervention. While supervised learning methods have achieved significant success in detecting surface cracks, their performance is often dependent on large, annotated datasets. However, labeling these datasets can be labor-intensive and expensive. Additionally, supervised models often face difficulties in generalizing to unseen datasets due to domain disparities between the source and target images. To address these challenges, we propose an unsupervised domain adaptation framework to mitigate the domain shift between labeled source domains and unlabeled target domains. Our approach incorporates a style transfer module to achieve pixel-level distribution alignment between training and testing images. Specifically, we utilize Fourier transform to integrate amplitude information from the target domain into the source domain in the frequency space. At the feature level, we employ adversarial learning to align the feature representations of the source and target domains. Furthermore, we introduce an edge refinement module to correct any distorted boundaries, ensuring improved segmentation accuracy. The effectiveness of our method is validated on five datasets, including Cracktree200, CRACK500, CFD, CrackFCN, and MVTec AD. Experimental results demonstrate that our approach outperforms existing methods for crack segmentation, establishing its superiority in addressing domain adaptation challenges.

---

## Usage
### Datasets
Download the Cracktree200, CRACK500, CrackForest, CrackFCN, MVTec AD datasets and the file follows the following structure.

```
|-- datasets
    |-- cracktree200
        |-- train
        |   |-- train.txt
        |   |--img
        |   |   |--<crack1.jpg>
        |   |--gt
        |   |   |--<crack1.bmp>
        |-- valid
        |   |-- Valid_image
        |   |-- Lable_image
        |   |-- Valid_result
        ......
```

train.txt format
```
./dataset/crack315/img/crack1.jpg ./dataset/crack315/gt/crack1.bmp
./dataset/crack315/img/crack2.jpg ./dataset/crack315/gt/crack2.bmp
.....
```
### Train

```
python train.py
```
### Valid

```
python test.py
```

---
## Baseline Model Implementation

Please refer to [src](https://github.com/Jun-Pu/UCOS-DA/tree/main/src) for the code of our baseline model.

The pre-trained model can be downloaded at [Google Drive](https://drive.google.com/file/d/1KubZTnGlNEUOyuZjrMpvii1uZnq_vW19/view?usp=sharing).

Our code heavily references the code in [UCOS](https://arxiv.org/abs/2308.04528).

---

