# Exploring the Robustness of ResNet-50 for Image Classification

## 1. Introduction
This project aims to briefly investigate how image classification accuracy changes as images undergo various forms of corruption. 

The study uses a pre-trained ResNet-50 model [1], experimenting with different types of image alterations. 

The goal is to understand the model's robustness and limitations in classifying progressively corrupted images.

Our code to reproduce the results is available at [https://colab.research.google.com/drive/1W75c_YtBpNiKy2AndzwQ842Hh7fRzOHv?usp=sharing](https://colab.research.google.com/drive/1W75c_YtBpNiKy2AndzwQ842Hh7fRzOHv?usp=sharing)

## 2. Methodology
### a. Choice of Image Corruption
The project explores five types of image corruption: adding noise, blurring, zooming, adding scratches, and washing out colors:

![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/c93e568a-818e-4f66-999b-d87ab2e928d6)
![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/f217685c-e63a-40f2-81d9-d4ca892f2d01)
![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/bbaeda3c-8c5a-474f-ae1e-df269e6f657e)
![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/0dbe0545-bf4f-4157-b83e-6c83721216b8)
![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/fc5fd84d-71a2-4900-ab7d-9f382f34bb8d)

### b. Classification Accuracy Evaluation
The ResNet-50 model is pretrained in the ImageNet dataset, and fine-tuned on CIFAR-10 [2], was used to classify images at each corruption level. The accuracy was then plotted against the magnitude of corruption.

## 3. Results


- **Noise (Red):** There is a gradual decline in the model's accuracy with increasing noise levels, suggesting that noise negatively impacts the classifier's performance.

- **Scratches (Purple):** Similar to noise, the addition of scratches leads to a gradual decrease in accuracy, indicating that scratches are detrimental to image recognition.

- **Blur (Blue):** Blurring the images results in a progressive drop in accuracy, reflecting the model's diminishing ability to correctly classify increasingly blurred images.

- **Zoom (Green):** The accuracy improves when the images are zoomed in at 1.2x and 1.6x magnifications, but there is a sharp decrease at 1.8x and 2.0x. This indicates a non-linear response where the model benefits from a moderate zoom but is significantly impaired by higher zoom levels.

- **Washout (Orange):** The model's accuracy remains consistent across all levels of color washout, suggesting that this type of corruption does not affect the classification capability of the model.


<div align="center">
    <img src="https://github.com/ywugwu/ywugwu.github.io/assets/128890731/6db3b6af-5aae-4593-aaae-e62e1e372554" width="66%">
</div>

## References

- [1] He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. arXiv preprint arXiv:1512.03385 [cs.CV].
- [2] Krizhevsky, A., Hinton, G., & others. (2009). *Learning Multiple Layers of Features from Tiny Images*. Toronto, ON, Canada.

