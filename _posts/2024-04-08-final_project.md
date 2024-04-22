# PlantTraits2024: Advanced Multimodal Approach to Predict Plant Traits - FGVC11 Competition

## 1. Introduction

The aim of this report is to outline our methodology and findings in the ongoing PlantTraits2024 challenge, part of the FGVC11 workshop at CVPR 2024. This competition involves utilizing a dataset of 50,000 labeled images alongside ancillary geodata to predict various plant traits like height, seed mass, and leaf size. The task requires the development of a robust multimodal neural network capable of integrating diverse data types for effective trait prediction. This document details our project's objectives, the challenges faced, and the innovations applied to achieve competitive results in the competition.

## 2. Problem Domain and Project Description

### Goal and Significance
Our project's goal is to predict specific plant traits based on a combination of high-resolution images and geospatial data. The inputs to our system are images and geodata, and the outputs are predicted values of traits such as plant height and leaf size. This predictive capability is crucial for agricultural researchers and farmers who seek to understand plant characteristics in varying geographical conditions without the need for extensive field measurements.

### Data Input and Output Examples
An example input consists of an image of a plant along with its corresponding geodata such as location, altitude, and climate conditions. The output from our system is a set of traits for each plant, quantified, such as height in centimeters and leaf area in square centimeters.

## 3. Detailed Description of Our Approach

![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/21a71229-9772-4198-8d03-9f8712457f9c)

### Two-Stage Predictive Framework

Our method employs a two-stage predictive framework, integrating ConvNext and LightGBM to predict plant traits from images and auxiliary data.

#### Stage 1: Initial Prediction with ConvNext

The process begins with ConvNext [1], which is focused on generating an initial set of predictions for the plant traits using only image data. 

#### Stage 2: Final Prediction with LightGBM

In the second stage, we refine the initial predictions using the LightGBM [2] regressor. This machine learning model takes in the feature vectors from the ConvNext output, along with additional auxiliary data (climate, soil, satellite data), and the initial predictions. The goal here is to use the gradient boosting method of LightGBM to fine-tune the trait predictions.

### Training Process: 5-Fold Cross-Validation

We employ a 5-fold cross-validation strategy for training. In each fold:

1. The neural network is trained on a specific subset of the data to produce image features and initial trait predictions.
2. We hold back the network's predictions on the validation set as 'unseen data' to simulate the performance on new images.
3. The LightGBM model is then trained on these predictions to bridge the gap between the seen training images and unseen validation images, aiming to prevent overfitting and ensure better model generalization.

This method guarantees that the LightGBM model is not influenced by the exact images used to train the ConvNext network, thus addressing the potential issue of feature representation variability between training and unseen images.

The combination of ConvNext and LightGBM leverages the respective advantages of deep learning for feature extraction and ensemble learning for prediction refinement, leading to improved accuracy in our plant trait predictions.

## 4. Results and Evaluation

Our solution achieved the following milestones in the competition leaderboard:
- 5th place out of 182 teams as of 22 Apr.

### Comparison with Other Approaches
Our use of CLIP for image processing, coupled with a robust handling of geospatial data, sets our approach apart from more traditional methods that might rely solely on image data. This integration allows for superior handling of the multimodal nature of the dataset.

### Real-World Applicability
While our results are promising, further validation and scaling are required for deployment in real-world agricultural settings. Additional testing across diverse environments is necessary to ensure the model's generalizability and reliability.

## 5. Tools and Resources

- **Data Sources:** Directly from the FGVC11 competition dataset.
- **Key Tools:** PyTorch and LightGBM for model building
- **Inspirational Sources:** Existing literature on multimodal learning and previous entries in similar Kaggle competitions provided foundational ideas for our approach.

## 6. References
- [1]. Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
- [2]. Ke, Guolin, et al. "Lightgbm: A highly efficient gradient boosting decision tree." Advances in neural information processing systems 30 (2017).

## 7. Future Work
(Outline potential extensions or refinements of your project that could be explored to enhance performance or applicability.)

