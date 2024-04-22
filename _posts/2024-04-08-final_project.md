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

### High-Level Strategy
Our approach combines a Convolutional Neural Network (CNN) using the CLIP architecture for image processing with a Multi-Layer Perceptron (MLP) that integrates image-derived features with geospatial data. This multimodal approach leverages the strengths of both data types to improve the accuracy of trait predictions.

### Data Handling
We utilized the provided dataset of 50,000 labeled images and corresponding geodata. Significant preprocessing included log transformations to address non-normal distribution of several features, enhancing model training efficiency.

### Challenges and Solutions
The integration of heterogeneous data types (images and geodata) presents alignment and scaling challenges. We addressed these by standardizing feature scales and employing batch normalization to stabilize the learning process.

### Data Volume and Quality Issues
We processed and used the entirety of the available dataset. Issues with data quality, particularly in geospatial measurements, were mitigated through data augmentation techniques and robust outlier detection.

## 4. Results and Evaluation

Our solution achieved the following milestones in the competition leaderboard:
- 5th place out of 182 teams as of 22 Apr.

### Comparison with Other Approaches
Our use of CLIP for image processing, coupled with a robust handling of geospatial data, sets our approach apart from more traditional methods that might rely solely on image data. This integration allows for superior handling of the multimodal nature of the dataset.

### Real-World Applicability
While our results are promising, further validation and scaling are required for deployment in real-world agricultural settings. Additional testing across diverse environments is necessary to ensure the model's generalizability and reliability.

## 5. Tools and Resources

- **Data Sources:** Directly from the FGVC11 competition dataset.
- **Key Tools:** PyTorch for model building, Python for data preprocessing, and various libraries such as XGBoost for postprocessing.
- **Inspirational Sources:** Existing literature on multimodal learning and previous entries in similar Kaggle competitions provided foundational ideas for our approach.

## 6. References
- [1]. Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
- [2]. Ke, Guolin, et al. "Lightgbm: A highly efficient gradient boosting decision tree." Advances in neural information processing systems 30 (2017).

## 7. Future Work
(Outline potential extensions or refinements of your project that could be explored to enhance performance or applicability.)

