# PlantTraits2024: Advanced Multimodal Approach to Predict Plant Traits - FGVC11 Competition

## 1. Introduction

The aim of this report is to outline our methodology and findings in the ongoing PlantTraits2024 challenge, part of the FGVC11 workshop at CVPR 2024. This competition involves utilizing a dataset of 50,000 labeled images alongside ancillary geodata to predict various plant traits like height, seed mass, and leaf size. The task requires the development of a robust multimodal neural network capable of integrating diverse data types for effective trait prediction. This document details our project's objectives, the challenges faced, and the innovations applied to achieve competitive results in the competition.

## 2. Problem Domain and Challenges

### Goal and Significance
Our project's goal is to predict specific plant traits based on a combination of high-resolution images and geospatial data. The inputs to our system are images and geodata, and the outputs are predicted values of traits such as plant height and leaf size. This predictive capability is crucial for agricultural researchers and farmers who seek to understand plant characteristics in varying geographical conditions without the need for extensive field measurements.

### Data Input and Output Examples
An example input consists of an image of a plant along with its corresponding geodata such as location, altitude, and climate conditions. The output from our system is a set of traits for each plant, quantified, such as height and leaf area.

### Challenge 1: Data Distribution

The table below presents statistical summaries for various plant trait measurements. Notably, the scale of the maximum values is significantly larger than the means, highlighting the wide range of the data. Also, the significantly high std values imply the challenge of predicting these traits accurately, as the models must be robust enough to handle such extremes without compromising performance on more typical values.

| Target     | Minimum | 1st Percentile | 5th Percentile | 25th Percentile | Median | 75th Percentile | 95th Percentile | 99th Percentile | Maximum    | Standard Deviation |
|------------|---------|----------------|----------------|-----------------|--------|-----------------|-----------------|-----------------|------------|---------------------|
| X4    | -1.16   | 0.23           | 0.30           | 0.41            | 0.51   | 0.62            | 0.80            | 0.91            | 2.62       | 0.175               |
| X11   | 0.02    | 3.10           | 4.97           | 10.81           | 15.28  | 20.13           | 33.25           | 52.75           | 27167.64   | 1.290179e+04        |
| X18   | 0.00    | 0.04           | 0.10           | 0.31            | 0.73   | 4.30            | 18.65           | 30.09           | 705177.70  | 2.331357e+06        |
| X50  | 0.00    | 0.48           | 0.74           | 1.15            | 1.47   | 1.92            | 2.98            | 4.38            | 218.29     | 1.369172e+03        |
| X26  | 0.00    | 0.01           | 0.05           | 0.58            | 2.65   | 16.84           | 257.58          | 1805.68         | 8920578.70 | 2.110374e+05        |
| X3112 | 0.00    | 12.38          | 44.42          | 265.27          | 793.31 | 2296.86         | 8871.99         | 25810.43        | 3145145.18 | 9.238101e+07        |



### Challenge 2: Implicit Requirement of Classification Ability

The examples depicted highlight a significant implicit challenge: the model must be capable of discerning species because the same species exhibit identical plant traits. The images show that objects such as a pine seed and a branch, although physically smaller, are labeled with the same height trait as full-grown pine trees. This implies that the model needs to understand context and classify the species accurately to make valid trait predictions, underscoring the complexity of the task.

<div align="center">
    <img width="640" alt="image" src="https://github.com/ywugwu/ywugwu.github.io/assets/128890731/cbc963e8-a171-4184-97c3-bfdee493bff1">
    <br>
    <em>A pine seed has the same height as the other pine trees.</em>
</div>

<div align="center">
    <img width="640" alt="image" src="https://github.com/ywugwu/ywugwu.github.io/assets/128890731/08123f44-7217-48a7-9ab7-d57fb4527a88">
    <br>
    <em> A small branch of some tree has the same height as some tall trees. </em>
</div>


## 3. Detailed Description of Our Approach

![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/21a71229-9772-4198-8d03-9f8712457f9c)

### Two-Stage Predictive Framework

Our method employs a two-stage predictive framework, integrating ConvNext_v2-tiny (with 27M parameters) and LightGBM to predict plant traits from images and auxiliary data.

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

Our current R2 scores are shown below:

### Quantitative Result

| Model                        | R² Score |
|------------------------------|----------|
| ConvNext Only                | 0.35     |
| ConvNext + LightGBM          | 0.43     |
| Ensemble with Varied Params  | 0.45     |

We can see a significant improvement using different models/strategies, indicating further room for improvement by designing more powerful neural networks.

We also observed an interesting phenomenon in our training process:

![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/e6f5c515-f74e-4a43-8230-9b05d8883400)

As the training epochs increase, the training score keeps increasing. However, the validation score doesn't decrease as the classic view of "overfitting" suggests.

The convergence in the validation scores rather than a consistent decline suggests that while the model may be learning the training data better with each epoch, it's not necessarily overfitting in a manner that degrades its performance on unseen data.


### Leaderboard Ranking

Our solution achieved:
- 5th place out of 194 teams as of 24 Apr. (you may check the live ranking at [kaggle](https://www.kaggle.com/competitions/planttraits2024/leaderboard))


![image](https://github.com/ywugwu/ywugwu.github.io/assets/128890731/34e18e4a-8bfb-46d9-817a-886efc1a71e7)


## 5. Tools and Resources

- **Data Sources:** Directly from the FGVC11 competition dataset.
- **Key Tools:** PyTorch and LightGBM for model building
- **Inspirational Sources:** Existing literature on multimodal learning and previous entries in similar Kaggle competitions provided foundational ideas for our approach.


## 6. Future Work

### High-resolution Training and Multi-scale Testing

We observed that high-resolution images are quite helpful in this task. While we fail to gain improvement by improving training resolution from 384x384 to 512x512, we believe it's a promising direction and we would do more experiments to make it work.

In addition, using multi-scale resolutions in the testing stage is an effective method for leveraging high-resolution information so we will accomplish it in the future.

### More Powerful Vision Models and iNaturalist Pretrained-Models

We aim to utilize more advanced vision models for improved feature extraction. 

Additionally, we will consider models pretrained on the iNaturalist dataset to enhance species classification ability. These steps are expected to refine our model's performance further.


## 7. References
- [1]. Woo, Sanghyun, et al. "Convnext v2: Co-designing and scaling convnets with masked autoencoders." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
- [2]. Ke, Guolin, et al. "Lightgbm: A highly efficient gradient boosting decision tree." Advances in neural information processing systems 30 (2017).



