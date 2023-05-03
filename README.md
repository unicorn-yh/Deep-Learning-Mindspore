# Deep-Learning-Mindspore
 基于华为云平台的 MindSpore1.3 框架的回归任务和图像分类任务（汽车里程数预测 / 花卉图像分类）

 Regression tasks and image classification tasks based on the MindSpore1.3 framework of Huawei Cloud Platform (Car mileage prediction / Flower image classification)

 华为云平台（Huawei Cloud Platform）: https://www.huaweicloud.com/

<br>

# 1. 汽车里程数预测 Car Mileage Prediction

![image-20230402175403489](README/image-20230402175403489.png)

<h4 align="center">Fig. 1: Train dataset display</h4>

<br>

## Aim

The main content of this project is to predict the fuel consumption mileage of the car. The frameworks used mainly include: MindSpore1.3, which is mainly used for the construction of deep learning algorithms, based on the open source auto-mpg data set, based on MindSpore1.3 deep learning The library applies a fully-connected neural network for vehicle mileage prediction. The main focus of this experiment is to distinguish the difference in network structure between classification tasks and regression tasks.

<br>

## Experimental design

1. **Import the modules required for the experiment:** This step is usually the first step in program editing, and import the module packages required by the experimental code with the import command. 

2. **Import the data set and preprocess:** the training of the neural network is inseparable from the data, and the data is imported here. One-hot encoding of character features in the dataset. At the same time, look at the relationship between data features. 

3. **Model building and training:** use the cell module of mindspore.nn to build a fully connected network, including an input layer, a hidden layer, and an output layer. At the same time, configure the optimizer, loss function and evaluation indicators required by the network. Pass in the data and start training the model. This experiment is a regression task, so the output of the output layer is 1-dimensional. 

4. **Check the model training status:** Use the two evaluation indicators of MAE and MSE to check the status of the model during training.

   

<br>

## Dataset overview

<p> <img src="README/image-20230402165544961.png"> </p>

<h4 align="center">Fig. 2: Train dataset after discrete feature processing </h4>

<br>

<br>

![image-20230402165936825](README/image-20230402165936825.png)

<h4 align="center">Fig. 3: Visualize the correlation of data items in train dataset</h4>

<br>

## Result

| Model Training | Result                                                       |
| :--------------: | :------------------------------------------------------------: |
| Iteration      | ![image-20230402170212823](README/image-20230402170212823.png) |
| Loss           | ![image-20230402170225256](README/image-20230402170225256.png) |

<h4 align="center">Table 1: Model training result</h4>



<br>

| Loss Function           | Graph                                                        |
| :-----------------------: | :------------------------------------------------------------: |
| MAE loss function | ![image-20230402170412342](README/image-20230402170412342.png) |
| MSE loss function | ![image-20230402170423622](README/image-20230402170423622.png) |

<h4 align="center">Table 2: Loss function graph</h4>

<br>

It can be seen from Table 2, the model's MAE and MSE began to converge when the number of iterations reached 400, and showed a gentle trend until the follow-up. 

This project emphasized the usage of the MindSpore1.3 framework of Huawei Cloud Platform and its deep learning library to realize the regression prediction of car mileage based on the fully connected neural network. The experiment uses MAE and MSE loss functions to evaluate the performance of the regression task of the model, and from the above experimental results and analysis, it can be seen that after the model reaches a certain number of iterations, the loss value gradually converges to a fixed value, realizing the regression task.



<br>



# 2. 花卉图像分类 Flower Photos Classification

| Flower Photos Classification: Tulips 郁金香 | ![image-20230402174855978](README/image-20230402174855978.png) |
| :-----------------------------------------: | ------------------------------------------------------------ |

<h4 align="center">Fig. 4: Tulips</h4>

<br>

## Aim

With the rapid development of electronic technology, it is more and more convenient for people to use portable digital devices (such as mobile phones, cameras, etc.) to obtain flower images, and how to automatically identify flower species has received extensive attention. Due to the complexity of the background of flowers, as well as the inter-category similarity and intra-category diversity of flowers, the traditional method of manually extracting features for image classification cannot solve the problem of flower image classification very well.

This experiment is a flower recognition experiment based on ordinary convolutional neural network and ResNet staggered network. Unlike traditional image classification methods, convolutional neural network does not need to manually extract features, and can automatically learn features containing rich semantic information according to the input image. A more comprehensive feature description of flower images can well express different categories of information of images.

<br>

## Experimental Design

1. Import the experimental environment; 

2. Data set acquisition and preprocessing; 

3. Construct CNN and ResNet image recognition models; 

4. Image classification model verification.

   <br>

![image-20230402172354325](README/image-20230402172354325.png)

<h4 align="center">Fig. 5: ResNet-50 architecture</h4>

<br>

Analyzing from left to right, the leftmost part of Figure 6 is the step diagram of ResNet-50, which includes the convolution block (conv_block) and standard block (identity_block) modules that deepen the network depth, avoiding training calculation difficulties and network degradation. question. MindSpore has launched to support this model, so we can directly call the interface of this model, and pass in the defined hyperparameters and data when using the model.

<br>

Model we used: ***ResNet-50***

Source: https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnet/src/resnet.py#

<br>

## Result 

| CNN                                                          | ResNet-50                                                    |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20230331003437730](README/image-20230331003437730.png) | ![image-20230331003151291](README/image-20230331003151291.png) |

<h4 align="center">Table 3: ResNet-50 architecture</h4>

<br>

From the training results in Table 3, it can be seen that the model trained by the ResNet-50 network achieved an accuracy rate of 0.98 in the flower classification task, while the model trained by a normal CNN network achieved an accuracy rate of 0.78. The epoch of both models is set to 600, and the final loss of the ResNet-50 network reaches 0.24, while the loss of the ordinary CNN network is 0.79.

It can be seen that the model performance of the ResNet-50 network is significantly better than that of ordinary CNN. This is because the staggered network overcomes the problem of "gradient disappearance" and continues to maintain low error rates and losses in deeper networks, making it possible to build thousands of Networks with up to 3 convolutional layers are possible, thus outperforming shallower networks. The deeper the level, the stronger the representation ability and the higher the classification accuracy. In addition, ResNet speeds up network performance by using Batch Normalization to adjust the input layer to solve the problem of covariate shift.

ResNet is divided into 18 layers, 34 layers, 50 layers, 101 layers, 110 layers, 152 layers, 164 layers, 1202 layers, etc., and we use the 50-layer ResNet-50. ResNet-50 uses a bottleneck design for the building blocks, the bottleneck residual block uses 1×1 convolution (called "bottleneck"), which reduces the number of parameters and matrix multiplications, and can train each layer more quickly, using Three layers stacked instead of two. ResNet uses the ReLU activation function because of its function as a regularizer, which reduces the information loss during forward propagation and backward propagation, and improves the classification performance of the model.

<br>

## Personal Summary

This project emphasized the usage of the MindSpore1.3 framework of Huawei Cloud Platform and its deep learning library to realize flower image classification based on ResNet-50 staggered network. From the above experimental results, it can be seen that the model trained by the ResNet network performs significantly better than ordinary CNN, and is better than ordinary CNN in terms of accuracy and loss. This stems from the fact that the former adjusts the input layer in batch normalization to solve the problem of covariate shift, and overcomes the problem of gradient disappearance, and keeps low error rate and loss in deeper networks, so its performance is better than shallow network.

<br>

# 3. 
