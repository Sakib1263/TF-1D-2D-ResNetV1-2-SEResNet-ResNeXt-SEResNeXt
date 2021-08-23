# ResNet-ResNeXt-Model-Builder-Tensorflow-Keras
This repository contains One-Dimentional (1D) and Two-Dimentional (2D) versions of ResNet (original) and ResNeXt (Aggregated Residual Transformations on ResNet) developed in Tensorflow-Keras. The models in this repository have been built following the original papers' implementation guidances (as much as possible).  
Read more about ResNets in this original preprint: https://arxiv.org/pdf/1512.03385.  
Read more about ResNeXts in this original preprint: https://arxiv.org/abs/1611.05431.  

Supported Models:  
1. ResNet18 - ResNeXt18
2. ResNet34 - ResNeXt34
3. ResNet50 - ResNeXt50
4. ResNet101 - ResNeXt101
5. ResNet152 - ResNeXt152

![ResNet Architecture Params](https://github.com/Sakib1263/ResNet-Model-Builder-KERAS/blob/main/Documents/Images/ResNet_Model.png "ResNet Architecture") 

All the models contain BatchNormalization (BN) blocks after Convolutional blocks and before activation (ReLU), which is deviant from the original implementation to obtain better performance. Read more about BN in this paper: https://arxiv.org/abs/1502.03167v3.

## ResNet Architectures
A table from the original paper containing the architectures of the ResNet models developed is shown below:  

![ResNet Architecture Params](https://github.com/Sakib1263/1DResNet-KERAS/blob/main/Documents/Images/ResNet.png "ResNet Parameters")  

The original implementation comes up with a fixed number of 1000 classification due to using ImageNet dataset for training and evaluation purposes. The developed ResNet model is flexible enough to accept any number of classed according to the user's requirements.  

Mentionable that ResNet18 and ResNet34 uses a lighter residual block that other three deeper models as shown in the Figure below where the deeper residual block with a bottleneck structure is for ResNet50, ResNet101 and ResNet152.  

![Residual Blocks](https://github.com/Sakib1263/1DResNet-KERAS/blob/main/Documents/Images/Residual_Block.png "Residual Blocks")  

## ResNeXt Architectures
The architecture of ResNeXt, also known as ResNet_v3, is almost same as that of the original ResNet, except the Residual Block as shown in the figure below. The aggregated residual block in ResNeXt divides the input tensor into multiple parallel paths based on the cardinality factor set by the user. Normally the more paths we have, the better is the performance and the lighter is the network. This image from the paper shows three equivalent structure for Aggregated Residual Blocks. In this code, only model (b) has been implemented (so far).  

![Aggregated Residual Block Models](https://github.com/Sakib1263/ResNet-ResNeXt-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/ResNeXt_Eq_Blocks.png "Aggregated Residual Blocks")  

The following Table represents a comparison between ResNet50 and ResNeXt50. It can be seen that with Cardinality = 32 (Default in the paper), the ResNeXt model has around 500000 less parameters than its equivalent ResNet counterpart.  

![Aggregated Residual Block Table](https://github.com/Sakib1263/ResNet-ResNeXt-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/ResNeXt_Table.png "ResNet50 vs. ResNeXt50")  

## Supported Features
The speciality about this model is its flexibility. The user has the option for: 
1. Choosing any of 5 available ResNet or ResNeXt models for either 1D or 2D tasks.
2. Varying number of input kernel/filter, commonly known as the Width of the model.
3. Varying number of classes for Classification tasks and number of extracted features for Regression tasks.
4. Varying number of Channels in the Input Dataset.
5. Varying Cardinality amount in the ResNext architecture (model default is 8, paper default is 32). Mentionable that, When Cardinality = 1, ResNeXt becomes ResNet.  

Details of the process are available in the DEMO provided in the codes section. The datasets used in the DEMO as also available in the 'Documents' folder.
