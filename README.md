# ResNet_Model_Builder_KERAS
This repository contains One-Dimentional (1D) and Two-Dimentional (2D) versions of original variants of ResNet developed in KERAS along with implementation guidance (DEMO) in Jupyter Notebook. The models in this repository have been built following the original paper's implementation as much as possible, though more efficient implementation could be possible due to the advancements in this field since then. Read more about ResNets in this original paper: https://arxiv.org/pdf/1512.03385.pdf. 

![ResNet Architecture Params](https://github.com/Sakib1263/ResNet-Model-Builder-KERAS/blob/main/Documents/ResNet_Model.png "ResNet Architecture") 

On the contrary, the models contain BatchNormalization (BN) blocks after Convolutional blocks and before activation, which is deviant from the original implementation. Read more about BN in this paper: https://arxiv.org/abs/1502.03167v3.

# ResNet Architectures
A table from the original paper containing the architectures of the ResNet models developed is shown below:  
![ResNet Architecture Params](https://github.com/Sakib1263/1DResNet-KERAS/blob/main/Documents/ResNet.png "ResNet Parameters")  
The original implementation comes up with a fixed number of 1000 classification due to using ImageNet dataset for training and evaluation purposes. The developed ResNet model is flexible enough to accept any number of classed according to the user's requirements.

The speciality about this model is its flexibility. Apart from choosing the ResNet models for 1D or 2D tasks, the user has the option for: 
1. Choosing any of 4 available VGG models for either 1D or 2D tasks.
2. Number of input kernel/filter, commonly known as Width of the model.
3. Number of classes for Classification tasks and number of extracted features for Regression tasks.
4. Number of Channels in the Input Dataset.
Details of the process are available in the DEMO provided in the codes section. The datasets used in the DEMO as also available in the 'Documents' folder.
