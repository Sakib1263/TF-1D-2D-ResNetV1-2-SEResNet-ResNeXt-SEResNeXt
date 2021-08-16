# 1DResNet_KERAS
Models supported: ResNet18, ResNet32, ResNet50, ResNet101, ResNet 152 (1D and 2D version with DEMO).  
This repository contains an One-Dimentional (1D) and Two-Dimentional (2D) versions of original variants of ResNet developed in KERAS along with implementation guidance (DEMO) in Jupyter Notebook.  
Read more about ResNets in this original paper: https://arxiv.org/pdf/1512.03385.pdf. The models in this repository have been built following the original paper's implementation as much as possible, though more efficient implementation could be possible due to the advancements in this field since then. On the contrary, the models contain BatchNormalization (BN) blocks after Convolutional blocks and before activation, which is deviant from the original implementation.  
Read more about BN in this paper: https://arxiv.org/abs/1502.03167v3. The models implemented in this repository are:

# ResNet Architectures
![ResNet Architecture Params](https://github.com/Sakib1263/1DResNet-KERAS/blob/main/Documents/ResNet.png "ResNet Parameters")  

The speciality about this model is its flexibility. Apart from choosing any of 5 available ResNet models in 1D, one can easily change the parameters such as number of input kernels/filters, number of classes for Classification tasks and number of extracted features for Regression tasks, etc. Details of the process are available in the attached Jupyter Notebooks containing a DEMO in the codes section.
