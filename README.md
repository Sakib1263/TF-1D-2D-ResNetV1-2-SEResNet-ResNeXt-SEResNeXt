# 1DResNet_KERAS
This repository contains an One-Dimentional (1D) version of original variants of ResNet developed in KERAS along with implementation guidance in Jupyter Notebooks  
Read more about ResNets in this original paper: https://arxiv.org/pdf/1512.03385.pdf  
The models in this repository have been built following the original paper's implementation as much as possible, though more efficient implementation could be possible due to the advancements in this field since then. The models implemented in this repository are:
1. ResNet18
2. ResNet32
3. ResNet50
4. ResNet101
5. ResNet152  

# ResNet Architectures
![ResNet Architecture Params](https://github.com/Sakib1263/1DResNet-KERAS/blob/main/Documents/ResNet.png "ResNet Parameters")  

Mentionable that in this case, the Kernels are 1D in each layers instead of the 2D implementation shown in this figure from the original paper (default implementation). So, nevertheless, the total number of parameters are also different than mentioned in the paper for all models.  
The speciality about this model is its flexibility. Apart from choosing any of 5 available ResNet models in 1D, one can easily change the parameters such as number of input kernels/filters, number of classes for Classification tasks and number of extracted features for Regression tasks, etc. Details of the process are available in the attached Jupyter Notebook in the codes section.
