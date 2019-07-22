---
layout:     post
title:      "Effetive use of dilated convolutions for segmentation small object instances in remote sensing imagery"
subtitle:   "用dilated convolutions 来做遥感影像中的小目标的分割"
date:       2019-06-10
author:     "柳阳飞"
tags:
    - remote sensing imagery
    - small object instances
    - dilated convolutions
---
###  动机： ###
解决遥感影像中目标小和拥挤的问题。

***

### 方法： ###
提出local feature extraction module(LFE)。dilation factors 扩大的卷积不能联合局部特征，LFE采用dilation factors缩小的卷积。

***

### 网络： ###
整个网络分为三个模块。
![net](/img/20190610/model.png)
#### 三个模块： ####

**The front-end module:**  为了满足大感受野，同时保持分辨率，采用了dilated convolution 
* extract features that cover large context, the dilation factors are gradually increased。
* 产生两个问题：  
	![problem](/img/20190610/problem.png)
	1. spatial consistency between neighboring units becomes weak  
	 如上图左边所示，所有蓝色单元构成一个信息金字塔，金字塔顶层的特征受到底层的影响。可以看到两个相邻的单元没有相互重叠的信息，因为dilation factor=2。随着dilation factor 的增大，没有信息重叠的单元会增多，输出将会逐渐与输入偏离。这种空间不连续性会导致预测结果产生锯齿现象。
	2. local structure cannot be extracted in higher layer  
	 顶层单元只能接收以上两个单元中的一个，不能同时获得所有信息，这意味着所有的顶层单元都不知道这两个单元的局部结构。这种情况下，如果目标足够大，可以从目标内部特征识别局部结构，上述的结构不会产生问题；如果目标比较小，其中的一些特征需要被高层提取和识别，此时由于没有重叠信息就很难提取特征。
	

**The LFE module:**  为了处理上述两个问题，提出了LFE结构。attach structure with decreasing dilation factor after increasing one, information pyramids of neighboring units can be connected again。递减的结构逐渐恢复相邻单元的连续性，并且可以从高层提取局部特征。

**The head module:**  通过对生成的probability map设置阈值来获得mask

#### 网络细节 ####
![ditail](/img/20190610/ditails.png)

### 设计实验 ###

整个实验考虑到三个方面：  
1. 对于不同的感受野，分为Small FOV和Large FOV。针对不同的FOV设计的网络是基于Pool的类似于VGG16的结构，对应的网络分别是**Front-S**和**Front-L**。 *only layers bellow third pooling layer of VGG-16 are used for small FOV (Front-S) and layers bellow fourth pooling layer are used for large FOV (Front-L)。*  
2. 验证dilation convolution 的作用，对应的网络分别为**Front-S+D**和**Front-L+D。** 移除了网络中的所有的pool层，把convolution层变成dilation convolution层。  
3. 验证LFE模块的作用，把LFE模块连接在网络后面，对应的网络为**Front-S+D+LFE**和**Front-L+D+LFE。** 同时为了保证参数量的公正，设计了网络**Front-S+D+Large**和**Front-L+D+Large。** 它们有相同的参数量，唯一的不同就是它们没有递减的dilation factors。  
4. 所有的卷积后都有Relu层激活，除了最后一层是softmax。所有的网络采用 76x76 patches作为输入，输出是概率图中间的 16x16 的区域。

### 实验结果 ###
![result](/img/20190610/result.png)

![result](/img/20190610/result1.png)

![result](/img/20190610/result2.png)

Very small (0-100 pixels), Small (100-400 pixels), Mid (400-1,600 pixels), Large (1,600-6,400 pixels) and Very large (over 6,400 pixels)。  
训练数据集： Toyota City Dataset  
测试数据集：
Massachusetts Buildings Dataset [https://www.cs.toronto.edu/~vmnih/data/](https://www.cs.toronto.edu/~vmnih/data/ "Massachusetts ")  
Vaihingen Dataset

### 参考 ###
[1] Effective Use of Dilated Convolutions for Segmenting Small Object Instances in 
Remote Sensing Imagery [https://arxiv.org/abs/1709.00179](https://arxiv.org/abs/1709.00179)