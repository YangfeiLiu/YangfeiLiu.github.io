---
layout:     post
title:      "Logistic Regression的理解与推导 "
subtitle:   "LR"
date:       2020-06-11
author:     "柳阳飞"
tags:
    - logistic regression
---

##### 线性模型

线性模型的一般向量形式为
$$
f(x)=w^Tx+b\tag{1} \label{eq1}
$$
线性回归的学习目标是使得模型的输出接近真实$f(x_i)\simeq y_i$

损失函数为均方误差
$$
l(w,b)=\sum_{i=1}^m(\|f(x_i)-y_i\|^2)\tag{2}
$$
通过最小二乘法求解。

##### 广义线性模型

$$
f(x)=g^{-1}(w^Tx+b) \tag{3}
$$

##### Logistic Regression

当$g(x) = \frac{1}{1+e^{-x}}$ 时，对应的线性模型形式为
$$
f(x)=\frac{1}{1+e^{-(w^T+b)}}\tag{4}\label{eq4}
$$
将$\eqref{eq4}$看作正类的概率，我们可以通过极大似然估计来求解参数$w,b$ 。

LR服从伯努利分布，因此
$$
p(y=1|x)=f(x)\\
p(y=0|x)=1-f(x)\tag{5}\label{eq5}
$$
合并为一个式子为
$$
f_{\theta}(x_i)=f(x_i)^{y_i}+(1-f(x_i))^{(1-y_i)}\tag{6}
$$
因此极大似然函数为
$$
l(w,b)=\prod_{i=1}^mf_\theta(x_i)\tag{7}
$$
转化为对数似然函数
$$
l(w,b)=\sum_{i=1}^my_i\ln f(x_i)+(1-y_i)\ln(1-f(x_i))\tag{8}\label{eq8}
$$
把$\eqref{eq5}$ 代入$\eqref{eq8}$ 中得到损失函数
$$
loss(w,b)=-l(w,b)=-\sum_{i=1}^m(-y_i\beta^Tx_i+\ln (1+e^{\beta^Tx_i}))\tag{9}\label{eq9}
$$
$\eqref{eq9}$ 式中$\beta=(w;b)$

为了防止过拟合，还会加上正则项
$$
loss(w,b)=-\frac{1}{m}\sum_{i=1}^m(-y_i\beta^Tx_i+\ln (1+e^{\beta^Tx_i}))+\frac{\gamma}{2m}\sum_{i=1}^m\beta^2\tag{10}\label{eq10}
$$
使用L2正则项的模型称为岭(Ridge)回归，使用L1正则项称为Lasso回归。采用梯度下降法求解

##### 优缺点

- 优点
  - 数据线性可分时表现好
  - 用途广法，不需要太多计算资源，可解释强，输入不需要归一化，输出可以指示概率
  - 当删除掉无用的特征或者相似特征时，表现更好
- 缺点
  - 无法处理非线性问题
  - 数据有缺失或者相关性大时效果不好