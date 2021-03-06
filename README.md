# HIT-Machine-Learning-2021
哈工大2021秋机器学习课程实验

### Lab1：多项式拟合曲线

**目标**：

掌握最小二乘法求解（无惩罚项的损失函数）、掌握加惩罚项（2范数）的损失函数优化、梯度下降法、共轭梯度法、理解过拟合、克服过拟合的方法(如加惩罚项、增加样本) 

**要求**：

1.  生成数据，加入噪声；

2. 用高阶多项式函数拟合曲线；

3. 用解析解求解两种`loss`的最优解（无正则项和有正则项）

4. 优化方法求解最优解（梯度下降，共轭梯度）；

5. 用你得到的实验数据，解释过拟合。

6. 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果。

7. 语言不限，可以用`matlab`，`python`。求解解析解时可以利用现成的矩阵求逆。梯度下降，共轭梯度要求自己求梯度，迭代优化自己写。不许用现成的平台，例如`pytorch`，`tensorflow`的自动微分工具。

### Lab2：逻辑回归

**目标**：

理解逻辑回归模型，掌握逻辑回归模型的参数估计算法。

要求：实现两种损失函数的参数估计（1，无惩罚项；2.加入对参数的惩罚），可以采用梯度下降、共轭梯度或者牛顿法等。

**要求**：

1. 可以手工生成两个分别类别数据（可以用高斯分布），验证你的算法。考察类条件分布不满足朴素贝叶斯假设，会得到什么样的结果。

2. 逻辑回归有广泛的用处，例如广告预测。可以到UCI网站上，找一实际数据加以测试。

### Lab3：实现k-means聚类方法和混合高斯模型

**目标**：

实现一个`k-means`算法和混合高斯模型，并且用EM算法估计模型中的参数。

**测试**：

用高斯分布产生k个高斯分布的数据（不同均值和方差）（其中参数自己设定）。

（1）用`k-means`聚类，测试效果；

（2）用混合高斯模型和你实现的EM算法估计参数，看看每次迭代后似然值变化情况，考察EM算法是否可以获得正确的结果（与你设定的结果比较）。

**应用**：可以`UCI`上找一个简单问题数据，用你实现的`GMM`进行聚类。

### Lab4：PCA模型实验

**目标**：

实现一个`PCA`模型，能够对给定数据进行降维（即找到其中的主成分）。

**测试**：

（1）首先人工生成一些数据（如三维数据），让它们主要分布在低维空间中，如首先让某个维度的方差远小于其它唯独，然后对这些数据旋转。生成这些数据后，用你的`PCA`方法进行主成分提取。

（2）找一个人脸数据（小点样本量），用你实现`PCA`方法对该数据降维，找出一些主成分，然后用这些主成分对每一副人脸图像进行重建，比较一些它们与原图像有多大差别（用信噪比衡量）。



**注**：

Lab3实验报告网址：https://q2edi86dln.feishu.cn/docs/doccniDe9ARKM60b6LQ2CBvza0f

Lab4实验报告网址：https://q2edi86dln.feishu.cn/docs/doccnvFgwDmUqf8SaOfq6AK6C3c
