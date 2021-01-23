# 常见的深度学习概念

## 1. 正则化方法
> 参考
> [[Deep Learning] 正则化](https://www.cnblogs.com/maybe2030/p/9231231.html)
> [机器学习中正则化项L1和L2的直观理解](https://blog.csdn.net/jinping_shi/article/details/52433975)
### 1.1 Lp范数
范数简单可以理解为用来表征向量空间中的距离，对范数的一个一般化表示如下：
$$
||x||_p = (\sum_i^nx_i^p)^{\frac{1}{p}}
$$

即所有向量的p次方的和的p次根，其中$p$的取值范围为$[1, +\infty )$，不能小于1，否则违反了空间中距离表示的三角不等式法则

其中特殊的是$L_0$范数，表示的是向量中非零元素的个数：
$$
||x||_0 = \# (i|x_i \ne 0)
$$

可以通过最小化L0范数，来寻找最少最优的稀疏特征项。但不幸的是，L0范数的最优化问题是一个NP hard问题（L0范数同样是非凸的）。因此，在实际应用中我们经常对L0进行凸松弛，理论上有证明，L1范数是L0范数的最优凸近似，因此通常使用L1范数来代替直接优化L0范数。

### 1.2 L1范数与L1正则化
L1范数表示如下：
$$
||x||_1 = \sum_i^n|x_i|
$$

L1范数就是向量各元素的绝对值之和，也被称为是"稀疏规则算子"（Lasso regularization）。

对应的就是**L1正则化 (l1 weight decay)**方法，其损失函数定义如下：
$$
J = J_0+\lambda\sum_w|w|
$$

其中$J_0$是原始的损失函数，而加号后面的是L1正则化项（惩罚项），$\lambda$是正则化系数。此时我们的任务变成在$L$约束下求出$J_0$最小值的解。

**L1正则化作用**
1. 进行特征选择：如果只有少数特征对这个模型有贡献，就可以只关注系数是非零值的特征。

### 1.3 L2范数和L2正则化
L2范数就是欧几里得距离，公式如下：
$$
||x||_2 = \sqrt{\sum_i^n(x_i)^2}
$$

与上面相似，将L2范式带入Loss函数作为惩罚项，即可得到L2正则化
**L2正则化作用**
1. 以L2范数作为正则项可以得到稠密解，即每个特征对应的参数𝑤都很小，接近于0但是不为0
2. L2范数作为正则化项，可以防止模型为了迎合训练集而过于复杂造成过拟合的情况，提高模型的泛化能力

### 1.4 L1范数和L2范数的区别
从图中可以直观感受L1范数和L2范数的区别
<p align="center"><img src="/docs/docs/DeepLearning/Pics/L1andL2.jpg" alt="L1和L2正则化" style="zoom:40%;" />

同时，从贝叶斯先验的角度看，当训练一个模型时，仅依靠当前的训练数据集是不够的，为了实现更好的泛化能力，往往需要加入先验项，而加入正则项相当于加入了一种先验。

* L1范数相当于加入了一个Laplacean先验；
* L2范数相当于加入了一个Gaussian先验。

## 2. Dropout的作用和原理
**L1和L2正则化修改代价函数，Dropout修改神经网络本身**
dropout直接让一些神经元随机停止工作，随机挑选几个神经元组成小团队高效地工作，不仅速度快，效率高，工作的内容更有重点，还消除减弱了神经元节点间的联合适应性，增强了泛化能力。常见的有：
1. 随机隐去一层神经元的layer-wise dropout
2. 纵向隐去神经元的path dropout

## 3. 过拟合和欠拟合
> 参考
> [欠拟合、过拟合及其解决方法
](https://blog.csdn.net/willduan1/article/details/53070777)

### 3.1欠拟合
欠拟合就是模型没有很好地捕捉到数据特征，不能够很好地拟合数据，一般是训练不充分导致的。

解决办法：
1. 添加其他特征项（联想GBDT回归，多加几棵树）
2. 添加多项式特征，例如将线性模型通过添加二次项或者三次项使模型泛化能力更强
3. 减少正则化参数和**weight decay**

### 3.2过拟合
模型过度拟合到了训练集上，通俗一点地来说过拟合就是模型把数据学习的太彻底，以至于把噪声数据的特征也学习到了，导致泛化能力变差。

解决办法：
1. 对数据进行清洗。过拟合可能是数据不纯导致的。
2. 采用正则化方法。
3. 使用dropout（也可以看作特殊的正则化）

## 4. Normalization
> 参考
> [【深度学习】深入理解Batch Normalization批标准化](https://www.cnblogs.com/guoyaohua/p/8724433.html)
> [《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》](https://arxiv.org/abs/1502.03167)

**IID独立同分布假设**：就是假设训练数据和测试数据是满足相同分布的，这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障

**BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的**

### 4.1 “Internal Covariate Shift”问题
Normalization是基于Mini-Batch SGD的，Mini-Batch SGD本身存在Internal Covariate Shift问题，即**batch数据的分布会随着神经网络深度的加深而发生改变，从而违反同分布假设**。

下面主要叙述解决“Internal Covariate Shift”问题的方法

### 4.2 Whitening

whitening （白化）的目的就是降低输入的冗余性，数据经过白化处理应该满足两个条件：
1. 降低（去除）不同维度的相关性
2. 数据每个维度的方差为1

条件1要求数据的协方差矩阵是个对角阵；条件2要求数据的协方差矩阵是个单位矩阵。

简单来说，**白化，就是对输入数据分布变换到0均值，单位方差的正态分布**，而BN可以理解为对深层神经网络每个隐层神经元的激活值做简化版本的白化操作。

### 4.3 BatchNorm的本质思想

因为深层神经网络在做非线性变换前的激活输入值（就是那个x=WU+B，U是输入）随着网络深度加深或者在训练过程中，其**分布逐渐发生偏移或者变动**，之所以训练收敛慢，一般是**整体分布逐渐往非线性函数的取值区间的上下限两端靠近**（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），从而在逆向传播的时候低层网络层失去梯度。这是训练深层神经网络收敛越来越慢的本质原因。

而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，**使得激活输入值落在非线性函数对输入比较敏感的区域**，这样输入的小变化就会导致损失函数较大的变化，意思是这样让梯度变大，避免梯度消失问题产生




一些常问的问题：
1. **在哪里做BatchNorm？**
   
   在X=WU+B计算之后，非线性函数（激活函数）变换之前
   <p align="center"><img src="/docs/DeepLearning/Pics/BatchNorm.png" alt="BatchNorm.png" style="zoom:60%;" />
2. **BatchNorm的两个参数$\gamma$和$\beta$作用是什么？**
   
   BatchNorm可能会导致网络表达能力下降，为了防止这一点，每个神经元增加两个调节参数（scale和shift），其实就是变换的反操作
3. **BatchNorm**的作用？
   1. 提升了训练速度，收敛过程大大加快
   2. 可以看作一种正则化，防止过拟合
   3. 使得对数据初始化要求降低，可以增大学习率等

其实上面的问题都可以在原文的伪代码找到答案**原始论文中BatchNorm的具体操作过程如下：**
<p align="center"><img src="/docs/DeepLearning/Pics/BatchNorm_Code.png" alt="BatchNorm_Code.png" style="zoom:30%;" />

## 5. 梯度更新优化器
> 参考：[深度学习——优化器算法Optimizer详解（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）](https://www.cnblogs.com/guoyaohua/p/8542554.html)

### 5.1 BGD（Batch Gradient Descent）
**TODO**

## 6. 激活函数(activation function)

<p align="center"><img src="/docs/DeepLearning/Pics/sample-activation-functions-square.png" alt="sample-activation-functions-square.png" style="zoom:45%;" />

激活函数负责给神经网络的线性映射引入非线性，提高网络的拟合能力。

以全连接神经网络为例，激活函数在线性计算求和后进行计算，如下：

<p align="center"><img src="/docs/DeepLearning/Pics/activationFunction.png" alt="activationFunction.png" style="zoom:60%;" />

由于需要处理成千上万的神经元的输出，激活函数必须是简单、可以快速计算的。下面介绍几个主要的激活函数
