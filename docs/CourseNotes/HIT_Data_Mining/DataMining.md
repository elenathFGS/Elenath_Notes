# 哈工大 Data Mining 复习笔记

> 参考资料为Stanford开设的[CS246: Mining Massive Datasets](http://www.mmds.org/)课程

[toc]

# 第一章 模型验证方法入门

## 交叉验证 (Cross-Validation)

> 背景：在机器学习里，通常来说我们不能将全部用于数据训练模型，否则我们将没有数据集对该模型进行验证，从而评估我们的模型的预测效果。交叉验证(CrossValidation)方法思想是为了在不动用测试集之前，就评估一下模型是否过于复杂而引起过度拟合。
>
> Refer: [【机器学习】Cross-Validation（交叉验证）详解](https://zhuanlan.zhihu.com/p/24825503?refer=rdatamining)

### 1. 传统的模型验证方法

一般我们会将整个数据集划分成两个部分，一部分用于训练，一部分用于验证，也就是训练集（training set）和验证集（validation set）。

但是该方法有两个弊端：

1. 验证集和测试集划分方法和各自的分布会影响模型的评估，从而影响各个模型参数的选取
2. 有一部分数据（验证集）不能参与模型训练

### 2. LOOCV (Leave-one-out cross-validation)

LOOCV方法也包含将数据集分为训练集和测试集这一步骤。但是不同的是，我们现在只用**一个数据**作为测试集，其他的数据都作为训练集，并将此步骤重复N次（N为数据集的数据数量）。

假设我们现在有n个数据组成的数据集，那么LOOCV的方法就是遍历数据集，每次训练都取出一个数据作为测试集的唯一元素，而其他n-1个数据都作为训练集。结果就是我们最终**训练了n个模型**，每次都能得到一个MSE。而计算最终test MSE则就是将这n个MSE取平均。
$$
\large CV_{(n)} = \frac{1}{n}\sum_{i=1}^{n}MSE_i
$$
**优点：**不受测试集合训练集划分方法的影响，因为每一个数据都单独的做过测试集，同时用了n-1个数据来做训练集，保证了各个模型bias最小

**缺点：**计算量大

### 3. K-fold Cross Validation

K折交叉验证是相比较上述两种方法的一种折中的办法，k指的是数据样本分组的个数。和LOOCV的不同在于，每次的测试集将不再只包含一个数据，而是多个，具体数目将根据K的选取决定。

以**5-fold**为例，五折交叉验证的步骤如下：

1. 将所有数据集分成5份

2. 不重复地每次取其中一份做测试集，用其他四份做训练集训练模型，之后计算该模型在测试集上的$MSE$

3. 将5次的 $MSE$ 取平均得到最后的 $MSE$
   $$
   \large CV_{(k)} = \frac{1}{k}\sum_{i=1}^{k}MSE_i
   $$

**思考：**K越大，每次投入的训练集的数据越多，模型的Bias越小。但是K越大，又意味着每一次选取的训练集之前的相关性越大（考虑最极端的例子，当k=N，也就是在LOOCV里，每次都训练数据几乎是一样的）。而这种大相关性会导致最终的test error具有更大的Variance。

### 4. **Stratified-K-Fold**

与上面的K-fold Cross Validation相比，**Stratified-K-Fold**对数据集是分层采样，确保测试集和训练集中各类样本的比例和原始的数据集相同。



## ROC曲线

>Reference [Sklearn ROC曲线的使用](https://blog.csdn.net/hfutdog/article/details/88079934)，[机器学习基础（1）- ROC曲线理解](https://www.jianshu.com/p/2ca96fce7e81)

### 1. ROC曲线定义

> * Precision（查准率）：预测为正例的样本中真正正例的比例
> * Recall（召回率）：真正为正例的样本有多少被预测出来

ROC曲线是Receiver Operating Characteristic Curve的简称，中文名为“受试者工作特征曲线”。

<p align="center"><p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/ROC.jpg" alt="ROC"></img>

ROC曲线的横坐标为假阳性率（False Positive Rate，FPR）（负样本中预测为正的比例）:
$$
FPR = \frac{FP}{FP+TN}
$$
纵坐标为真阳性率（True Positive Rate, TPR）（正样本中预测为正的比例）:
$$
TPR = \frac{TP}{TP+FN}
$$
其中TP，FP，FN，TN可以用混搅矩阵来表示：

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/confusionMatrix.png" alt="confusionMatrix" style="zoom:50%;" />

### 2. ROC曲线的理解&应用

**ROC曲线及AUC常被用来评价一个二值分类器的优劣**。其能反映模型在选取不同阈值的时候其敏感性（sensitivity, FPR）和其精确性（specificity, TPR）的趋势走向。不过，相比于其他的P-R曲线（精确度和召回率），ROC曲线有一个巨大的优势就是，当正负样本的分布发生变化时，其形状能够基本保持不变，而P-R曲线的形状一般会发生剧烈的变化。

例如，从上面的混搅矩阵可以看到，$TPR$和$FPR$的值不会因为正样本或者负样本的增加或减少而改变（e.g. 若将负样本的数量增加，可以预见FP,TN都会增加，必然会影响到P,R。但ROC曲线的俩个值，FPR只考虑第二列，则FP,TN也会成比例增加，并不影响其值）



### 3. 如何绘制ROC曲线

以二值分类器为例，模型的输出一般是预测为正或负的Score，而不同的阈值决定了不同Score对应的样本到底是正还是负，ROC曲线正是通过不断移动分类器的“阈值”来生成曲线上的一组关键点。

下图是一个二分模型真实的输出结果，一共有20个样本，输出的概率就是模型判定其为正例的概率，第二列是样本的真实标签。

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/binary_classification.png" alt="binary_classification" style="zoom:30%;" />

现在我们指定一个阈值为0.9，那么只有第一个样本（0.9）会被归类为正例，而其他所有样本都会被归为负例，因此，对于0.9这个阈值，我们可以计算出FPR为0，TPR为0.1（因为总共10个正样本，预测正确的个数为1），那么我们就知道曲线上必有一个点为(0, 0.1)。依次选择不同的阈值（或称为“截断点”），画出全部的关键点以后，再连接关键点即可最终得到ROC曲线如下图所示

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/ROC_bar_example.png" alt="ROC_bar_example" style="zoom:30%;" />

### 4. AUC面积

AUC（Area Under Curve）就是ROC曲线下的面积大小（沿着ROC横轴做积分），它能够量化地反映基于ROC曲线衡量出的模型性能。AUC的取值一般在0.5和1之间，AUC越大，说明分类器越可能把实际为正的样本排在实际为负的样本的前面，即正确做出预测。

# 第二章 可视化与数据降维ll

## 1. 数据可视化

### 1.1 KDE (Kernel Density Estimate)

> 由给定样本集合求解随机变量的分布密度函数问题是概率统计学的基本问题之一。解决这一问题的方法包括参数估计和非参数估计。
>
> **参数估计**又可分为参数回归分析和参数判别分析。在参数回归分析中，人们假定数据分布符合某种特定的性态，如线性、可化线性或指数性态等，然后在目标函数族中寻找特定的解
>
> **非参数估计**，即核密度估计方法。由于核密度估计方法不利用有关数据分布的先验知识，对数据分布不附加任何假定，是一种从数据样本本身出发研究数据分布特征的方法

**个人理解：**核密度估计就是用数据分布直方图通过小区间采样来和核函数来估计每个横坐标对应的密度，让离散的数据分布变成连续的密度函数。是一种<u>非参数估计方法</u>。

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/KDE.jpg" alt="KDE" style="zoom:33%;" />

### 1.2 其他的各种plot图及其关系看PPT



## 2. MDS: Multiple Dimensional Scalling

**该方法的核心是要求原始空间中样本之间的距离在低维空间中得以保持**

基本推导流程：

1. 样本数量$n$，原始空间 $D \in R^{n \times d}$，距离矩阵$Dist \in R^{n\times n}$，目标降维后的空间$Z\in R^{n\times d'}$，$d'<d$
2. 先计算降维空间$Z$的内积矩阵 $B = ZZ^T$
3. 用约束条件 $d_{ij}^2=Dist_{ij}^2$和中心化的目标样本空间$Z$，得到<u>可用原始样本空间的距离矩阵$Dist$求得内积矩阵$B$</u>
4. 然后对 $B$ 进行矩阵特征值分解，得到$Z$

## 3. 流形学习（Manifold Learning）

> 流形即多样化的形体： 天地有正气，杂然赋**流形**----文天祥《正气歌》

<u>许多高维的数据本质上并不是高维度的，数据可能分布在高维空间的一个低维的流形上面</u>。例如：二维空间的圆，本质上可以用一维空间表示（极坐标的半径），三维空间的球面可以用（经度，纬度）二维空间坐标表示。

流形学习借鉴了拓扑流形概念的降维方法，能够刻画数据的本质：将数据从高维空间降维，还能不损失信息。其采用的是非线性降维：降维的过程不但考虑到了距离，更考虑到了生成数据的拓扑结构

### 3.1 等度量映射 （Isometric Mapping (Isomap) ）

认为低维流形嵌入到高维空间后，直接在高维空间中计算直线距离具有误导性。<u>因为高维空间中的直线距离在低维嵌入流形上是不可达的。</u>例如，地球上南极到北极之间的距离，可以直接计算这两点之间的距离，但是这种距离是毫无意义（总不能从南极打个洞到北极吧），因此引入了测地距离

**做法**：借助流形在局部上与欧式空间同胚的性质，对每个点基于欧式距离找到其近邻点，然后建立近邻连接图，从而获得测地线距离的近似

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/Isomap.png" alt="Isomap" style="zoom:45%;" />

* 例如上面这幅图，上下两个点距离，从一点出发，逐个经过曲面上的近邻点到另一点，而不是直接画一条直线

### 3.2 Isomap算法

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/IsomapAlgo.png" alt="IsomapAlgo" style="zoom:45%;" />



# 第三章 基本预测模型

## 3.1 数据获取

根据已有的特征构造新的特征



## 3.2 超参调优

1. 网络搜索（grid search）和人工搜索：遍历所有的参数可能性
2. 随机搜索：从超参数空间中随机选择参数组合
3. 贝叶斯搜索（Bayesian Search）：利用概率模型来进行模型调优，下一次超参数的构建基于之前超参数对应模型的效果。核心想法是**在更有可能得到更好结果的超参数范围内选择新的超参数**。



# 第四章 推荐系统

> 信息超载+时间有限：需要系统推荐来辅助用户主动搜索

## 4.1 用户建模两种方法

1. **Explicit Feedback （显性反馈）**：用户定制和用户评分
   1. 定制：用户对系统所列问题的回答，如年龄、性别、职业等
   2. 评分：两极评分和多级评分
2. **Implicit Feedback（隐性反馈）**：是否点击、停留时间、是否加入收藏、评论内容等



## 4.2 推荐算法

> 四个简单介绍，两个重点介绍
>
> 效用矩阵：User-Item矩阵，矩阵值为用户对此项的喜好程度。通常该矩阵是稀疏的，推荐系统目标是预测矩阵的空白元素，如基于矩阵分解方法。

**基于人口统计学的推荐**：仅使用用户的基本信息衡量用户的相似性，将与当前用户相似的用户所偏好的物品推荐给当前用户

**基于关联规则的推荐**：比如购买牛奶的同时很多人 也会购买面包。典型的为Apriori算法。

**基于效用的推荐**：用户对项目的效用描述函数，然后用该函数对所有项目进行排序，取前 N 个项目作为对目标用户的推荐。

* Pro: 能在决策的时候考虑诸如 到货时间之类的项目非自身因素问题，提高了推荐的全面性
* Con: 难点是如何设计出考虑周全且性能良好的效用 函数，且该函数的使用不具有通用性

**基于知识的推荐**：某种程度是可以看成一种推理(Inference)技术。针对特定领域制定规则(rule)来进行基于规则和实例的推理。

### 4.2.1 基于内容的推荐系统(Content-based systems)

#### 1. 项模型 (Item Profiles)

* 对各个项的特征进行描述，如电影推荐中电影的演员、流派等，基于项的相似度来进行推荐
* 由特征-值构成，一般特征用布尔向量描述，如下

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/Item-based.png" alt="Item-based" style="zoom:30%;" />

> 两部电影的profile：各包含5个演员，有2个同时出现在两部电影中，一共8位不同的演员（前8列），最后一列是电影的评分（α是缩放因子）

#### 2. 用户模型 (User Profiles)

为项建立向量表示后，需要将用户的偏好表示成同一空间下的向量，<u>例如：需要将用户对电影的打分转换为用户对各个演员的评分，如下</u>

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/ContentSystem.png" alt="ContentSystem" style="zoom:45%;" />

> 解释：
>
> B用户平均分4.7，演员A1出现在HP1和HP3中，则在B的模型中，A1对应分量应该是（5+4）/2 -4.7=-0.2
>
> 或计算效用矩阵用户中心化向量与项模型列向量点积的均值 : (0.3,0.3,-0.7…).(1,0,1,…)=-0.4 -0.4/2=-0.2 

#### 3. 基于项模型和用户模型进行推荐

转换为同一空间的项模型和用户模型后，只需要计算对应向量之间的余弦距离，即可估计用户喜好某个项的程度。如：估计用户A喜欢哈利波特1的程度
$$
u(A,HP1) = cos(A,HP1) = \frac{A \cdot HP1}{||A||\cdot ||HP1||}
$$

#### 4. 优缺点

Pros: 不需要其他用户的信息、解释性强、可以推荐冷门项目

Cons：无法利用其他用户来判断、过度特化、适当的特征难以确定

### 4.2.2 协同过滤系统（Collaborative filtering systems）

> <u>基于内容的推荐方法使用项的特征来确定项的相似度, 协同过滤方法关注对两个项的用户评分之间的相似度</u>
>
> 先识别相似用户，再进行推荐

#### 1. 基于邻域的协同过滤

##### 1.1 基于用户的（User-User）

适用于item更新频繁的应用。根据用户对物品的行为（如都购买某些的物品），找出兴趣爱好相似的一些用户，将他们大都喜欢的东西推荐给另一个用户。

该算法最大的问题是<u>如何判断并量化两个用户的相似性</u>，如歌曲的播放顺序，收藏或者跳过，如何量化这些行为？

具体算法如下（其中改进相当于用用户相似度加权）：

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/User-User.png" alt="User-User" style="zoom:33%;" />

##### 1.1 基于项的（Item-Item）

适用于user更新频繁的应用。给用户推荐和他原有喜好类似的物品。这种相似是基于项的共同出现几率（例如用户买了X，同时时也买了Y）。具体算法如下：

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/Item-Item.png" alt="Item-Item" style="zoom:50%;" />

> 在实践中，往往Item-Item方式效果好，因为Item比User要简单些，User可能有多种偏好

#### 2.基于模型的协同过滤（U-V分解）

矩阵因子分解将**用户偏好**矩阵分解成一个**用户-潜在因子**矩阵乘以一个**潜在因子-项**矩阵，代表用户和项之间的潜相互作用。

<u>潜在因素模型（Latent Factor Models）</u>：通过数量相对少的**未被观察到的底层原因**，来解释大量用户和产品之间可观察的交互。如：**UV分解**

**UV分解**步骤：

1. 初始化 $u\times k$ 的 $U$ 矩阵和 $k\times v$ 的 $V$ 矩阵。
2. 交替地改变U和V矩阵的某个元素值，设为x，并带入x求出两矩阵的乘积，与原矩阵求RMSE，是带x的表达式，对表达式对x求导，得最佳x值
3.  重复2直到收敛

## 4.3 推荐算法的冷启动问题

### 三种冷启动问题：

1. **用户冷启动**：如何给**新用户**做个性化推荐
2. **物品冷启动**：如何将**新物品**推荐给用户
3. **系统冷启动**：新网站在**数据稀少**的情况下如何做个性化推荐



### 解决冷启动的办法

1. **提供非个性化的推荐**：给用户推荐热门排行榜，等到用户数据收集到一定的时候，再切换为个性化推荐
2. **利用用户注册信息**：如人口统计信息、用户自己填写的兴趣描述、用户在其他网站的行为
3. **选择合适的物品启动用户的兴趣**：用户在登录时对一些物品进行反馈，收集用户对这些物品的兴趣信息
4. **利用物品的内容信息**：（针对物品冷启动）
   1. UserCF算法：一旦有用户反馈新物品，将其推荐给相似用户
   2. ItemCF算法：利用Item的相似度
5. **采用专家标注**：找人标注物品的各个维度特征，再根据这些特征做相似度计算




# 第五章 在线广告计算问题

> 互联网广告是迄今为止，大数据领域唯一形成规模化营收的应用
>
> * 离线算法：将算法所需要的所有数据准备好，然后算法以任意次序访问数据，最后得出结果
> * 在线算法：搜索查询到达时，必须立刻选择跟搜索结果一起显示的广告

## 5.1 竞争率

**竞争率定义**：算法的竞争率是所有可能输入下得到的最差结果和最优结果的比值。
$$
\large Competitive \ ratio = min_{(all \ possible \ inputs)}(\frac{|M_{algo}|}{|M_{opt}|})
$$
其中$M$是算法的输出结果（如果是匹配算法就是匹配的 **节点对** 数量）



## 5.2 二分图匹配

### 5.2.1 二分图匹配的贪心算法

按照某个顺序遍历所有边，遇到边 (x, y) 时，如果x和y都不是已有匹配中的边的端点，则将其加入，否则（x或y已经有边匹配了）跳过。 

### 5.2.2 二分图贪心算法匹配的分析

#### 1. 贪心算法的竞争率上界

贪心算法的竞争率上界为 $\large \bold{\frac{1}{2}} $ ，可以举特例，或者直观上分析： 

假设OPT（最优）的算法可以匹配的节点对为$X$，最差的情况就是，在贪心算法中，前一半的节点的匹配的正好把后一半的节点的匹配给占用了（最多也只可能占用这么多），而前一半节点本来可以不占用这些匹配的，因此贪心算法在该情况下匹配的节点数为 $ \frac{X}{2}$。如下图所示：

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/binaryMatch.jpeg" alt="binaryMatch" style="zoom:50%;" />

> 蓝色的是OPT算法的匹配，红色的是greedy算法得到的匹配



#### 2. 贪心算法的竞争率下界

贪心算法的竞争率下界也为 $\large \bold{\frac{1}{2}} $，证明如图，图片来源于[CS246: Mining Massive Datasets](http://www.mmds.org/)

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/greedyLower.png" alt="greedyLower" style="zoom:40%;" />

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/greedyLower2.png" alt="greedyLower2" style="zoom:40%;" />

## 5.3 Adwords问题在线算法

> 一句话概括，Adwords问题研究的是一个网站广告投放的策略，让网站的拥有者获得最大收益

### 5.3.1 Adwords问题设置

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/Adwords算法.png" alt="Adwords算法" style="zoom:50%;" />

### 5.3.2 Adwords问题Greedy算法

**Greedy**: **每次选择 $\large 投标价格（Bid）\times 点击率（CTR）$ 最大的广告商来给出广告结果。**最差竞争率为 $\large \bold{\frac{1}{2}} $

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/greedy_Adwords.png" alt="greedy_Adwords" style="zoom:33%;" />



### 5.3.3 Adwords问题Balance算法

**Balance**: **将查询分配给出价最高且剩余预算最多的广告商**。对于两个advertiser的情况，竞争率为 $\large \bold{\frac{3}{4}} $

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/balance.png" alt="balance" style="zoom:33%;" />

**分析**：

<p align="center"><img src="/Users/elenath/Library/Application Support/typora-user-images/image-20201210182932906.png" alt="image-20201210182932906" style="zoom:40%;" />

**Balance算法的一般情况（>2的advertiser）**

竞争率为 $ \bold{ 1 - \large\frac{1}{e}}$ ，证明见[CS246: Mining Massive Datasets](http://www.mmds.org/)



# 第六章 社会网络图挖掘

## 1. Community Detection (社区发现)

> 社区发现是**社会网络聚类**的一个分支，需要考虑不仅仅是图上边的权重，更重要的是图的结构信息，因此引入了边的中介度 (Edge Betweenness) 概念

### 1.1 社会网络图的局部性

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/SocialNetwork.png" alt="SocialNetwork" width=400 />

下面计算存在边(X,Y)和(X,Z)时，边(Y,Z)也存在的概率：

* 当X是{A、C、E、G}中的某一个时（度为2）：边(Y,Z)都存在，共4个正例

* 当X是{F、B}中的一个时（度为3）：有3个正例、3个反例

* 当X=D时，4个邻居节点的6个可能的节点对中，只有两条边，所以有2个正例，4个反例

**正例一共9个，反例一共7个**，所以边(Y,Z)存在的概率为9/(9+7)=0.563

明显高于期望值0.368。上图确实表现出社会网络中期望出现的**局部性**


### 1.2 Girvan-Newman算法（边中介度算法）

**边(a,b)的中介度**（Betweenness）：最短路径通过这条边的节点对(x,y)的数目。

更准确的定义是，如果x和y之间存在多条最短路径，则边(a,b)的贡献记为这些最短路径中通过边(a,b)的比例，否则若只有一条最短路径，则贡献度为1，如下图所示：（尝试算一下那个中介度为7.5的）

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/betweenness.png" alt="betweenness" style="zoom:50%;" />

**Girvan-Newman算法**本质是一个Hierarchical的方法，思想是每一轮计算各个节点间的中介度，然后将中介度最大的那个（些）边删去，然后根据删边的位置来划分社区。如下图所示：

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/GN-algo.png" alt="GN-algo" style="zoom:30%;" />

（原来的图是7和8间有一条边，是中介度最大的，所以第一步删去了）

算法描述如下：

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/Girvan-Newman.png" alt="Girvan-Newman" style="zoom:30%;" />



### 1.3 G-N算法中，中介度如何计算

1. 算法访问每个节点X一次，以X为根，计算X到其它连接节点的最短路径数目

   这一步采用BFS算法，每个父节点最短路径数目为子节点的最短路径数目之和，如下图

   <p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/GN_BFS.png" alt="GN_BFS" width=500 />

2. 将每个节点用根节点X到它最短路径的数目来标记（如上）

3. 自底向上计算，根据第2步得到的最短路径数目来计算各个边的贡献度（对初始贡献度1加权，权重为子节点和父节点的比值）

   <p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/GN_compute.jpeg" alt="GN_compute" width=500 />

4. 以每个节点为根重复上述计算过程

5. 最后对计算得到的贡献值求和，将这些和**除以2**得到真正的中介度（边是对称的，实际每条边贡献了两次）



## 2. 社区的直接发现

> 直观思路是找图的一个大团(**Clique**)，但是寻找最大团是一个很难的NP-完全问题，即使在一个有很多边的大图中，也不一定存在大的团

### 2.1 完全二部图与频繁项挖掘

**完全二部图(Complete Bipartite Graphs)：**由一边的s个节点和另一边的t个节点组成，这两部分任意一对节点之间都有边。记为$\large K_{(s,t)}$

完全二部图的发现和频繁项挖掘是等价的，如下所示。

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/KST.png" alt="KST" width=500 />

## 3. Partitioning of Graphs（图划分）

> 如何得到一个“好”的图划分?
>
> * 最大化分支内的连接数量
> * 最小化分支之间的连接数量

最直接的方法是使用最小割，但是**最小割不一定是最优划分**，因为最小割只关心分支外部的连接，没有考虑分之内部的连接

### 3.1 归一化割（Normalized Cuts）

> 归一化割能够产生更加均衡的划分

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/NormalizedCuts.png" alt="NormalizedCuts" width=550 />

### 3.2 谱方法的图划分

**基本思想就是：基于拉普拉斯矩阵的第二小特征值计算中的$x$向量的正负值划分**

* **邻接矩阵**：$n\times n$的矩阵$A$，其中$A_{ij}$当且仅当节点i和j之间有边
  * 是对称矩阵
  * 特征向量是实正交向量

* **d-正则图：**图G中每个顶点的度都为d
  * $x = (1,1,...,1), \lambda=d$是d正则图G的邻接矩阵特征向量和特征值
  * 定理：设G是一个连通图，则邻接矩阵A最大特征值的绝对值等于G中最大的度$\Leftrightarrow$𝐺 是正则的

#### 图的拉普拉斯矩阵

$$
\Large L = D - A
$$

* 特征值都是非负实数，最小特征值为0
* 特征向量是实正交向量

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/LAP_min.png" alt="LAP_min" width=400 />

由上述性质，x必然有正也有负，将正的和负的进行划分即可

## 4. 图生成模型（Generative model for networks）

> **目标**：定义一个可以产生网络的模型
>
> * 模型有若干个参数构成
> * 参数确定任一具体实例的生成概率，该概率称为这些参数值的似然(likelihood)

### 4.1 关系图模型：Affiliation Graph Model （AGM）

模型表示：
$$
B(V,C,M,\{{p_c}\})
$$

* 图节点**V**, 给定数目的社区**C**, 成员关系**M**
* 社区C中任意两个节点有边连接的概率为$p_c$

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/community_detect.png" alt="community_detect" width=500 />

总体来看，整个图中任意两个节点u,v间有边的概率为：
$$
\Large P(u,v) = 1-\prod_{c\in M_u \cap M_v} (1-p_c)
$$
解释：

1. 若两个节点都在一个社区里面，那么只有一项参与累乘 

2. 若在交叠的多个社区里面，则概率是$1-所有这些社区都没边的概率$
3. 否则，若在不相交的两个社区，$p_c=0$，即$P(u,v)=0$

### 4.2 极大似然估计求图生成模型

给定某个特定的图，假定图的模型是上面的AGM模型（任意两点有边的概率上吗已经给出），用极大似然来求模型的参数
$$
\underset {B(V,C,M,\{{p_c}\})}{argmax} \prod_{u,v\in E}P(u,v) \prod_{uv\notin E}(1-P(u,v))
$$
其中$E$是给定的某个确定的图的边的集合

> 这个模型是离散的，其优化需要conbinatorial search，因此需要将模型变成连续的，来使用类似梯度下降的方法进行优化



### 4.3 BigCLAM：AGM的relaxed版本

> **Relaxation:** 考虑个体属于社区的隶属强度

隶属强度 $F_{uA}$：对每个社区A和个体u，隶属强度参数，可取任何非负值。（$F_{uA}$=0: 该个体确定不属于该社区) 

这样，任意社区A中的两个节点$u,v$存在边的概率变成：
$$
\large P_A(u,v) = 1- \frac{1}{e^{F_{uA} \cdot F_{vA}}}
$$
这里$F$是一个矩阵，各行表示各节点，列为节点对各个社区的隶属强度

考虑整个图，任意两点存在边的概率可以推导为：

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/AGM_weighted.png" alt="AGM_weighted" width=300 />

同样，基于一个确定的图，可以用极大似然来估计$F_i$
$$
argmax \prod_{u,v\in E}P(u,v) \prod_{uv\notin E}(1-P(u,v))
$$


## 5. 寻找三角形算法

**重节点**：图中度数 $\ge \sqrt m$ 的节点，重节点的数目不可能超过$2\sqrt m$

**寻找重节点三角形算法**：

1. 找到所有的重节点。（最多$\sqrt m$个重节点，复杂度为 $O(\sqrt m)$）

2. 对所有重节点的三节点组合，验证其相互是不是都有边。（由于对边有索引，验证边复杂度为O(1)，这一步复杂度为 $C_{\sqrt{m}}^3=O(m^{3/2})$）
3. 全局时间复杂度为$O(m^{3/2})$

**寻找其他三角形的算法**

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/sanjiaox.png" alt="sanjiaox" width=500 />

m条边，$O(m×m^\frac{1}{2})$

上面的那个计数部分问题，目的是防止重复计数（边是对称的）

因此，寻找三角形算法的整体时间复杂度为$\large \bold{O(m^{3/2})}$



# 第七章 异常检测

> 异常的特征：少而不同
>
> 1. 异常是少数数据
> 2. 异常数据与正常数据相比，具有非常不同的属性值

## 1. 异常检测算法分类

1. **基于统计学的方法：**假定数据是由某个随机模型产生，如果某个数据概率很低，那么它是异常的概率就很大。

   如高斯模型产生的**数据的 $\large 3\sigma$ 准则** （$\pm\sigma, \pm2\sigma, \pm3\sigma$ 区间分别对应概率 $0.64, 0.95, 0.99$）

2. **基于临近性的方法：**如果一个数据离它最近的邻居都很远，那么它很可能是异常。如KNN算法

   如计算每个向量与均值向量之间**马氏距离**，若数值太大就认为异常

3. **基于聚类的方法：**假定正常数据的数量很大，很稠密，而异常值则很小，很稀疏。

   如下面讲到的 **局部异常因子算法 LOP**和 **隔离森林算法**

## 2. 基于隔离森林的异常检测

基本思想：用一个随机超平面来分割数据空间，那些密度很高的簇需要被切很多次才会停止切割，但是那些密度很低的点很快就被划分到一个子空间了

隔离森林由多棵隔离树（**iTree**）构成，每棵树都对各个样本根据属性进行递归划分，隔离一个数据点所需的划分数量等于从根节点到终止节点的路径长度，多棵隔离树的平均路径长度来计算期望路径长度

**隔离森林采用子采样算法**：不需要隔离所有正常情况，在采样量很小的情况下仍然有良好的表现

隔离森林使用异常评分来衡量样本的异常程度

## 3. LOF：局部异常因子算法

主要思想：通过比较每个点p和其邻域点的密度来判断该点是否为异常点，**点p的密度越低，越可能被认定是异常点**。

关键概念定义：

1. **k-distance：第k距离**：距离p第k远的点的距离，不包括p

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/k-dis.png" alt="k-dis" width=500 />

2. **k-distance neighborhood of p：第k距离邻域 $N_k(P)$**：p的第k距离及以内的所有点，包括第k距离 

   <p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/kima.png" alt="kima" width=400 />

3. **reach-distance 可达距离**：$reach−distancek(p,o)=max\{k−distance(o),d(p,o)\}$

   1. 如果是离点o最近的k个点，o到它们的可达距离被认为相等，为o的k-distance；
   2. 否则就是真实距离

4.  **local reachability density：局部可达密度**：表示点p的第k邻域内点到p的平均可达距离的倒数。（注意是p的邻域点 Nk(p)到p的可达距离，不是p到 Nk(p)的可达距离）(下面图错了。分子中应该是$reach-dist_k(o,p)$)

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/k-dist.png" alt="k-dist" width=300 />

5. **local outlier factor 局部离群因子**：表示点p的邻域点 $N_k(p)$的局部可达密度与点p的局部可达密度之比的平均数。

   <p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/kform.png" alt="kform" width=500 />

   如果这个比值越接近1，说明p的其邻域点密度差不多，p可能和邻域同属一簇；如果这个比值越小于1，说明p的密度高于其邻域点密度，p为密集点；如果这个比值越大于1，说明p的密度小于其邻域点密度，p越可能是异常点。

## 4. 重采样方法

> 信用卡诈骗问题中，Fraud case数量占比非常小，导致衡量模型的准确率等matric都很高，其实模型实际效果可能并不好

分为两类

1. **过采样**：通过采样通过增加少数类使数据达到平衡

2. **欠采样（undersampling）**：通过减少多数类来使数据达到平衡

   <p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/oversample.png" alt="oversample" width=400 />

### 3.1 欠采样（undersampling）

**欠采样（undersampling）**：通过减少多数类来使数据达到平衡

#### 剪辑最近邻算法 ENN（Editd Nearest Neighbor Rule）

> 剪辑近邻能够除去kNN训练集中样本交错位置附近的样本，即从分类器中去除可能影响分类效果或造成overfitting的样本点，使分类面更平滑。

对于集合中的每一个**多数类**样本，考察它的k 个近邻，如果它的类别与这 k 近邻的多数类别不同，就**抛弃**这个样本

#### 压缩最近邻算法 CNN（Compressed Nearest Neighbor Rule）

> 压缩样本的思想，它利用现有样本集，逐渐生成一个新的样本集。使该样本集在保留最少量样本的条件下, **仍能对原有样本的全部用最近邻法正确分类**

* 首先将所有的少数类和其中一个多数类样本组合成子集S，剩余的多数类样本构成子集C
* 通过逐一遍历集合C, 对C中每一个样本考虑它在S中的 1-近邻, 如果它的类别与近邻给出的决策结果不一致时（说明该样本不能被舍弃掉，不然正确率无法保持）, 将该样本加入到S 中, 并从 C 中移除
* 循环遍历 C 直到没有样本再加入 S 为止

#### 改进CNN $\rightarrow$ 单边选择OSS（one side selection）

> 在CNN 的基础上进一步对噪声型和边界型的样本数据进行清洗

**tomek 连接**：若两个**不同类别**的样本x和y满足 不存在任意一个样本z使得$d(x,z)<d(x,y) 或d(y,z)<d(x,y)$，则称x和y存在tomek连接

单边选择在完成CNN 后得到集合C, 再从 C 中清除所有存在tomek 连接的**多数类**样本,得到最终的采样集合

#### NearMiss系列

1. 选择到最近的K个少数类样本平均距离最近的多数类样本
2. 选择到最远的K个少数类样本平均距离最近的多数类样本
3. 对于每个少数类样本选择K个最近的多数类样本，目的是保证每个少数类样本都被多数类样本包围



### 3.2 过采样

#### SMOTE算法

基于k近邻用两个旧样本合成新样本

<p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/makenew.png" alt="makenew" width=500 />



# 第八章 GBDT, XGBOOST,LightGBM

三个方法都是**Boosting方法**：迭代地构造一系列的弱分类器，然后将这些分类器的结果组合成一个预测函数。每一个分类器都基于前一个分类器的残差进行学习。

* Adaboost：对各个分类器进行加权，对准确率高的分类器给予大的权值。
* 随机森林（random forest）
* GBDT

## 1. GBDT

**GBDT（Gradient Boost Decision Tree）**：迭代地建立多棵决策树，每棵新树都基于之前模型的残差的负梯度方向学习，最后所有树的结果累加起来得到最终结果。

* GBDT的每个节点的预测值为分布在该节点的各个样本的均值
* 节点分裂的标准不是最大熵，而是最小化损失函数（Log损失、平方误差）
* 通过穷举**每一个特征**的**每个阈值**，寻找最好的分割点
* GBDT采用梯度下降的方式进行训练，每棵新树都基于之前模型的残差的负梯度方向学习
* 若损失函数是均方误差，则求导后得负梯度就是2倍的残差 $2(y_i-F(x_i))$

## 2. XGBOOST

**XGBOOST (Extreme gradient boosting Tree)**：是对GBDT的一个改进，同样用的是梯度提升的办法，但是在**目标函数中加入了正则化项**，可以用来防止模型过拟合。（accuracy 和 complexity 的 trade off）

* XGBOOST也使用加法训练（梯度提升），但是加入了树复杂度的惩罚（正则化项），防止过拟合同时使得生成的模型更加简单

* 各个节点可以使用贪心策略进行分裂，增益函数如下

  <p align="center"><img src="/docs/CourseNotes/HIT_Data_Mining/Pics/gain.png" alt="gain" width=550 />

* 传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数
* 在各个特征上对所有样本做了预排序，将样本按照特征取值排序，然后从全部特征取值中找到最优的分裂点位，该算法的**候选分裂点数量与样本数量成正比**。

## 3. LightGBM

**LightGBM**：是对XGBOOST的进一步改进，主要特性如下

* **LightGBM**采用了GOSS（Gradient-based One-Side Sampling）技术来节省时间同时保持精确度，根据数据的梯度绝对值排序，保留所有的梯度较大的样本，在梯度小的实例上使用随机采样。

  > 基本假设是对损失函数的负梯度进行拟合，样本误差越大，梯度的绝对值越大，这说明模型对该样本的学习还不足够，相反如果越小则表示模型对该样本的学习已经很充分

* **LightGBM**用直方图算法来替代XGBOOST中对预排序算法，大大减少了寻找分裂点时的计算量。通过将连续特征值离散化到固定数量(如255个)的$bins$上，使得候选分裂点为**常数**个$num\_bins -1$.

  * 同时直方图的计算有trick，一个叶子的直方图可以由它父节点的直方图与它兄弟直方图做差得到
  * bin的索引数量（常数a）也比原数据量排序后索引的数量（数据量N）少很多，可以用更小的内存来存储

* XGBOOST采用的是Level-wise的树增长策略（每次增长所有叶子结点都分裂），而**LightGBM**采用的是Leaf-wise的增长策略（考虑在Loss最大的子节点上分裂），虽然可能导致过拟合，但是可以通过控制树的高度来控制
* **LightGBM**可以使用分布式来进行并行运算，不同的worker分配不同数据，每个worker先选出最佳的几个划分特征，然后消息汇总给全局进行voting，选出最佳的几个特征建立直方图，最后生成分裂点的值



# 第九章 分布式机器学习系统

> 大数据和大模型需要：
>
> 1. **并行度**更高的处理器或者**集群**来完成训练任务
> 2. **分布式存储**

## 1. 分布式机器学习的三个模块

分布式机器学习系统包括三个模块：**数据和模型划分模块、单机优化模块、通信模块**

### 1.1 划分模块

**数据划分**的两个角度：

1. 对训练样本划分：
   1. 随机采样：对数据有放回的采样 -> pro：分布和原来一致 cons：计算复杂度较高、低频数据难以选出
   2. 置乱切分采样：数据随机打乱并按照工作节点数目k来划分k份 -> pro:计算量小、能保留每个样本 cons：分布和原来的有偏差
2. 对每个样本按照特征维度进行划分：可能产生较大的通讯开销

**模型划分**：将模型划分为多个子模型，划分方法决定了模型间的依赖关系和通信开销

1. 线性模型：对不同的特征维度进行划分
2. 神经网络：
   1. 横向逐层神经元划分：子模型实现简单，但是并行度不高（梯度需要一层层传递）
   2. 纵向跨层划分：模型依赖关系复杂，实现难度大，通信代价高

### 1.2 单机优化模块

除去各工作节点之间相互的通信以外，在每个工作节点里，基本就是一个传统的单机机器学习任务

### 1.3 通信模块

每个工作节点会学到基于局部数据的子模型，为实现全局的信息共享，需要把这些子模型或**子模型的更新(如梯度)**作为通信的内容；也可以以数据作为通信内容，如**重要的样本**或者**计算的中间结果**等。



## 2. Iterative Mapreduce

> 迭代式的mapreduce

### 2.1 为什么不能基于原始MapReduce实现？

> 原始的MR更擅长单次的MR任务

1. 任务每次迭代开始都需要重新初始化
3. 通信和数据转换开销大
   1. 每轮结束结果都写入文件，下一轮还要重新读取 
   2. 静态数据和动态数据不区分对待，每次都要重新加载静态数据

### 2.2 Iterative MapReduce（IMP）模型

1. 静态数据和动态数据区分对待，静态数据只加载一次
2. 对Map和Reduce任务进行缓存，不需要每次初始化
3. Reduce结束由全局的combine操作

### 2.3 IMP的通讯模块 -> AllReduce接口

**直接让一个GPU当Reducer收集数据？**
直接从单个GPU发送和接收数据的机制中，单个GPU必须从所有GPU接收所有参数，并将所有参数发送到所有GPU。系统中的gpu越多，通信成本就越大。

#### 星形的拓扑结构

环中的GPU都被安排在一个逻辑环中。每个GPU应该有一个左邻和一个右邻;它只会向它的右邻居发送数据，并从它的左邻居接收数据。**通讯成本是恒定的，只由系统中带宽最低的GPU决定**

星形拓扑中，**假设有N个节点，则每个节点中的数据被划分成N个块**（这个特性是由其通讯的性质决定的）

#### 星形拓扑的两个通信步骤

1. **scatter-reduce**

   进行N-1次 Scatter-Reduce 迭代，每次迭代，GPU将向其右邻居发送一个**块**，并从其左邻居接收一个块并累积到该块中。

   第n个GPU从发送块N和接收块N - 1开始，这一步以后，每次迭代都发送它在前一次迭代中接收到的块

2. **Allgather**

   在scatter-reduce步骤完成之后，每个GPU都有一个块是最终的值，需要交换这些块，以便所有gpu都具有所有必需的值。

   过程与scatter-reduce相同(N-1次迭代)，只是gpu接收的值没有累加，而是简单地**覆盖**块。

N个GPU中的每一个都将发送和接收N-1次scatter-reduce，N-1次allgather。每次，GPU都会发送K / N值，其中K是单个节点数据总量。因此，传输到每个GPU和从每个GPU传输的**数据总量**为：
$$
2(N-1)K/N
$$

### 2.4 参数服务器

系统中的节点被逻辑上分为工作节点(worker)和服务器节点(server)，各个工作节点主要负责处理本地的训练任务，并通过客户端接口与参数服务器通信， 从**参数服务器处获取最新的模型参数，或将本地训练产生的模型(或模型更新) 发送到参数服务器**。





## 3. 分布式学习算法

### 3.1 单机学习

梯度下降法：

1. 随机梯度下降：一个一个读入样本进行训练并及时更新参数，常用于大规模训练，往往容易收敛到局部最优
2. 批量梯度下降：每次加载一批样本，对参数的update进行累计，然后更新参数。用于在已知整个训练集时的一种训练方式，但对于大规模数据并不合适。
3. 小批量梯度下降：相比于随机梯度下降，小批量采样可以有效地减小方差，是个折中的办法

### 3.2 分布式学习：同步算法

#### 1. 同步SGD算法（SSGD）

对各个节点进行本地训练，然后将得到的梯度叠加起来

* **等价于一个批量大小增大K倍的单机SGD算法**
* 适用于小批量训练开销大，而模型规模小的情况

* 增加batch size可以减小方差，但是也会增加一次迭代的代价，需要折衷

#### 2. 模型平均方法（MA）

各个节点的模型进行本地训练，然后用本地的梯度更新参数。然后同步通信获得所有节点上的参数的平均，并用来更新全局模型，全局模型作为新模型同步分发给各个节点

* 按照通信间隔的不同，可以分为两种情况：
  * 只在所有工作节点完成本地训练之后，做一次模型平均：**通讯开销小，但是模型差异可能增大**
  * 本地完成一定轮数的迭代之后，就做一次模型平均，然后用这次平均的模型的结果作为接下来的训练的起点，继续进行迭代：**通讯开销大，但是模型差异小，不容易落入局部最优**
* 可以在求模型参数平均的时候与当前的全局模型参数求差异，加入动量



### 3.3 分布式机器学习：异步算法

各个工作节点不再需要互相等待，以一个或多个全局服务器作为中介，实现对全局模型的更新和读取

#### 1. 异步SGD算法（ASGD）

**和上面的区别是，每个模型计算完梯度后，直接发往参数服务器，而不是等待其他节点**

* 参数服务器若收到新的梯度送来，则接收梯度，并更新服务器端端模型；
* 如果收到参数获取请求，则发送参数给对应工作节点

**ASGD**会产生延迟现象，比如：**用一个比较旧的参数计算了梯度，将“延迟”的梯度更新到了最新的模型参数上**