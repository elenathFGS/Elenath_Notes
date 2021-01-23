# Machine Learning 笔记

> 该笔记记录我学习Machine Learning相关知识的一些思考和知识点

---

## ROC曲线

>Reference [Sklearn ROC曲线的使用](https://blog.csdn.net/hfutdog/article/details/88079934)，[机器学习基础（1）- ROC曲线理解](https://www.jianshu.com/p/2ca96fce7e81)

### 1. ROC曲线定义

> * Precision（查准率）：预测为正例的样本中真正正例的比例
> * Recall（召回率）：真正为正例的样本有多少被预测出来

ROC曲线是Receiver Operating Characteristic Curve的简称，中文名为“受试者工作特征曲线”。

<p align="center"><img src="/docs/MachineLearning/Pics/ROC.jpg" alt="ROC" style="zoom:67%;"/>

ROC曲线的横坐标为假阳性率（False Positive Rate，FPR）（负样本中预测为正的比例）:
$$
FPR = \frac{FP}{FP+TN}
$$
纵坐标为真阳性率（True Positive Rate, TPR）（正样本中预测为正的比例）:
$$
TPR = \frac{TP}{TP+FN}
$$
其中TP，FP，FN，TN可以用混搅矩阵来表示：

<p align="center"><img src="/docs/MachineLearning/Pics/confusionMatrix.png" alt="confusionMatrix" style="zoom:50%;" />

### 2. ROC曲线的理解&应用

**ROC曲线及AUC常被用来评价一个二值分类器的优劣**。其能反映模型在选取不同阈值的时候其敏感性（sensitivity, FPR）和其精确性（specificity, TPR）的趋势走向。不过，相比于其他的P-R曲线（精确度和召回率），ROC曲线有一个巨大的优势就是，当正负样本的分布发生变化时，其形状能够基本保持不变，而P-R曲线的形状一般会发生剧烈的变化。

例如，从上面的混搅矩阵可以看到，$TPR$和$FPR$的值不会因为正样本或者负样本的增加或减少而改变（e.g. 若将负样本的数量增加，可以预见FP,TN都会增加，必然会影响到P,R。但ROC曲线的俩个值，FPR只考虑第二行，则FP,TN也会成比例增加，并不影响其值）



### 3. 如何绘制ROC曲线

以二值分类器为例，模型的输出一般是预测为正或负的Score，而不同的阈值决定了不同Score对应的样本到底是正还是负，ROC曲线正是通过不断移动分类器的“阈值”来生成曲线上的一组关键点。

下图是一个二分模型真实的输出结果，一共有20个样本，输出的概率就是模型判定其为正例的概率，第二列是样本的真实标签。

<p align="center"><img src="/docs/MachineLearning/Pics/binary_classification.png" alt="binary_classification" style="zoom:30%;" />

现在我们指定一个阈值为0.9，那么只有第一个样本（0.9）会被归类为正例，而其他所有样本都会被归为负例，因此，对于0.9这个阈值，我们可以计算出FPR为0，TPR为0.1（因为总共10个正样本，预测正确的个数为1），那么我们就知道曲线上必有一个点为(0, 0.1)。依次选择不同的阈值（或称为“截断点”），画出全部的关键点以后，再连接关键点即可最终得到ROC曲线如下图所示

<p align="center"><img src="/docs/MachineLearning/Pics/ROC_bar_example.png" alt="ROC_bar_example" style="zoom:30%;" />

### AUC面积

AUC（Area Under Curve）就是ROC曲线下的面积大小（沿着ROC横轴做积分），它能够量化地反映基于ROC曲线衡量出的模型性能。AUC的取值一般在0.5和1之间，AUC越大，说明分类器越可能把实际为正的样本排在实际为负的样本的前面，即正确做出预测。

## F score

