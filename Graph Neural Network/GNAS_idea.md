# My GNAS Idea

> 这里是我整理的自己的GNN文章可能用到的idea和论述点，每个点都是从自己看文章的过程中摘录或者总结下来的，麻烦以后的我写文章之前看看哈
>
> 一个典型的图神经网络主要包括一个用于传播邻居节点信息的**类卷积算子**（必选），以及一个**类池化算子**（可选）。另外，在图网络规模较大时，通常也会采用一个**图采样器**来降低计算复杂度。

## IDEA

* 架构上和oprator上的搜索
* 多任务搜索（graph classification, node classification）
* 考虑how powerful那篇论文中，三种aggregator的不同和MLP的优势，以及**各种操作可以发挥优势的条件**，写的时候对引入各种操作的理论解释，可以参考引用
  * 因为各种图都有各自的特征，有的需要架构特征来区分，此时要用MLP aggr，有的节点特征已经很有区分度了，就可以用mean aggr，而有的只需要代表性的节点的特征，此时可以用max aggr
* Multi-Objective? Accuracy and inference latency tradeoff
* Progressive? 划分为细粒度的，从简单搜到复杂
* 引入cell架构和inception-like的架构
* Skip connection? 为了解决over-smooth问题，可以尝试让间接相连的节点产生skip connection
* 为何效果不好：
  * The over-smoothing problem arises for nodes that are connected but lie on different sides of the class decision boundary. Due to information exchanging over these edges, stacking multiple attention layers causes excessive smoothing of node features, and makes nodes from different classes become indistinguishable.
* **Important:** Hierachical，先搜索小的cell层面的架构，cell内部可以引入skip connection和inception架构，然后再搜索layer-wise的架构，这部分也可以引入skip connection (<u>adaptive sampling 那篇文章</u>) 和 inception-like的架构 (<u>SIGN那篇文章</u>)



## Progressive Hierarchical Graph Network Architecture Search

### 目前的构思

---

#### Hierarchical的搜索空间

GCN layer部分的搜索空间: 引入不同的layer作为算子



1. Hierarchical: 将搜索空间分割为两个部分，一个是GCN layer 部分，见2，另一个是Cell-wise部分，和GCN NAS那篇文章一样
2. 搜索空间扩大，将GCN layer分成massage mapper, aggregator, updater 三个部分，然后generalize表示
3. Progressive: 先搜索带有1个layer的架构，然后逐渐加入layer，可以设置layer上限
4. 先固定两层的Layer架构用来搜索layer，然后再引入多样化的架构
5. 重要：对于progressive search，不用和原文章一样用predictor来给模型效果预测和排序，而是固定已经搜索过的部分，只搜索新的扩展的部分，这样一方面可以缩小搜索空间，同时也可以继承之前更新过的q表(action的数量不变)

---

### 引入的layer

|  Layer  | Origin | Map  | Aggr |  Up  |
| :-----: | :----: | :--: | :--: | :--: |
|   GCN   |        |      |      |      |
|   GAT   |        |      |      |      |
| ChebNet |        |      |      |      |
|  APPNP  |        |      |      |      |
|  GCNII  |        |      |      |      |
|         |        |      |      |      |
|         |        |      |      |      |



---

 ### TODO

* 给出搜索空间
* 给出图卷积的formulation
* 如何progressive， 搜索算法框架
* 具体的搜索算法
* 搜索出来的架构

### Scripts

> 这部分只是为了快速记录文章的一些叙述写的，真正要写入到文章中还需要好好精炼一下文字

* Initial residual connection and Identity mapping: APPNP proposed a skip connection of the initial input and use $\alpha$ to control the weight, which can be expressed as [x]. And GCNII enabled GCN to express a K order polunomial filter by further introducing a identity mapping from ResNet to [x]  as [y]
* Attention Mechanism: GAT presented a efficient graph attention mechanism that allow for assigning different weight to different adjacent nodes based on the nodes' features, thus could increase the model capability.



### 前人工作的不足之处

* 之前的GraphNAS工作只有用到 GraphSage的几个聚合函数
* Former works mainly focuses on **small graphs** which only contains thousands of nodes with a relatively **low diameter**, where each node could reach any other node within a few hops, such as Cora, PubMed.



---



### 一些知识点

* **Graph sampling** could 
  * <u>serve as a graph edge-wise dropout</u>, which regularises the model thus help the performance. 
  * <u>reduce the bottleneck problem</u> and the resulting “over-squashing” phenomenon that stems from the exponential expansion of the neighbourhood.

* **Simple fixed aggregators** (such as GCN) were shown to often outperform in many cases more complex ones, such as GAT or MPNN (from [SIGN](https://towardsdatascience.com/simple-scalable-graph-neural-networks-7eb04f366d07))



---



### 一些研究点

* **Scalability**: when dealing with very <u>large graph data</u> which automatically rules out the traditional methods which is primarily designed for small graphs. (e.g. concerned with latency, efficiency and so on). 
  * Many current works such as GCN and ChebNet, MoNet and GAT are trained using **full-batch** gradient descent
  * The <u>first work to tackle the scalability problem</u> is **GraphSAGE** which uses neighbourhood sampling combined with mini-batch training to train GNNs on large graphs
  * Some recent works such as  ClusterGCN and GraphSAINT aim to take the approach of **graph-sampling** *(for each batch, a subgraph of the original graph is sampled, and a full GCN-like model is run on the entire subgraph)*  instead of neighbourhood-sampling of GraphSAGE.
* **Layer depth**: [Do we need deep graph neural networks?](https://towardsdatascience.com/do-we-need-deep-graph-neural-networks-be62d3ec5c59)
  * [Simplifying graph neural networks](https://arxiv.org/abs/1902.07153) argues that a GCN model with a single multi-hop diffusion layer can perform on par with models with multiple layers.

* **Transductive VS. Inductive**
  * **Transductive setting**, which assumes that the <u>entire graph is known</u>, and thus the same graph is used during training and testing (albeit different nodes are used for training and testing)
  * **Inductive setting**, in which training and testing are performed on different graphs.