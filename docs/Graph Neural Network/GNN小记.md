# GNN学习笔记
> 本文用来记录一些目前搜集和想到的GNN 图神经网络的一些idea和知识, 比较全面的指导书籍是[William L. Hamilton](https://cs.mcgill.ca/~wlh)的[Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/)
>
> #### 一些资源
>
> - [GNN推荐阅读论文分类汇总Github](https://github.com/thunlp/GNNPapers)
>
> #### 一些文章
>
> - [如何理解GCN](https://www.zhihu.com/question/54504471/answer/332657604) 注：[另一个高赞回答](https://www.zhihu.com/question/54504471/answer/630639025)对拉普拉斯算子和图卷积的联系做了很生动简洁的说明
> - 

## Spectral Based ConvGNN

> 学习这个部分需要对Linear Algebra 有一定的掌握，包括[矩阵特征值](https://www.zhihu.com/question/21874816/answer/181864044)等

基于**Spectral**的图卷积神经网络，主要思想是利用信号变换理论，利用**拉普拉斯矩阵（Laplacian Matrix）**进行图的卷积变换，拉普拉斯矩阵主要有以下三种定义. *(这里参考Hamilton书中定义)*：

$D$是图中各个定点的度的矩阵，$A$是邻接矩阵，对无向图来说是对称矩阵

1. **$L = D - A$**  Unnormalized Laplacian 
2. **$L^{sys} = D^{-\frac{1}{2}}LD^{-\frac{1}{2}}$**   Symmetric normalized Laplacian
3. **$L^{rw} = D^{-1}L$**   Random walk normalized Laplacian



## Dynamic GNN

> the graphs are dinamic which keeps changing every seconds with edges or nodes created or deleted.(e.t. twitter Graph) -> graphs could be considered as a stream of pairwise events



## idea小集

1. 广义上来讲任何数据在赋范空间内都可以建立拓扑关联，[谱聚类](https://www.cnblogs.com/pinard/p/6221564.html)就是应用了这样的思想，所以说拓扑连接是一种广义的数据结构，GCN有很大的应用空间。
2. <u>Spectral Domain</u>的GNN希望借助[图谱的理论](https://en.wikipedia.org/wiki/Spectral_graph_theory)来实现图数据上的卷积操作，也就是借助**图的拉普拉斯矩阵的特征值和特征向量来研究图的性质**



## 论文阅读

> 这个部分是一些阅读的经典论文的主体思路和介绍



### GraphSAGE

GraphSAGE是一篇经典的探究graph scalability来做batch trainning的论文

#### Main Idea

The main idea is that in order to compute the training loss on a single node with an *L*-layer GCN, only the *L*-hop neighbours of that node are necessary, as nodes further away in the graph are not involved in the computation. 

The problem is that, for graphs of the “[small-world](https://en.wikipedia.org/wiki/Small-world_network#:~:text=A small-world network is,number of hops or steps.)” type, such as social networks, the 2-hop neighbourhood of some nodes may already contain millions of nodes, making it too big to be stored in memory [2]. GraphSAGE <u>tackles this problem</u> by sampling the neighbours up to the *L*-th hop: starting from the training node, it samples uniformly with replacement[1] a fixed number *k* of 1-hop neighbours, then for each of these neighbours it again samples *k* neighbours, and so on for *L* times. In this way, for every node we are guaranteed to have a bounded *L*-hop sampled neighbourhood of 𝒪(*kᴸ*) nodes. If we then construct a batch with *b* training nodes, each with its own independent *L*-hop neighbourhood, we get to a memory complexity of 𝒪(*bkᴸ*) independent of the graph size *n*. The computational complexity of one batch of GraphSAGE is 𝒪(*bLd*²*kᴸ*).

> [1]Sampling with replacement means that some neighbour nodes can appear more than once, **in particular if the number of neighbours is smaller than *k*.**
>
> [2]The number of neighbours in such graphs tends to grow exponentially with the neighbourhood expansion.

#### Drawbacks of GraphSAGE

* A notable drawback of GraphSAGE is that sampled nodes might appear multiple times, thus potentially introducing a lot of redundant computation.

