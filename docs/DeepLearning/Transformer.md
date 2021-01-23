# Transformer 相关
> Transformer 是Google团队在2017年提出的一个解决Word Embedding的模型，其摒弃了RNN结构，完全由Attention Model构成
## Transformer
> 原始论文: [Attention Is All You Need
](https://arxiv.org/abs/1706.03762)
> refer: [transformer blog](https://jalammar.github.io/illustrated-transformer/)

Transformer由Encoder 和 Decoder两个部分组成，其基本架构如下：
<p align="center"><img src="/docs/DeepLearning/Pics/Transformer.png" alt="Lenet.png" style="zoom:40%;" />

左边的部分是Eecoder，右边的那个长得比较高的是Decoder，下面分别展开说明

### Decoder

其中，左边的灰色区域代表了一个**identical layer**，由两个sub-layer组成，根据数据通过的次序分别为:
1. **self-attention mechanism**: 
   <p align="center"><img src="/docs/DeepLearning/Pics/TAtt.png" alt="Lenet.png" style="zoom:40%;" />

   如上图所示，其中左边的 Scaled Dot-Product Attention可以写作

   $$
   Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
   $$

   其中 Q, K, V 三个矩阵由输入的embedding乘以三个不同的权重矩阵得到
   > Pic refer [transformer blog](https://jalammar.github.io/illustrated-transformer/)
   <p align="center"><img src="/docs/DeepLearning/Pics/self-attention-matrix-calculation.png" alt="Lenet.png" style="zoom:40%;" />

   如果拆分来看，Scaled Dot-Product Attention 实际上做了下面的事情
   <p align="center"><img src="/docs/DeepLearning/Pics/self-attention-output.png" alt="Lenet.png" style="zoom:60%;" />

   概括就是：输入的embedding乘以权重矩阵得到的Q和K矩阵（上面 q1 对应了Q矩阵的第一行, 其他类似）通过 点乘 来计算attention的score，然后softmax归一化输出得到权重，权重乘以 value 向量得到输出的 embedding

   作者在这个基础上加入了multi-attention机制，从而可以更好地捕获到各个position的信息，即:
   <p align="center"><img src="/docs/DeepLearning/Pics/transformer_multi-headed_self-attention-recap.png" alt="Lenet.png" style="zoom:60%;" />


   
2. **fully connected feed-forward network**: 
   这层主要是提供非线性变换，每个位置的输出都输入到同一个 Feed Forward 网络中（变换的参数是一样的），如下
   $$
   FFN(x) = max(0, xW_1+b_1)W_2+b_2
   $$

上面的两个sub-layer都采用了residual connection架构，经过residual的数据与sub-layer输出的数据进行 add 后再 Normalize，可以表示为：

$$
output_{sub\_layer} = LayerNorm(x+Sublayer(x))
$$

Encoder部分堆叠了 6 个这样的**identical layer**，因此上面图中的 N = 6

### Decoder

decoder和encoder类似，变化如下：
* 输出：对应的 i 位置的词的概率分布（整个词典各个词的概率）
* 输入：上一时刻decoder的输出
* 中间多了Attention Layer用来接受来自Encoder的Q和K（注意观察图中只有两个箭头，对应了Q和K，V用的是Masked Attention的输出），同时第一个Attention Layer加入了Mask机制
* 之所以要mask，是为了防止i位置predict时可以潜在地看到i和后面的内容

注意，Encoder可以并行计算，但Decoder不是一次把所有序列解出来的，而是像rnn一样一个一个解出来的，因为要用上一个位置的输入当作attention的query

### Positional Encoding
 <p align="center"><img src="/docs/DeepLearning/Pics/PE.png" alt="Lenet.png" style="zoom:50%;" />

即，输入到第一个encoder的embedding是原始的embedding和positional encoding求 element-wise add 得到的
