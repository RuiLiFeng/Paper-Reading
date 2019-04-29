# Deep Learning On Graph Survey

## 概论
GNN的发展有两条主线，一条是试图从signal process的角度，依据数学推导，将CNN推广到一般结构的数据上，被称为spectral GCN，另一条是设计能适应各种复杂图结构并具备优秀性能的locally connected network，被称为spatial GCN。

第一条路线起源于2013年IEEE signal processing magazine上的文章：The emerging filed of signal processing on graphs[1]。这篇文章从信号处理的视角将卷积、平移、扩张和缩放等操作推广到了无向图上，直接成为了后续spectral network和chebnet的理论基础。Spectral network是第一篇实现了GCN的论文，发表于2014年NIPS，其主要内容是对[1]中理论的直接应用。为了解决Spectral network缺乏局部性（卷积核对整个图上任意位置信号变化都有响应），以及需要计算图laplace特征分解（对于大规模图意味着超大的计算量），Defferaard等于NIPS2016提出了chebnet，应用[1]中K邻域locally connected network等价于K次多项式卷积，以及chebyshev多项式递归性质，chebnet缩小了卷积核的响应区间至中心点的K邻域，并且规避了对laplace矩阵的特征值分解，并且第一次将图上的pooling操作应用到GCN中来。ICLR2017提出了1stchebnet，将chebnet在1邻域中展开，并重规范化邻接矩阵，获得了spectral GCN上到目前最好的效果，并将多通道多卷积核滤波总结为形式$\hat{A}XW$，其中$\hat{A}$是重规范化后的邻接矩阵。之后在ICLR2018，FASTGCN使用importance sampling对顶点加权，对spectral GCN做了进一步补充。

第二条路线则占据了GCN领域尤其是应用上的大部分内容，这类网络经常受到receptive field思想的影响，设计时希望利用中心点的领域来更新中心点的值。这一条路线又进一步的分为两大方法论：
- 第一种方法论可以追溯到2009年最初始的GNN，其将GNN视为一个动力系统，限制GNN为一个压缩映射，将GNN反复迭代后的不动点视为GNN的输出。这一方案存在两大问题，第一是重复执行迭代操作计算量庞大；第二是限制GNN为压缩映射带来诸多不便。为了解决这两点Gated GNN使用GRU代替迭代过程，将迭代步数缩减为固定的GRU层数，这一做法同时减少了迭代次数并移除了压缩映射的限制，缺点是内存开销巨大，尤其是在面对大规模图数据的时候。更进一步的对效率的改进是Stochastic Steady-state Embedding，其随机并异步的更新各个节点的状态，使用一种类似动量的方法来使节点状态最终趋于稳定。
- -第二种方法论则见于Message Passing NN与GraphSage，其在每一次层都使用不同的GCN，隐层状态经给全部不同的GCN处理后得到最终的状态。

以上两种方法论的共同缺点在于对中间状态存储的巨大内存开销与计算时必须跑遍所有节点，使得训练过程非常低效，尽管作者们开发了诸如子图训练（GraphSage），异步随机训练（SSE）等训练技巧，如何在图结构上高效训练神经网络还是一个问题。

总的来说，spectral GCN和spatial GCN各有优劣
- spectral GCN优势：有坚实的数学理论，可以充分借鉴信号处理和CNN领域的思想，移花接木；
- spectral GCN劣势：扩展性差，训练集中所有图的结构必须都相同，无法扩展的新的图结构上；必须一次处理图上所有节点，在应对大规模图结构时效率偏低；无法处理有向图等其他类型图结构；
- spatial GCN优势：物理含义明显，易于设计和变化；扩展性强于spectral GCN，但仍然有一定问题；可以将深度学习中的许多内容直接的移植到图网络中；
- spatial GCN劣势：获取隐层状态的计算开销过大；可解释性差；

综合来看，在图网络领域研究中，亟待解决的有以下问题：
1.增加网络的可扩展性，让网络可以同时适用于不同结构的图；
2.增加训练的计算效率；
3.如何使图网络应用于动态图结构、有向图结构等更多更复杂的图结构中；
4.研究表明深度图网络会使得信号逐渐光滑直至最终退化，如何增加图网络的深度以提升性能？



