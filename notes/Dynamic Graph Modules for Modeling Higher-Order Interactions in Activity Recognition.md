# 概要
为了进行视频动作识别，有必要考虑以下两点：
- 同一帧中对象（object）之间的关系
- 不同帧之间对象的关系
现有的2D ConVnet+LSTM或3D ConvNet或是无法处理对象之间的关系，或是无法处理流动的视频。

本文用动态的图模型来建模对象之间的交互

本文主要的变量有两个：
- 各帧中的region proposal，由Region Proposal Network(RPN)划分区域（通过记录左上与右下角坐标），RoIQAlign提取特征
- 动态隐式图，图的结构（点的个数，边全链接）事先指定，图中的结点（node）将会用于学习一些特定的动作特征，类似于CNN中的卷积核，LSTM中的unit状态，边的权重与结点中的值会随各帧信息的流入而变化

信息将在以下两个维度上流动和汇聚：
- 在同一帧的各个结点之间，结点之间的关系与结点特征作为信息在结点之间交换，构成一个结点图（visual/location graph）
- 在时间维度上，不同帧中的结点图（visual/location graph）如同LSTM中的各隐层一样交换上下文语义信息，形成一个动态图（dynamic graph）

## Graph
### Visual Graph
为了捕捉视觉上相似的proposal在连续时间上的关系，本文构建了一个visual graph
假设RPN提取出的top-N proposal为$B^t=\{b_1^t,...,b^t_N\}$, t表示帧。构建一个动态的全链接图，$\mathcal{G}=(\mathcal{X},\mathcal{E}),\mathcal{X}=\{x_1,...,x_M\},\mathcal{E}=\{E(x_m,x_k)\}$. 每个结点拥有两个属性，特征与位置（记录左上与右下角坐标），用以捕捉在每一帧中合适的region proposal。
- 初始时，t=1，使用proposal region的平均feature来初始化所有的结点特征
- 在第t步region proposal与node的相关性度量如下：$F_v(b^t_n,x_m)=softmax(h(b^t_n)^Tg(x_m))$,而后更新结点m的特征为：$x_m=x_m+ReLU(\sum_{i=1}^NF_v(b^t_n,x_m)h(b^t_n)$, 可以看作是图结构的LSTM，由proposal region与node状态的相似程度决定输入强度
- 当t帧的所有信息流入动态图，结点更新完毕以后，使用结点之间的相似性更新边$E_v(x_k,x_m)=\psi(x_k)^T\psi(x_m)$.

### Location Graph
为了捕捉位置上相近的proposal在连续时间上的关系，本文构建了一个location graph
与visual graph的不同之处在于
- location graph中更新结点坐标（t=1时用平均坐标初始化，之后以位置上相邻度为权，作为全图坐标的加权平均）而visual graph不需要结点坐标信息
- 结点状态的输入门由位置上的相邻度（Intersection-Over-Union）决定而非视觉上的相似度

## Graph Attention
