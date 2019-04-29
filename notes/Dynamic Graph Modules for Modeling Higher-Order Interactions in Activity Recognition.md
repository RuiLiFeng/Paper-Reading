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
- 在同一帧的各个结点之间，结点之间的关系作为信息在结点之间交换，构成一个结点图（visual graph）
- 在时间维度上，不同帧中的结点图（visual graph）如同LSTM中的各隐层一样交换上下文语义信息，形成一个动态图（dynamic graph）


## Visual Graph
假设RPN提取出的top-N proposal为$B^t=\{b_1^t,...,b^t_N\}$, t表示帧。构建一个动态的全链接图，$\mathcal{G}=(\mathcal{X},\mathcal{E}),\mathcal{X}=\{x_1,...,x_M\},\mathcal{E}=\{E(x_m,x_k)\}$. 每个结点拥有两个属性，特征与位置（记录左上与右下角坐标），用以捕捉在每一帧中合适的region proposal。
- 初始时，t=1，使用proposal region的平均feature来初始化所有的结点特征
- 在第t步region proposal与node的相关性度量如下：$F_v(b^t_n,x_m)=softmax(h(b^t_n)^Tg(x_m))$,而后更新结点m的特征为：$x_m=x_m+ReLU(\Sum_{i=1}^NF_v(b^t_n,x_m)h(b^t_n)$
