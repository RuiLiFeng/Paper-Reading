# 概要
为了进行视频动作识别，有必要考虑以下两点：
- 同一帧中对象（object）之间的关系
- 不同帧之间对象的关系
现有的2D ConVnet+LSTM或3D ConvNet或是无法处理对象之间的关系，或是无法处理流动的视频。

本文用动态的图模型来建模对象之间的交互

视频记为$V=\{f_1,...,f_T\}$, $f_t$表示一个由2D或3D ConvNet抽取的第t帧的feature map
使用Region Proposal Network来提取当前帧可能的object，
