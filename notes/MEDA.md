# MEDA
@(Domain Adaptation)

## Introduction
本领域传统方法主要集中于两个思路上：instance reweighting与feature matching。本文主要关注feature matching。

在feature matching这个思路下，存在两个主要问题：
- degenerated feature transformation：子空间学习与分布对齐只能减少但不能消除分布差异，
-- 子空间学习进行子空间迁移获得更好的特征表示，但无法消除分布差异
-- 特征对齐只在原本的特征空间减少分布距离，反而使得特征失真

- Unevaluated distribution alignment：现有工作对齐边缘分布与条件分布时使用相同的权重，而不考虑他们对于实际任务的不同重要性

Tips：为什么边缘分布与条件分布重要性不相同？
当迁移任务发生在条件分布已经比较对齐的场景（数据集A中猫比较多，数据集B中狗比较多，但来源比较统一），和迁移学习发生在边缘分布已经比较对齐（用同样的方式在不同环境下拍摄猫和狗），但条件分布不对齐的情况，这两者看来不同分布的对齐重要性是不一致的。但是比较trick的地方在于，对齐程度高这件事本身已经弱化了相应分布对齐的重要性，再拉出来强调一番必要何在？

## Method
###Problem setting
Source domain $\mathcal{D}_s=\{x_{s_i},y_{s_i}\}_{i=1}^n$, target domain $\mathcal{D}_t=\{x_{t_j}\}_{j=n+1}^{n+m}$。这里假定特征空间一致，即$\mathcal{X}_s=\mathcal{X}_t$，标签空间一致$\mathcal{Y}_s=\mathcal{Y}_t$，但是边缘分布$P_s(x_s)\neq P_t(x_t)$，条件分布$Q_s(y_s|x_s)\neq Q_t(y_t|x_t)$。任务目标在于学习一个domain invariant的分类函数$f$。


### Manifold feature learning
主要为了针对degenerated feature transformation
令$g$为流形映射，将$x$映射至特征空间，遵循结构风险最小化指导，$f$：
$$f=\arg\min_{f\in\sum_{i=1}^n\mathcal{H}_K}l(f(g(x)),y_i)+\eta||f||_K^2+\lambda\overline{D_f}(\mathcal{D}_s,\mathcal{D}_t)+\rho R_f(\mathcal{D}_s,\mathcal{D}_t)$$
这里$\overline{D}_f$是本文提出的动态分布对齐，$R_f$是Laplace正则，目的是进一步利用流形上的邻近点的局部几何结构。

Tips：这里介绍一下Grassman流形的基本情况
Grassman流形主要是为了应对数据是线性子空间而非向量的情况（比如关于一个人的很多副照片，构成一个数据），传统的特征学习，在一个欧式的空间使用了一个非欧式的度量，存在着不一致性，本文将空间，度量，分类全部在grassman流形上完成

Definition 1 Grassman manifold $\mathcal{G}(m,D)$是m维的$\mathbb{R}^D$线性子空间的集合。

从而，$\mathcal{G}(m,D)$中的任何一个元素可以被表示为m个D维的正交向量组成的矩阵$Y\in\mathbb{R}^{D\times m}$，$Y^TY=I_m$. 但是显然这个表示不是唯一的，如果把两个存在线性等价关系的矩阵$Y_1R=Y_2,R\in\mathbb{R}^{m\times m}$视为一个等价类的元素，$\mathcal{G}(m,D)$实际上是商空间$\mathcal{O}(D)/\mathcal{O}(m)\times\mathcal{O}(D-m)$。（作为练习，试证明之）

Definition 2 $Y_1,Y_2\in \mathbb{R}^{D\times m}$,所有列正交，他们所代表等价类的principal angle $0\leq\theta_1\leq...\leq\theta_m\leq\pi/2$定义为：
$$cos\theta_k=\max_{u_k\in span(Y_1),v_k\in span(Y_2)}u_k'v_k$$
$$s.t. u_k'u_k=1,v_k'v_k=1$$
$$u_k'u_i=0,v_k'v_i=0,i=1,...,k-1$$

这里其实已经可以循着GCN的思路来推导了，首先这里的距离函数定义在两个子空间中，实际上是在度量两个子空间的相似性，度量可以写为：$x^TY_1^TY_2y,\forall x,y\in\mathbb{R}^m$，之后的条件与特征值特征向量的约束如出一辙。$Y_1^TY_2$在这里就是Laplace矩阵，我们下面来证明这个问题的解是特征值：
Lemma 1: For any $A\in\mathbb{R}^{m\times m}$,  $|\lambda|_{max}=\max_{||x||=1,||y||=1}x^TAy$.
proof:$x^TAy=x^TU^T\Sigma Vy=\hat{x}^T\Sigma\hat{y}=\sum_{i=1}^m\lambda_i\hat{x}_i\hat{y}_i\leq|\lambda_{max}|.$ The equality is taken when $j=\arg\max_{i=1,...,m}|\lambda_i|,|\hat{x}_j|,|\hat{y}_j|=1$, that is $Ux=(0,...,1,0,..,0)^T,Vy=(0,...,1,0,...,0)^T$, which means x and y are the j-th row of U,V correspondingly.