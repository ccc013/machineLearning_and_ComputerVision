
> 2019年第 84 篇文章，总第 108 篇文章

AI知识点（AI Knowledge）系列第一篇文章--激活函数。

本文主要的目录如下：

- 激活函数的定义
- 为什么需要激活函数
- 常见的激活函数


------

### 1. 激活函数的定义

激活函数是神经网络中非常重要的一个内容，神经网络是受到生物神经网络的启发，在生物神经网络中也存在着激活函数，而且激活函数决定了神经元之间是否要传递信号，而在人工的神经网络中，**激活函数的作用则主要是给网络添加非线性因素，使得网络可以逼近任意复杂的函数**，一个简单的神经元如下图所说，其中 `f` 表示的就是激活函数。

![](https://github.com/ccc013/AI_Knowledge/raw/master/images/activation_functions/activation_function.jpg)

如果是从数学上来定义的话，在ICML2016的一篇论文 [Noisy Activation Functions](https://arxiv.org/pdf/1603.00391v3.pdf)中，作者将激活函数定义为一个几乎处处可微的 h : R → R 。

另外还有以下的一些概念：
**a.饱和**
当一个激活函数h(x)满足
$$
lim_{𝑛→+∞}ℎ′(𝑥)=0
$$
时我们称之为**右饱和**。



当一个激活函数h(x)满足
$$
lim_{𝑛→−∞}ℎ′(𝑥)=0
$$
时我们称之为**左饱和**。

当一个激活函数既满足左饱和又满足右饱和时，我们称之为**饱和**。

**b.硬饱和与软饱和**
对任意的 x，如果存在常数 c，当`𝑥>𝑐`时恒有 `ℎ′(𝑥)=0` 则称其为**右硬饱和**，当`x<c`时恒 有`ℎ′(𝑥)=0` 则称其为**左硬饱和**。

若既满足左硬饱和，又满足右硬饱和，则称这种激活函数为**硬饱和**。

但如果只有**在极限状态下偏导数等于0**的函数，称之为**软饱和**。



------

### 2. 为什么需要激活函数

事实上这个问题应该是问为什么需要非线性的激活函数，因为目前常用的都是非线性激活函数，原因如下：

1. 激活函数是可以讲当前特征空间通过一定的线性映射转换到另一个空间，让数据能够更好的被分类；
2. 但神经网络中，基本的函数 `y=wx+b` 就是一个线性函数，如果激活函数还是线性函数，那么线性的组合还是线性，和单独一个线性分类器没有差别，是无法逼近任意函数，而实际生活中的很多问题都不是简单的线性分类问题；
3. 非线性激活函数可以引入非线性因素，让神经网络具有更强大的能力，可以真正做到逼近和表示任意的复杂的函数，解决更多复杂的问题，可以学习解决很多类型的问题，比如图像、文本、视频、音频等，这也是现在深度学习被全面应用到很多传统的机器学习问题的基础之一 



------

### 3. 常见的激活函数

#### 3.1 常见的激活函数定义和其图像

##### 3.1.1 Sigmoid 激活函数

函数的公式定义, 其值域是 （0, 1)
$$
f(x) = \frac{1}{1+e^{-x}}
$$
![](https://github.com/ccc013/AI_Knowledge/raw/master/images/activation_functions/sigmoid.jpg)

导数为：
$$
\sigma′(x)=\sigma(x)(1-\sigma(x))=\frac{e^{-x}}{(e^{-x}+1)^2}
$$
**优点**：

1. `Sigmoid`函数的输出映射在 (0,1) 之间，单调连续，输出范围有限，优化稳定，可以用作输出层。
2. 求导容易。

**缺点**：

1. **梯度消失**：注意：`Sigmoid` 函数趋近 0 和 1 的时候变化率会变得平坦，也就是说，`Sigmoid` 的梯度趋近于 0。神经网络使用 `Sigmoid` 激活函数进行反向传播时，**输出接近 0 或 1 的神经元其梯度趋近于 0**。这些神经元叫作**饱和神经元**。因此，这些神经元的权重不会更新。此外，与此类神经元相连的神经元的权重也更新得很慢。该问题叫作梯度消失。因此，想象一下，如果一个大型神经网络包含 `Sigmoid` 神经元，而其中很多个都处于饱和状态，那么该网络无法执行反向传播。

2. **不以零为中心**：`Sigmoid` 输出不以零为中心的，值域是 0-1

3. **计算成本高昂**：`exp()` 函数与其他非线性激活函数相比，计算成本高昂。

因为不以 0 为中心，接下来介绍的就是以 0 为中心的一个激活函数。

##### 3.1.2 Tanh 激活函数

公式定义如下所示，值域是 （-1, 1）
$$
f(x) = tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$
其图像如下所示：

![](https://github.com/ccc013/AI_Knowledge/raw/master/images/activation_functions/tanh.jpg)

导数为：
$$
f^{'}(x)=-(tanh(x))^2
$$


在实践中，`Tanh` 函数的使用优先性高于 `Sigmoid` 函数

**优点：**
1.比 `Sigmoid`函数收敛速度更快。
2.相比 `Sigmoid` 函数，其输出以0为中心。
**缺点：**

由于饱和性产生的梯度消失问题依然存在。

`Sigmoid` 和`Tanh` 都存在梯度消失问题，所以又有新的一个激活函数，这个激活函数变得非常常用，并且有了更堵哦的改进版。

##### 3.1.3 ReLU 激活函数

函数的公式定义如下所示，值域是 [0, +∞)
$$
f(x) = max(0, x)
$$
其图像如下所示：

![](https://github.com/ccc013/AI_Knowledge/raw/master/images/activation_functions/relu.jpg)

导数是：
$$
c(u)=
\begin{cases} 
0,x \le 0 \\ 
1,x \gt 0 
\end{cases}
$$


`ReLU` 激活函数，即修正线性单元函数，相比前两个激活函数，由于其特点使得它变成目前最常用的激活函数。

**优点**

1. 收敛速度更快；
2. 相比 `sigmoid` 和 `tanh`，计算速度更快
3. 有效缓解了梯度消失的问题
4. 在没有无监督与训练的时候，也能有较好的表现
5. 提供了神经网络的稀疏表达能力

**缺点**

1. **不以零为中心**：和 Sigmoid 激活函数类似，ReLU 函数的输出不以零为中心。

2. 前向传导（forward pass）过程中，**如果 x < 0，则神经元保持非激活状态，且在后向传导（backward pass）中「杀死」梯度。这样权重无法得到更新，网络无法学习。**当 x = 0 时，该点的梯度未定义，但是这个问题在实现中得到了解决，通过采用左侧或右侧的梯度的方式。

`relu` 激活函数实际上在小于0的一边，还是存在梯度消失问题，所以有了几个改进版本的函数。

##### 3.1.4 LReLU、PReLU和RReLU

**LReLU**

第一个改进版的 `relu` 函数，即 leaky relu，LReLU，函数定义如下所示：
$$
f(x) = max(ax, x), 0<a<1
$$
其图像如下所示，图中 `a=0.1` ，

![](https://github.com/ccc013/AI_Knowledge/raw/master/images/activation_functions/leaky_relu.png)

导数如下：
$$
LReLU′(x)=
\begin{cases}
1 & if x \gt 0 \\
a & if x \le 0
\end{cases}
$$


Leaky ReLU 的概念是：当 x < 0 时，它得到 0.1 的正梯度。

该函数一定程度上缓解了 dead ReLU 问题，但是使用该函数的结果并不连贯。尽管它具备 ReLU 激活函数的所有特征，如计算高效、快速收敛、在正区域内不会饱和。

**PReLU**

Leaky ReLU 可以得到更多扩展。不让 x 乘常数项，而是让 x 乘超参数，这看起来比 Leaky ReLU 效果要好。该扩展就是 Parametric ReLU，也就是 PReLU，这是 LReLU 的改进。

其公式定义和 LReLU 一样，但这里的参数 `a` 是一个超参数，不需要自定义，而是自适应从数据中学习，也就是可以进行反向传播，这让神经元可以选择负区域最好的梯度

**RReLU**

公式定义如下所示：
$$
y_{ji}=\begin{cases}
x_{ji}& if(x_{ji}>0)\\
a_{ji}x_{ji}& if(x_{ji}\le0)
\end{cases}
$$

$$
a_{ji} \sim U(l,u),l<u\;\;and\;\;l,u\in [0,1)
$$

这里的超参数 $a_{ji}$ 就是给定范围内取样的随机变量，但在测试中是固定的，该激活函数在一定程度上可以起到正则效果。

在论文《Empirical Evaluation of Rectified Activations in Convolution Network》，作者对比了`ReLU`激活函数和其三个改进版本 `LReLU`、`PReLU`、`RReLU` 在数据集 `CIFAR-10`、`CIFAR-100`、`NDSB` 中相同网络模型的性能。

想了解的可以具体看看这篇论文，当然在实际应用中，初次训练可以选择 `ReLU` 激活函数，然后可以再尝试这三个改进版来看看对比的结果

##### 3.1.5 ELU

公式定义：
$$
f(x)=\begin{cases}
a(e^x-1)& if(x<0)\\
x& if(0\le x)
\end{cases}
$$
导数如下：
$$
ELU′(x)=
\begin{cases}
1 & if x \gt 0 \\
ELU(x) + a & if x \le 0
\end{cases}
$$


**优点**：

1.ELU减少了正常梯度与单位自然梯度之间的差距，从而加快了学习。
2.在负的限制条件下能够更有鲁棒性。

**缺点**

1. 由于包含指数运算，所以计算时间更长；
2. 无法避免梯度爆炸问题；
3. 神经网络不学习 α 值。



ELU 激活函数可以参考 ICLR 2016的论文《FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)》。



##### 3.1.6 Swish

这个函数也叫做自门控激活函数，来自2017年谷歌的论文《Swish: a Self-Gated Activation Function》：https://arxiv.org/abs/1710.05941v1

公式定义如下所示
$$
\sigma(x)=\frac{x}{1+e^{-x}}
$$
其图像如下所示：

![](https://github.com/ccc013/AI_Knowledge/raw/master/images/activation_functions/swish.png)

从图中可以看到，在 x<0 的部分和 ReLU 激活函数是不同的，会有一段 x 增大，但输出值下降的区域；

所以 `Swish` 函数具备单侧有界的特性，它是平滑、非单调的。

具体的介绍可以参考文章--[Swish 激活函数的性能优于 ReLU 函数。](http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650732184&idx=1&sn=7e7ded430f5884d6d099980267fcfb15&chksm=871b32e6b06cbbf07c133e826351bae045858699d72f65474c7cebcbc5b2762c3522dfc62ef7&scene=21#wechat_redirect)



##### 3.1.7 Softplus 和 Softsign

`Softplus` 的公式定义：
$$
f(x)=log(e^x+1)
$$
`Softsign` 的公式定义：
$$
f(x)=\frac{x}{|x|+1}
$$
这两个激活函数的使用较少，可以参考 Tensorflow 提供的 api--[激活函数相关TensorFlow的官方文档](https://www.tensorflow.org/api_docs/python/nn/activation_functions_#top_of_page)

##### 3.1.8 Softmax(归一化指数函数)

公式定义：
$$
\sigma(z)j = \frac{e^{z_j}}{\sum{k=1}^K e^{z_k}} 
$$
该函数主要用于多分类神经网络的输出层。

##### 3.1.9 SELU 和 GELU

这两个激活函数实际上都是最近提出或者最近才使用较多的激活函数，详细介绍可以查看文章--[从ReLU到GELU，一文概览神经网络的激活函数](https://mp.weixin.qq.com/s/np_QPpaBS63CXzbWBiXq5Q)，这里简单给出公式定义和优缺点。

SELU，即扩展型指数线性单元激活函数，其公式如下所示：
$$
SELU(x)=\lambda
\begin{cases}
x & if x \gt 0 \\
ae^x-a & if x \le 0
\end{cases}
$$
**优点**：

- 内部归一化的速度比外部归一化快，这意味着网络能更快收敛；
- 不可能出现梯度消失或爆炸问题，见 SELU 论文附录的定理 2 和 3。

**缺点**：

- 这个激活函数相对较新——需要更多论文比较性地探索其在 CNN 和 RNN 等架构中应用。
- 这里有一篇使用 SELU 的 CNN 论文：https://arxiv.org/pdf/1905.01338.pdf

GELU，即高斯误差线性单元激活函数是2016年提出的，但在最近的 Transformer 模型（谷歌的 BERT 和 OpenAI 的 GPT-2）中得到了应用，其公式如下所示：
$$
GELU(x)=0.5x(1+tanh(\sqrt{2/\pi})(x+0.044715x^3))
$$
**优点**：

- 似乎是 NLP 领域的当前最佳；尤其在 Transformer 模型中表现最好；
- 能避免梯度消失问题。

**缺点**：

- 尽管是 2016 年提出的，但在实际应用中还是一个相当新颖的激活函数。





#### 3.2 常见的激活函数的性质

1. **非线性**： 当激活函数是线性的，一个两层的神经网络就可以基本上逼近所有的函数。但如果激活函数是恒等激活函数的时候，即 $ f(x)=x $，就不满足这个性质，而且如果 MLP 使用的是恒等激活函数，那么其实整个网络跟单层神经网络是等价的；
2. **可微性**： 当优化方法是基于梯度的时候，就体现了该性质；
3. **单调性**： 当激活函数是单调的时候，单层网络能够保证是凸函数；
4. $ f(x)≈x $： 当激活函数满足这个性质的时候，如果参数的初始化是随机的较小值，那么神经网络的训练将会很高效；如果不满足这个性质，那么就需要详细地去设置初始值；
5. **输出值的范围**： 当激活函数输出值是有限的时候，基于梯度的优化方法会更加稳定，因为特征的表示受有限权值的影响更显著；当激活函数的输出是无限的时候，模型的训练会更加高效，不过在这种情况小，一般需要更小的 Learning Rate。



#### 3.3 如何选择激活函数

通常选择一个激活函数并不容易，需要考虑很多因素，最好的做法还是一一实验所有的激活函数，但需要保证其他因素相同。

一些技巧：（来自深度学习500问第三篇深度学习基础）

1. 如果**输出是 0、1 值（二分类问题）**，则**输出层选择 sigmoid 函数**，然后其它的所有单元都选择 Relu 函数。
2. 如果在隐藏层上不确定使用哪个激活函数，那么**通常会使用 Relu 激活函数**。有时，也会使用 tanh 激活函数，但 Relu 的一个优点是：当是负值的时候，导数等于 0。
3. **Sigmoid 激活函数**：除了输出层是一个二分类问题基本不会用它。
4. **Tanh 激活函数**：tanh 是非常优秀的，几乎适合所有场合。
5. **ReLU 激活函数**：最常用的默认函数，如果不确定用哪个激活函数，就使用 ReLu 或者 Leaky ReLu，再去尝试其他的激活函数。
6. 如果遇到了一些死的神经元，我们可以使用 `Leaky ReLU` 函数。

#### 3.4 激活函数以零为中心的问题

在介绍 `Sigmoid` 的缺点的时候说到它不是以 0 为中心，这个特点主要是影响收敛速度，因为它的输出值是恒为正的，那么在梯度下降，进行参数更新的时候，所有参数每次更新的方向都是同个方向，要不都是正方向，或者要不都是负方向，其更新就是一个 `z` 字形，如下图所示：

![图片来自文章https://liam.page/2018/04/17/zero-centered-active-function/](https://github.com/ccc013/AI_Knowledge/raw/master/images/activation_functions/zig-zag-gradient.png)

借用文章 [cs231n_激活函数](https://blog.csdn.net/weixin_38646522/article/details/79534677) 的例子：

> 假设我们有权值 w=[1,−1,1]]，我们需要将权值更新为 w=[−1,1,−1] ，如果梯度是同时有正和有负的，我们可以只更新一次就可得到结果： w=[1,−1,1]+[−2,2,−2]=[−1,1,−1]；但是如果梯度只能是正或者只能是负，则需要两次更新才能得到结果： w=[1,−1,1]+[−3,−3,−3]+[1,5,1]=[−1,1,−1] 。

所以需要选择以 0 为中心的激活函数，可以提高收敛速度。







------

### 参考文章

1. [The Activation Function in Deep Learning 浅谈深度学习中的激活函数](https://www.cnblogs.com/rgvb178/p/6055213.html)
2. [一文概览深度学习中的激活函数](https://mp.weixin.qq.com/s/kmrz5TaaD_JnufbN7QkpwA)
3. [深度学习500问第三篇深度学习基础--3.4激活函数](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch03_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/%E7%AC%AC%E4%B8%89%E7%AB%A0_%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80.md#341-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E9%9D%9E%E7%BA%BF%E6%80%A7%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0)
4. [Swish 激活函数的性能优于 ReLU 函数](http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650732184&idx=1&sn=7e7ded430f5884d6d099980267fcfb15&chksm=871b32e6b06cbbf07c133e826351bae045858699d72f65474c7cebcbc5b2762c3522dfc62ef7&scene=21#wechat_redirect)
5. 《Empirical Evaluation of Rectified Activations in Convolution Network》：https://arxiv.org/abs/1505.00853
6. 《FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)》：https://arxiv.org/pdf/1511.07289v5.pdf
7. 《Swish: a Self-Gated Activation Function》：https://arxiv.org/abs/1710.05941v1
8. [激活函数相关TensorFlow的官方文档](https://www.tensorflow.org/api_docs/python/nn/activation_functions_#top_of_page)
9. [从ReLU到GELU，一文概览神经网络的激活函数](https://mp.weixin.qq.com/s/np_QPpaBS63CXzbWBiXq5Q)
10. SELU 论文：https://arxiv.org/pdf/1706.02515.pdf
11. GELU 论文：https://arxiv.org/pdf/1606.08415.pdf
12. [谈谈激活函数以零为中心的问题](https://liam.page/2018/04/17/zero-centered-active-function/)
13. [cs231n_激活函数](https://blog.csdn.net/weixin_38646522/article/details/79534677)


---

欢迎关注我的微信公众号--**算法猿的成长**，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_0601.png)

**如果觉得不错，在看、转发就是对小编的一个支持！**


