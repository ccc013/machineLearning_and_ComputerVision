> 2019 年第 60 篇文章，总第 84 篇文章

#### 前言

上次写的文章-- [一文了解下 GANs可以做到的事情](https://mp.weixin.qq.com/s/twDS3pNo_WGm1Ka2_2RuQA)，如果想进一步了解 GAN，学习研究 GAN，可以先从这 10 篇论文开始。

本文翻译自：

https://towardsdatascience.com/must-read-papers-on-gans-b665bbae3317

原文介绍 10 篇介绍 GANs 以及最新进展的论文，跟原文介绍顺序有所不同，我是根据时间顺序，从最开始提出的 GANs 论文到目前最新的来介绍，这十篇分别如下所示：

1. Generative Adversarial Networks，2014
2. Conditional GANs，2014
3. DCGAN，2015
4. Improved Techniques for Training GANs，2016
5. Pix2Pix，2016
6. CycleGAN，2017
7. Progressively Growing of GANs，2017
8. StackGAN，2017
9. BigGAN，2018
10. StyleGAN，2018

原文作者推荐开始的第一篇论文是 DCGAN 。

[TOC]



------

#### 1. Generative Adversarial Networks

论文名称：Generative Adversarial Nets

论文地址: https://arxiv.org/abs/1406.2661

“GAN之父” Ian Goodfellow 发表的第一篇提出 GAN 的论文，这应该是任何开始研究学习 GAN 的都该阅读的一篇论文，它提出了 GAN 这个模型框架，讨论了非饱和的损失函数，然后对于最佳判别器(optimal discriminator)给出其导数，然后进行证明；最后是在 Mnist、TFD、CIFAR-10 数据集上进行了实验。

#### 2. Conditional GANs

论文名称：Conditional Generative Adversarial Nets

论文地址：https://arxiv.org/abs/1411.1784

如果说上一篇 GAN 论文是开始出现 GAN 这个让人觉得眼前一亮的模型框架，这篇 cGAN 就是当前 GAN 模型技术变得这么热门的重要因素之一，事实上 GAN 开始是一个无监督模型，生成器需要的仅仅是随机噪声，但是效果并没有那么好，在 14 年提出，到 16 年之前，其实这方面的研究并不多，真正开始一大堆相关论文发表出来，第一个因素就是 cGAN，第二个因素是等会介绍的 DCGAN；

cGAN 其实是将 GAN 又拉回到**监督学习**领域，如下图所示，它在生成器部分添加了**类别标签这个输入**，通过这个改进，缓和了 GAN 的一大问题--训练不稳定，而这种思想，引入先验知识的做法，在如今大多数非常有名的 GAN 中都采用这种做法，后面介绍的生成图片的 BigGAN，或者是图片转换的 Pix2Pix，都是这种思想，可以说 cGAN 的提出非常关键。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/cGANs.png)

#### 3. DCGAN

论文名称：Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

论文地址：https://arxiv.org/abs/1511.06434

其实原作者推荐第一篇论文应该是阅读这篇 DCGAN 论文，2015年发表的。这是第一次采用 CNN 结构实现 GAN 模型，它介绍如何使用卷积层，并给出一些额外的结构上的指导建议来实现。另外，它还讨论如何可视化 GAN 的特征、隐空间的插值、利用判别器特征训练分类器以及评估结果。下图是 DCGAN 的生成器部分结构示意图

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/DCGAN.png)

#### 4. Improved Techniques for Training GANs

论文名称：Improved Techniques for Training GANs

论文地址：https://arxiv.org/abs/1606.03498

这篇论文的作者之一是 Ian Goodfellow，它介绍了很多如何构建一个 GAN 结构的建议，它可以帮助你理解 GAN 不稳定性的原因，给出很多稳定训练 DCGANs 的建议，比如特征匹配(feature matching)、最小批次判别(minibatch discrimination)、单边标签平滑(one-sided label smoothing)、虚拟批归一化(virtual batch normalization)等等，利用这些建议来实现 DCGAN 模型是一个很好学习了解 GANs 的做法。

#### 5. Pix2Pix

论文名称：Image-to-Image Translation with Conditional Adversarial Networks

论文地址：https://arxiv.org/abs/1611.07004

Pix2Pix 的目标是实现图像转换的应用，如下图所示。这个模型在训练时候需要采用成对的训练数据，并对 GAN 模型采用了不同的配置。其中它应用到了 PatchGAN 这个模型，PatchGAN 对图片的一块 70*70 大小的区域进行观察来判断该图片是真是假，而不需要观察整张图片。

此外，生成器部分使用 U-Net 结构，即结合了 ResNet 网络中的 skip connections 技术，编码器和解码器对应层之间有相互连接，它可以实现如下图所示的转换操作，比如语义图转街景，黑白图片上色，素描图变真实照片等。



![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pix2pix_examples.jpg)

#### 6. CycleGAN

论文名称：Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks 

论文地址：https://arxiv.org/abs/1703.10593

上一篇论文 Pix2Pix 的问题就是训练数据必须成对，即需要原图片和对应转换后的图片，而现实就是这种数据非常难寻找，甚至有的不存在这样一对一的转换数据，因此有了 CycleGAN，仅仅需要准备两个领域的数据集即可，比如说普通马的图片和斑马的图片，但不需要一一对应。这篇论文提出了一个非常好的方法--循环一致性(Cycle-Consistency)损失函数，如下图所示的结构：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/cyclegan_architecture.png)

这种结构在接下来图片转换应用的许多 GAN 论文中都有利用到，cycleGAN 可以实现如下图所示的一些应用，普通马和斑马的转换、风格迁移（照片变油画）、冬夏季节变换等等。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/cycleGAN_examples.jpg)

#### 7. Progressively Growing of GANs

论文名称：Progressive Growing of GANs for Improved Quality, Stability, and Variation

论文地址：https://arxiv.org/abs/1710.10196

这篇论文必读的原因是因为它取得非常好的结果以及对于 GAN 问题的创造性方法。它利用一个多尺度结构，从 `4*4 ` 到 `8*8` 一直提升到 `1024*1024` 的分辨率，如下图所示的结构，这篇论文提出了一些如何解决由于目标图片尺寸导致的不稳定问题。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/progressiveGANs.png)

#### 8. StackGAN

论文名称：StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks 

论文地址：https://arxiv.org/abs/1612.03242

StackGAN 和 cGAN 、 Progressively GANs 两篇论文比较相似，它同样采用了先验知识，以及多尺度方法。整个网络结构如下图所示，第一阶段根据给定文本描述和随机噪声，然后输出 `64*64` 的图片，接着将其作为先验知识，再次生成 `256*256` 大小的图片。相比前面 推荐的 7 篇论文，StackGAN 通过一个文本向量来引入文本信息，并提取一些视觉特征

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/stackGANs.png)



#### 9. BigGAN

论文地址：Large Scale GAN Training for High Fidelity Natural Image Synthesis

论文地址：https://arxiv.org/abs/1809.11096

BigGAN 应该是当前 ImageNet 上图片生成最好的模型了，它的生成结果如下图所示，非常的逼真，但这篇论文比较难在本地电脑上进行复现，它同时结合了很多结构和技术，包括自注意机制(Self-Attention)、谱归一化(Spectral Normalization)等，这些在论文都有很好的介绍和说明。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/BigGAN.png)



#### 10. StyleGAN

论文地址：A Style-Based Generator Architecture for Generative Adversarial Networks

论文地址：https://arxiv.org/abs/1812.04948

StyleGAN 借鉴了如 Adaptive Instance Normalization (AdaIN)的自然风格转换技术，来控制隐空间变量 `z` 。其网络结构如下图所示，它在生产模型中结合了一个映射网络以及 AdaIN 条件分布的做法，并不容易复现，但这篇论文依然值得一读，包含了很多有趣的想法。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/StyleGAN_archi.png)

------

#### 小结

本文主要介绍了 10 篇值得一读的 GAN 论文，从最开始提出这个模型的论文，到截止至 2018 年的论文，其中既有影响很大的 cGAN 和 DCAN，也有图像转换领域非常重要的 Pix2Pix 和 CycleGAN，还有最近效果非常不错的 BigGAN。

如果是希望研究这个方向的，可以看下这 10 篇论文。另外，再推荐一个收集了大量 GAN 论文的 Github 项目，并且根据应用方向划分论文：

- [AdversarialNetsPapers](https://github.com/zhangqianhui/AdversarialNetsPapers)

以及 3 个复现多种 GANs 模型的 github 项目，分别是目前主流的三个框架，TensorFlow、PyTorch 和 Keras：

- [tensorflow-GANs](https://github.com/TwistedW/tensorflow-GANs)：TensorFlow 版本
- [Pytorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)：PyTorch 版本
- [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)：Keras 版本

最后，对于文章介绍的 10 篇论文都已经下载打包后，获取方式：

1. 关注公众号“算法猿的成长”
2. 在公众号会话界面回复 “GAN论文”，即可获取网盘链接。

欢迎关注我的微信公众号--**算法猿的成长**，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_0601.png)

**如果觉得不错，在看、转发就是对小编的一个支持！**

















