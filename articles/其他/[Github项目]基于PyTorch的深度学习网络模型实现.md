
> 2019 年第 48 篇文章，总第 72 篇文章

今天主要分享两份 Github 项目，都是采用 PyTorch 来实现深度学习网络模型，主要是一些常用的模型，包括如 ResNet、DenseNet、ResNext、SENet等，并且也给出相应的实验结果，包含完整的数据处理和载入、模型建立、训练流程搭建，以及测试代码的实现。

接下来就开始介绍这两个项目。


---
#### 1. PyTorch Image Classification

这份代码目前有 200+ 星，主要实现以下的网络，在 MNIST、CIFAR10、FashionMNIST等数据集上进行实验。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_image_classification_1.png)

使用方法如下：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_image_classification_2.png)

然后就是给出作者自己训练的实验结果，然后和原论文的实验结果的对比，包括在训练设置上的区别，然后训练的迭代次数和训练时间也都分别给出。

之后作者还研究了残差单元、学习率策略以及数据增强对分类性能的影响，比如

- 类似金字塔网络的残差单元设计(PyramidNet-like residual units)
- cosine 函数的学习率递减策略(Cosine annealing of learning rate)
- Cutout
- 随机消除(Random Erasing)
- Mixup
- 降采样后的预激活捷径(Preactivation of shortcuts after downsampling)

实验结果表明：

- 类似金字塔网络的残差单元设计有帮助，但不适宜搭配 Preactivation of shortcuts after downsampling
- 基于 cosine 的学习率递减策略提升幅度较小
- Cutout、随机消除以及 Mixup 效果都很好，其中 Mixup 需要的训练次数更多

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_image_classification_3.png)

除了这个实验，后面作者还继续做了好几个实验，包括对 batch 大小、初始学习率大小、标签平滑等方面做了不少实验，并给出对应的实验结果。

最后给出了这些网络模型的论文、不同训练策略的论文。

这个项目除了实现对应的网络模型外，使用不同技巧或者研究基本的 batch 大小、初始学习率都是可以给予我们一定启发，有些技巧是可以应用到网络中，提高分类性能的。


链接：

https://github.com/hysts/pytorch_image_classification


---
#### 2. PyTorch Image Models

这份代码目前有 600+ 星，并且最近几天也有更新，实现的网络更多，包括 DPN、Xception、InceptionResNetV2,以及最近比较火的 EfficientNet。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_image_models.png))

这个项目有以下几个特点：

- 对所有的模型都做了封装，也就是都有默认的配置接口和 API，包括统一的调用分类器接口`get_classifier`、`reset_classifier`，对特征的前向计算接口`forward_features`
- 模型都有一致的预训练模型加载器，即可以决定是否采用预训练模型最后一层或者输入层是否需要从 3 通道变为 1通道；
- 训练脚本可以在不同模式下使用，包括分布式、单机多卡、单机单卡或者单机 CPU
- 动态实现池化层的操作，包括平均池化(average pooling)、最大池化(max pooling)、平均+最大、或者平均和最大池化结果连接而不是叠加；
- 不同训练策略的实现，比如 cosine 学习率、随机消除、标签平滑等
- 实现 Mixup
- 提供一个预测脚本

作者同样给出训练的实验结果，然后还有使用方法，同样也是在指定位置准备好数据，就可以使用了。

另外，作者还给出 ToDo 列表，会继续完善该项目。


链接：

https://github.com/rwightman/pytorch-image-models


---

欢迎关注我的微信公众号--**算法猿的成长**，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_0601.png)

**如果觉得不错，在看、转发就是对小编的一个支持！**

