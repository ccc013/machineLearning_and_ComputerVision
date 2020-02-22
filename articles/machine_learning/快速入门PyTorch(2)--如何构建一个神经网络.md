
> 2019 第 43 篇，总第 67 篇文章

快速入门 PyTorch 教程第二篇，这篇介绍如何构建一个神经网络。上一篇文章：

- [快速入门Pytorch(1)--安装、张量以及梯度](https://mp.weixin.qq.com/s/WZdBm2JQ4yKjISQmXN4TMg)

本文的目录：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_tutorial_content2.png)

---
### 3. 神经网络

在 PyTorch 中 `torch.nn` 专门用于实现神经网络。其中 `nn.Module` 包含了网络层的搭建，以及一个方法-- `forward(input)` ，并返回网络的输出 `outptu` .

下面是一个经典的 LeNet 网络，用于对字符进行分类。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/mnist.png)

对于神经网络来说，一个标准的训练流程是这样的：

- 定义一个多层的神经网络
- 对数据集的预处理并准备作为网络的输入
- 将数据输入到网络
- 计算网络的损失
- 反向传播，计算梯度
- 更新网络的梯度，一个简单的更新规则是 `weight = weight - learning_rate * gradient`

#### 3.1 定义网络

首先定义一个神经网络，下面是一个 5 层的卷积神经网络，包含两层卷积层和三层全连接层：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 输入图像是单通道，conv1 kenrnel size=5*5，输出通道 6
        self.conv1 = nn.Conv2d(1, 6, 5)
        # conv2 kernel size=5*5, 输出通道 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # max-pooling 采用一个 (2,2) 的滑动窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 核(kernel)大小是方形的话，可仅定义一个数字，如 (2,2) 用 2 即可
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        # 除了 batch 维度外的所有维度
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
```

打印网络结构：

```python
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

这里必须实现 `forward` 函数，而 `backward`  函数在采用 `autograd` 时就自动定义好了，在 `forward` 方法可以采用任何的张量操作。

`net.parameters()` 可以返回网络的训练参数，使用例子如下：

```python
params = list(net.parameters())
print('参数数量: ', len(params))
# conv1.weight
print('第一个参数大小: ', params[0].size())
```

输出：

```python
参数数量:  10
第一个参数大小:  torch.Size([6, 1, 5, 5])
```

然后简单测试下这个网络，随机生成一个 32*32 的输入：

```python
# 随机定义一个变量输入网络
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

输出结果：

```python
tensor([[ 0.1005,  0.0263,  0.0013, -0.1157, -0.1197, -0.0141,  0.1425, -0.0521,
          0.0689,  0.0220]], grad_fn=<ThAddmmBackward>)
```

接着反向传播需要先清空梯度缓存，并反向传播随机梯度：

```python
# 清空所有参数的梯度缓存，然后计算随机梯度进行反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))
```

**注意**：

> `torch.nn` 只支持**小批量(mini-batches)**数据，也就是输入不能是单个样本，比如对于 `nn.Conv2d` 接收的输入是一个 4 维张量--`nSamples * nChannels * Height * Width` 。
>
> 所以，如果你输入的是单个样本，**需要采用 `input.unsqueeze(0)` 来扩充一个假的 batch 维度，即从 3 维变为 4 维**。

#### 3.2 损失函数

损失函数的输入是 `(output, target)` ，即网络输出和真实标签对的数据，然后返回一个数值表示网络输出和真实标签的差距。

PyTorch 中其实已经定义了不少的[损失函数](https://pytorch.org/docs/nn.html#loss-functions)，这里仅采用简单的均方误差：`nn.MSELoss` ，例子如下：

```python
output = net(input)
# 定义伪标签
target = torch.randn(10)
# 调整大小，使得和 output 一样的 size
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

输出如下：

```python
tensor(0.6524, grad_fn=<MseLossBackward>)
```

这里，整个网络的数据输入到输出经历的计算图如下所示，其实也就是数据从输入层到输出层，计算 `loss` 的过程。

```python
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

如果调用 `loss.backward()` ，那么整个图都是可微分的，也就是说包括 `loss` ，图中的所有张量变量，只要其属性 `requires_grad=True` ，那么其梯度 `.grad`  张量都会随着梯度一直累计。

用代码来说明：

```python
# MSELoss
print(loss.grad_fn)
# Linear layer
print(loss.grad_fn.next_functions[0][0])
# Relu
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
```

输出：

```python
<MseLossBackward object at 0x0000019C0C349908>

<ThAddmmBackward object at 0x0000019C0C365A58>

<ExpandBackward object at 0x0000019C0C3659E8>
```

#### 3.3 反向传播

反向传播的实现只需要调用 `loss.backward()` 即可，当然首先需要清空当前梯度缓存，即`.zero_grad()` 方法，否则之前的梯度会累加到当前的梯度，这样会影响权值参数的更新。

下面是一个简单的例子，以 `conv1` 层的偏置参数 `bias` 在反向传播前后的结果为例：

```python
# 清空所有参数的梯度缓存
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

输出结果：

```python
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])

conv1.bias.grad after backward
tensor([ 0.0069,  0.0021,  0.0090, -0.0060, -0.0008, -0.0073])
```

了解更多有关 `torch.nn` 库，可以查看官方文档：

https://pytorch.org/docs/stable/nn.html

#### 3.4 更新权重

采用随机梯度下降(Stochastic Gradient Descent, SGD)方法的最简单的更新权重规则如下：

`weight = weight - learning_rate * gradient`

按照这个规则，代码实现如下所示：

```python
# 简单实现权重的更新例子
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

但是这只是最简单的规则，深度学习有很多的优化算法，不仅仅是 `SGD`，还有 `Nesterov-SGD, Adam, RMSProp` 等等，为了采用这些不同的方法，这里采用 `torch.optim` 库，使用例子如下所示：

```python
import torch.optim as optim
# 创建优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 在训练过程中执行下列操作
optimizer.zero_grad() # 清空梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
# 更新权重
optimizer.step()
```

**注意**，同样需要调用 `optimizer.zero_grad()` 方法清空梯度缓存。

本小节教程：

https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

本小节的代码：

https://github.com/ccc013/DeepLearning_Notes/blob/master/Pytorch/practise/neural_network.ipynb


---
### 小结

第二篇主要介绍了搭建一个神经网络，包括定义网络、选择损失函数、反向传播计算梯度和更新权值参数。


欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)



#### 往期精彩推荐

##### 机器学习系列

- [初学者的机器学习入门实战教程！](https://mp.weixin.qq.com/s/HoFiD0ItcO5_pVMspni_xw)
- [模型评估、过拟合欠拟合以及超参数调优方法](https://mp.weixin.qq.com/s/1NxVNtKNsZFWYI62KzL1GA)
- [常用机器学习算法汇总比较(完）](https://mp.weixin.qq.com/s/V2C4u9mSHmQdVl9ZYs1-FQ)
- [常用机器学习算法汇总比较(上）](https://mp.weixin.qq.com/s/4Ban_TiMKYUBXTq4WcMr5g)
- [机器学习入门系列(2)--如何构建一个完整的机器学习项目(一)](https://mp.weixin.qq.com/s/nMG5Z3CPdwhg4XQuMbNqbw)
- [特征工程之数据预处理（上）](https://mp.weixin.qq.com/s/BnTXjzHSb5-4s0O0WuZYlg)
- [来了解下计算机视觉的八大应用](https://mp.weixin.qq.com/s/z9QbjeoLoycDaawUV_QcCA)

##### Github项目 & 资源教程推荐

- [[Github 项目推荐] 一个更好阅读和查找论文的网站](https://mp.weixin.qq.com/s/ImQcGt8guLKZawNLS-_HzA)
- [[资源分享] TensorFlow 官方中文版教程来了](https://mp.weixin.qq.com/s/Si1YaYLfhL1upbjQkvireQ)
- [必读的AI和深度学习博客](https://mp.weixin.qq.com/s/0J2raJqiYsYPqwAV1MALaw)
- [[教程]一份简单易懂的 TensorFlow 教程](https://mp.weixin.qq.com/s/vXIM6Ttw37yzhVB_CvXmCA)
- [[资源]推荐一些Python书籍和教程，入门和进阶的都有！](https://mp.weixin.qq.com/s/jkIQTjM9C3fDvM1c6HwcQg)
- [[Github项目推荐] 机器学习& Python 知识点速查表](https://mp.weixin.qq.com/s/kn2DUJHL48UyuoUEhcfuxw)
- [[Github项目推荐] 推荐三个助你更好利用Github的工具](https://mp.weixin.qq.com/s/Mtijg-AXN4zCeZktkr7nqQ)
- [Github上的各大高校资料以及国外公开课视频](https://mp.weixin.qq.com/s/z3XS6fO303uVZSPmFKBKlQ)
- [这些单词你都念对了吗？顺便推荐三份程序员专属英语教程！](https://mp.weixin.qq.com/s/yU3wczdePK9OvMgjNg5bug)
