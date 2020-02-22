
> 2019 第 42 篇，总第 66 篇文章

这是翻译自官方的入门教程，教程地址如下：

[DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

虽然教程名字是 60 分钟入门，但是内容还是比较多，所以会分成大概 4 篇来介绍，这是第一篇，目录如下：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_tutorial_content.png)


------

### 1. Pytorch 是什么

Pytorch 是一个基于 Python 的科学计算库，它面向以下两种人群：

- 希望将其代替 Numpy 来利用 GPUs 的威力；
- 一个可以提供更加灵活和快速的深度学习研究平台。

#### 1.1 安装

pytorch 的安装可以直接查看官网教程，如下所示，官网地址：https://pytorch.org/get-started/locally/

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_install.png)

根据提示分别选择系统(Linux、Mac 或者 Windows)，安装方式(Conda，Pip，LibTorch 或者源码安装)、使用的编程语言(Python 2.7 或者 Python 3.5,3.6,3.7 或者是 C++)，如果是 GPU 版本，就需要选择 CUDA 的 版本，所以，如果如上图所示选择，安装的命令是：

```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch 
```

这里推荐采用 Conda 安装，即使用 Anaconda，主要是可以设置不同环境配置不同的设置，关于 Anaconda 可以查看我之前写的 [Python 基础入门--简介和环境配置](https://mp.weixin.qq.com/s/DrGr8eiZXj_wTnyDaKFpbg) 。

当然这里会安装最新版本的 Pytorch，也就是 1.1 版本，如果希望安装之前的版本，可以点击下面的网址：

http://pytorch.org/get-started/previous-versions/

如下图所示，安装 0.4.1 版本的 pytorch，在不同版本的 CUDA 以及没有 CUDA 的情况。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_install2.png)

然后还有其他的安装方式，具体可以自己点击查看。

安装后，输入下列命令：

```python
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
```

输出结果类似下面的结果即安装成功：

```python
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

然后是验证能否正确运行在 GPU 上，输入下列代码，这份代码中 `cuda.is_available()` 主要是用于检测是否可以使用当前的 GPU 显卡，如果返回 True，当然就可以运行，否则就不能。

```python
import torch
torch.cuda.is_available()
```

#### 1.2 张量(Tensors)

Pytorch 的一大作用就是可以代替 Numpy 库，所以首先介绍 Tensors ，也就是张量，它相当于 Numpy 的多维数组(ndarrays)。两者的区别就是 Tensors 可以应用到 GPU 上加快计算速度。

首先导入必须的库，主要是 torch

```python
from __future__ import print_function
import torch
```

##### 1.2.1 声明和定义

首先是对 Tensors 的声明和定义方法，分别有以下几种：

- **torch.empty()**: 声明一个未初始化的矩阵。

```python
# 创建一个 5*3 的矩阵
x = torch.empty(5, 3)
print(x)
```

输出结果如下：

```python
tensor([[9.2737e-41, 8.9074e-01, 1.9286e-37],
        [1.7228e-34, 5.7064e+01, 9.2737e-41],
        [2.2803e+02, 1.9288e-37, 1.7228e-34],
        [1.4609e+04, 9.2737e-41, 5.8375e+04],
        [1.9290e-37, 1.7228e-34, 3.7402e+06]])
```

- **torch.rand()**：随机初始化一个矩阵

```python
# 创建一个随机初始化的 5*3 矩阵
rand_x = torch.rand(5, 3)
print(rand_x)
```

输出结果：

```python
tensor([[0.4311, 0.2798, 0.8444],
        [0.0829, 0.9029, 0.8463],
        [0.7139, 0.4225, 0.5623],
        [0.7642, 0.0329, 0.8816],
        [1.0000, 0.9830, 0.9256]])
```

- **torch.zeros()**：创建数值皆为 0 的矩阵

```python
# 创建一个数值皆是 0，类型为 long 的矩阵
zero_x = torch.zeros(5, 3, dtype=torch.long)
print(zero_x)
```

输出结果如下：

```python
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```

类似的也可以创建数值都是 1 的矩阵，调用 `torch.ones`

- **torch.tensor()**：直接传递 tensor 数值来创建

```python
# tensor 数值是 [5.5, 3]
tensor1 = torch.tensor([5.5, 3])
print(tensor1)
```

输出结果：

```python
tensor([5.5000, 3.0000])
```

除了上述几种方法，还可以根据已有的 tensor 变量创建新的 tensor 变量，这种做法的好处就是可以保留已有 tensor 的一些属性，包括尺寸大小、数值属性，除非是重新定义这些属性。相应的实现方法如下：

- **tensor.new_ones()**：new_*() 方法需要输入尺寸大小

```python
# 显示定义新的尺寸是 5*3，数值类型是 torch.double
tensor2 = tensor1.new_ones(5, 3, dtype=torch.double)  # new_* 方法需要输入 tensor 大小
print(tensor2)
```

输出结果：

```python
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
```

- **torch.randn_like(old_tensor)**：保留相同的尺寸大小

```python
# 修改数值类型
tensor3 = torch.randn_like(tensor2, dtype=torch.float)
print('tensor3: ', tensor3)
```

输出结果，这里是根据上个方法声明的 `tensor2` 变量来声明新的变量，可以看出尺寸大小都是 5*3，但是数值类型是改变了的。

```python
tensor3:  tensor([[-0.4491, -0.2634, -0.0040],
        [-0.1624,  0.4475, -0.8407],
        [-0.6539, -1.2772,  0.6060],
        [ 0.2304,  0.0879, -0.3876],
        [ 1.2900, -0.7475, -1.8212]])
```

最后，对 tensors 的尺寸大小获取可以采用 `tensor.size()` 方法：

```python
print(tensor3.size())  
# 输出: torch.Size([5, 3])
```

**注意**： `torch.Size` 实际上是**元组(tuple)类型，所以支持所有的元组操作**。

##### 1.2.2 操作(Operations)

操作也包含了很多语法，但这里作为快速入门，仅仅以加法操作作为例子进行介绍，更多的操作介绍可以点击下面网址查看官方文档，包括转置、索引、切片、数学计算、线性代数、随机数等等：

https://pytorch.org/docs/stable/torch.html

对于加法的操作，有几种实现方式：

- **+** 运算符
- **torch.add(tensor1, tensor2, [out=tensor3])** 
- **tensor1.add_(tensor2)**：直接修改 tensor 变量

```python
tensor4 = torch.rand(5, 3)
print('tensor3 + tensor4= ', tensor3 + tensor4)
print('tensor3 + tensor4= ', torch.add(tensor3, tensor4))
# 新声明一个 tensor 变量保存加法操作的结果
result = torch.empty(5, 3)
torch.add(tensor3, tensor4, out=result)
print('add result= ', result)
# 直接修改变量
tensor3.add_(tensor4)
print('tensor3= ', tensor3)
```

输出结果

```python
tensor3 + tensor4=  tensor([[ 0.1000,  0.1325,  0.0461],
        [ 0.4731,  0.4523, -0.7517],
        [ 0.2995, -0.9576,  1.4906],
        [ 1.0461,  0.7557, -0.0187],
        [ 2.2446, -0.3473, -1.0873]])

tensor3 + tensor4=  tensor([[ 0.1000,  0.1325,  0.0461],
        [ 0.4731,  0.4523, -0.7517],
        [ 0.2995, -0.9576,  1.4906],
        [ 1.0461,  0.7557, -0.0187],
        [ 2.2446, -0.3473, -1.0873]])

add result=  tensor([[ 0.1000,  0.1325,  0.0461],
        [ 0.4731,  0.4523, -0.7517],
        [ 0.2995, -0.9576,  1.4906],
        [ 1.0461,  0.7557, -0.0187],
        [ 2.2446, -0.3473, -1.0873]])

tensor3=  tensor([[ 0.1000,  0.1325,  0.0461],
        [ 0.4731,  0.4523, -0.7517],
        [ 0.2995, -0.9576,  1.4906],
        [ 1.0461,  0.7557, -0.0187],
        [ 2.2446, -0.3473, -1.0873]])
```

**注意**：可以改变 tensor 变量的操作都带有一个后缀 `_`, 例如 `x.copy_(y), x.t_()` 都可以改变 x 变量

除了加法运算操作，对于 Tensor 的访问，和 Numpy 对数组类似，可以使用索引来访问某一维的数据，如下所示：

```python
# 访问 tensor3 第一列数据
print(tensor3[:, 0])
```

输出结果：

```python
tensor([0.1000, 0.4731, 0.2995, 1.0461, 2.2446])
```

对 Tensor 的尺寸修改，可以采用 `torch.view()` ，如下所示：

```python
x = torch.randn(4, 4)
y = x.view(16)
# -1 表示除给定维度外的其余维度的乘积
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
```

输出结果：

```python
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

如果 tensor 仅有一个元素，可以采用 `.item()` 来获取类似 Python 中整数类型的数值：

```python
x = torch.randn(1)
print(x)
print(x.item())
```

输出结果:

```python
tensor([0.4549])
0.4549027979373932
```

更多的运算操作可以查看官方文档的介绍：

https://pytorch.org/docs/stable/torch.html

#### 1.3 和 Numpy 数组的转换

Tensor 和 Numpy 的数组可以相互转换，并且两者转换后共享在 CPU 下的内存空间，即改变其中一个的数值，另一个变量也会随之改变。

##### 1.3.1 张量转换为 Numpy 数组

实现 Tensor 转换为 Numpy 数组的例子如下所示，调用 `tensor.numpy()` 可以实现这个转换操作。

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
```

输出结果：

```
tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
```

此外，刚刚说了两者是共享同个内存空间的，例子如下所示，修改 `tensor` 变量 `a`，看看从 `a` 转换得到的 Numpy 数组变量 `b` 是否发生变化。

```python
a.add_(1)
print(a)
print(b)
```

输出结果如下，很明显，`b` 也随着 `a` 的改变而改变。

```python
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

##### 1.3.2 Numpy 数组转换为张量

转换的操作是调用 `torch.from_numpy(numpy_array)` 方法。例子如下所示：

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

输出结果：

```python
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

在 `CPU` 上，除了 `CharTensor` 外的所有 `Tensor` 类型变量，都支持和 `Numpy` 数组的相互转换操作。

#### 1.4. CUDA 张量

`Tensors` 可以通过 `.to` 方法转换到不同的设备上，即 CPU 或者 GPU 上。例子如下所示：

```python
# 当 CUDA 可用的时候，可用运行下方这段代码，采用 torch.device() 方法来改变 tensors 是否在 GPU 上进行计算操作
if torch.cuda.is_available():
    device = torch.device("cuda")          # 定义一个 CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 显示创建在 GPU 上的一个 tensor
    x = x.to(device)                       # 也可以采用 .to("cuda") 
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # .to() 方法也可以改变数值类型
```

输出结果，第一个结果就是在 GPU 上的结果，打印变量的时候会带有 `device='cuda:0'`，而第二个是在 CPU 上的变量。

```python
tensor([1.4549], device='cuda:0')

tensor([1.4549], dtype=torch.float64)
```

本小节教程：

https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

本小节的代码：

https://github.com/ccc013/DeepLearning_Notes/blob/master/Pytorch/practise/basic_practise.ipynb

### 2. autograd

对于 Pytorch 的神经网络来说，非常关键的一个库就是 `autograd` ，它主要是提供了对 Tensors 上所有运算操作的自动微分功能，也就是计算梯度的功能。它属于 `define-by-run` 类型框架，即反向传播操作的定义是根据代码的运行方式，因此每次迭代都可以是不同的。

接下来会简单介绍一些例子来说明这个库的作用。

#### 2.1 张量

`torch.Tensor` 是 Pytorch 最主要的库，当设置它的属性 `.requires_grad=True`，那么就会开始追踪在该变量上的所有操作，而完成计算后，可以调用 `.backward()` 并自动计算所有的梯度，得到的梯度都保存在属性 `.grad` 中。

调用 `.detach()` 方法分离出计算的历史，可以停止一个 tensor 变量继续追踪其历史信息 ，同时也防止未来的计算会被追踪。

而如果是希望防止跟踪历史（以及使用内存），可以将代码块放在 `with torch.no_grad():` 内，这个做法在使用一个模型进行评估的时候非常有用，因为模型会包含一些带有 `requires_grad=True` 的训练参数，但实际上并不需要它们的梯度信息。

对于 `autograd` 的实现，还有一个类也是非常重要-- `Function` 。

`Tensor` 和 `Function` 两个类是有关联并建立了一个非循环的图，可以编码一个完整的计算记录。每个 tensor 变量都带有属性 `.grad_fn` ，该属性引用了创建了这个变量的 `Function` （除了由用户创建的 Tensors，它们的 `grad_fn=None` )。

如果要进行求导运算，可以调用一个 `Tensor` 变量的方法 `.backward()` 。如果该变量是一个标量，即仅有一个元素，那么不需要传递任何参数给方法 `.backward()` ，当包含多个元素的时候，就必须指定一个 `gradient` 参数，表示匹配尺寸大小的 tensor，这部分见第二小节介绍梯度的内容。

接下来就开始用代码来进一步介绍。

首先导入必须的库：

```python
import torch
```

开始创建一个 tensor， 并让 `requires_grad=True` 来追踪该变量相关的计算操作：

```python
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

输出结果：

```python
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
```

执行任意计算操作，这里进行简单的加法运算：

```python
y = x + 2
print(y)
```

输出结果：

```python
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward>)
```

`y` 是一个操作的结果，所以它带有属性 `grad_fn`：

```python
print(y.grad_fn)
```

输出结果：

```python
<AddBackward object at 0x00000216D25DCC88>
```

继续对变量 `y` 进行操作：

```python
z = y * y * 3
out = z.mean()

print('z=', z)
print('out=', out)
```

输出结果：

```python
z= tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward>)

out= tensor(27., grad_fn=<MeanBackward1>)
```

实际上，一个 `Tensor` 变量的默认 `requires_grad` 是 `False` ，可以像上述定义一个变量时候指定该属性是  `True`，当然也可以定义变量后，调用 `.requires_grad_(True)` 设置为 `True` ，这里带有后缀 `_` 是会改变变量本身的属性，在上一节介绍加法操作 `add_()` 说明过，下面是一个代码例子：

```python
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```

输出结果如下，第一行是为设置 `requires_grad` 的结果，接着显示调用 `.requires_grad_(True)`，输出结果就是 `True` 。

```
False

True

<SumBackward0 object at 0x00000216D25ED710>
```

#### 2.2 梯度

接下来就是开始计算梯度，进行反向传播的操作。`out` 变量是上一小节中定义的，它是一个标量，因此 `out.backward()` 相当于 `out.backward(torch.tensor(1.))` ，代码如下：

```python
out.backward()
# 输出梯度 d(out)/dx
print(x.grad)
```

输出结果：

```python
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

结果应该就是得到数值都是 4.5 的矩阵。这里我们用 `o` 表示 `out` 变量，那么根据之前的定义会有：
$$
o = \frac{1}{4}\sum_iz_i,\\
z_i = 3(x_i + 2)^2, \\
z_i|_{x_i=1} = 27
$$
详细来说，初始定义的 `x` 是一个全为 1 的矩阵，然后加法操作 `x+2` 得到 `y` ，接着 `y*y*3`， 得到 `z` ，并且此时 `z` 是一个 2*2 的矩阵，所以整体求平均得到 `out` 变量应该是除以 4，所以得到上述三条公式。

因此，计算梯度：
$$
\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2),\\
\frac{\partial o}{\partial x_i}|_{x_i=1} = \frac{9}{2} = 4.5
$$
从数学上来说，如果你有一个向量值函数：
$$
\vec{y}=f(\vec{x})
$$
那么对应的梯度是一个雅克比矩阵(Jacobian matrix)：
$$
\begin{split}J=\left(\begin{array}{ccc}
 \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
 \vdots & \ddots & \vdots\\
 \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
 \end{array}\right)\end{split}
$$
一般来说，`torch.autograd` 就是用于计算雅克比向量(vector-Jacobian)乘积的工具。这里略过数学公式，直接上代码例子介绍：

```python
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
```

输出结果：

```python
tensor([ 237.5009, 1774.2396,  274.0625], grad_fn=<MulBackward>)
```

这里得到的变量 `y` 不再是一个标量，`torch.autograd` 不能直接计算完整的雅克比行列式，但我们可以通过简单的传递向量给 `backward()` 方法作为参数得到雅克比向量的乘积，例子如下所示：

```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
```

输出结果：

```python
tensor([ 102.4000, 1024.0000,    0.1024])
```

最后，加上 `with torch.no_grad()` 就可以停止追踪变量历史进行自动梯度计算：

```python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
```

输出结果：

```python
True

True

False
```

更多有关 `autograd` 和 `Function` 的介绍：

https://pytorch.org/docs/autograd

本小节教程：

https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

本小节的代码：

https://github.com/ccc013/DeepLearning_Notes/blob/master/Pytorch/practise/autograd.ipynb



---

### 小结

第一篇主要简单介绍 Pytorch 的两大作用，替代 Numpy 以及一个新的深度学习工具，当然主要还是后者让其能够在短短两三年内快速发展，并且由于 Tensorflow 的一些缺点，越来越多人会选择采用 Pytorch 工具，特别是对于学术界的科研学者来说，Pytorch 其实会上手更加快。

另外，还介绍了最重要也是最基础的张量的知识，其方法、操作和 Numpy 的数组非常相似，两者还可以互相转换，稍微不同的是张量可以应用到 GPU 上加快计算速度。

最后简单介绍了 `autograd` 这个库，对于深度学习非常重要，它可以自动计算梯度，非常有用。


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












