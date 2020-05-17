原文：https://github.com/vahidk/EffectivePyTorch

作者：vahidk

### 前言

这是一份 PyTorch 教程和最佳实践笔记，目录如下所示：

1. PyTorch 基础
2. 将模型封装为模块
3. 广播机制的优缺点
4. 使用好重载的运算符
5. 采用 TorchScript 优化运行时间
6. 构建高效的自定义数据加载类
7. PyTorch 的数值稳定性



首先 PyTorch 的安装可以根据官方文档进行操作：

https://pytorch.org/

```shell
pip install torch torchvision
```



------

### 1. PyTorch 基础

PyTorch 是数值计算方面其中一个最流行的库，同时也是机器学习研究方面最广泛使用的框架。在很多方面，它和 NumPy 都非常相似，但是它可以在不需要代码做多大改变的情况下，在 CPUs，GPUs，TPUs 上实现计算，以及非常容易实现分布式计算的操作。PyTorch 的其中一个最重要的特征就是自动微分。它可以让需要采用梯度下降算法进行训练的机器学习算法的实现更加方便，可以更高效的自动计算函数的梯度。我们的目标是提供更好的 PyTorch 介绍以及讨论使用 PyTorch 的一些最佳实践。

对于 PyTorch 第一个需要学习的就是张量（Tensors）的概念，张量就是多维数组，它和 numpy 的数组非常相似，但多了一些函数功能。

一个张量可以存储一个标量数值、一个数组、一个矩阵：

```python
import torch
# 标量数值
a = torch.tensor(3)
print(a)  # tensor(3)
# 数组
b = torch.tensor([1, 2])
print(b)  # tensor([1, 2])
# 矩阵
c = torch.zeros([2, 2])
print(c)  # tensor([[0., 0.], [0., 0.]])
# 任意维度的张量
d = torch.rand([2, 2, 2])
```

张量还可以高效的执行代数的运算。机器学习应用中最常见的运算就是矩阵乘法。例如希望将两个随机矩阵进行相乘，维度分别是 $3\times 5$ 和 $5\times 4$，这个运算可以通过矩阵相乘运算实现（@）：

```python
import torch

x = torch.randn([3, 5])
y = torch.randn([5, 4])
z = x @ y

print(z)
```

对于向量相加，如下所示：

```python
z = x + y
```

将张量转换为 `numpy` 数组，可以调用 `numpy()` 方法：

```python
print(z.numpy())
```

当然，反过来 `numpy` 数组转换为张量是可以的：

```python
x = torch.tensor(np.random.normal([3, 5]))
```

#### 自动微分

PyTorch 中相比 `numpy`  最大优点就是可以实现自动微分，这对于优化神经网络参数的应用非常有帮助。下面通过一个例子来帮助理解这个优点。

假设现在有一个复合函数：`g(u(x))` ，为了计算 `g` 对 `x` 的导数，这里可以采用链式法则，即
$$
\frac{dg}{dx} = \frac{dg}{du} * \frac{du}{dx}
$$
而 PyTorch 可以自动实现这个求导的过程。

为了在 PyTorch 中计算导数，首先要创建一个张量，并设置其 `requires_grad = True` ，然后利用张量运算来定义函数，这里假设 `u` 是一个二次方的函数，而 `g` 是一个简单的线性函数，代码如下所示：

```python
x = torch.tensor(1.0, requires_grad=True)

def u(x):
  return x * x

def g(u):
  return -u
```

在这个例子中，复合函数就是 $g(u(x)) = -x*x$，所以导数是 $-2x$，如果 `x=1` ，那么可以得到 `-2` 。

在 PyTorch 中调用梯度函数：

```python
dgdx = torch.autograd.grad(g(u(x)), x)[0]
print(dgdx)  # tensor(-2.)
```



#### 拟合曲线

**为了展示自动微分有多么强大**，这里介绍另一个例子。

首先假设我们有一些服从一个曲线（也就是函数 $f(x)=5x^2 + 3$）的样本，然后希望基于这些样本来评估这个函数 `f(x)` 。我们先定义一个带参数的函数:
$$
g(x, w) = w_0 x^2 + w_1 x + w_2
$$
函数的输入是 `x`，然后 `w` 是参数，目标是找到合适的参数使得下列式子成立：
$$
g(x, w) = f(x)
$$
实现的一个方法可以是通过优化下面的损失函数来实现：
$$
L(w) = \sum(f(x) - g(x, w))^2
$$
尽管这个问题里有一个正式的函数（即 `f(x)` 是一个具体的函数），但这里我们还是采用一个更加通用的方法，可以应用到任何一个可微分的函数，并采用随机梯度下降法，即通过计算 `L(w)` 对于每个参数 `w` 的梯度的平均值，然后不断从相反反向移动。

利用 PyTorch 实现的代码如下所示：

```python
import numpy as np
import torch

# Assuming we know that the desired function is a polynomial of 2nd degree, we
# allocate a vector of size 3 to hold the coefficients and initialize it with
# random noise.
w = torch.tensor(torch.randn([3, 1]), requires_grad=True)

# We use the Adam optimizer with learning rate set to 0.1 to minimize the loss.
opt = torch.optim.Adam([w], 0.1)

def model(x):
    # We define yhat to be our estimate of y.
    f = torch.stack([x * x, x, torch.ones_like(x)], 1)
    yhat = torch.squeeze(f @ w, 1)
    return yhat

def compute_loss(y, yhat):
    # The loss is defined to be the mean squared error distance between our
    # estimate of y and its true value. 
    loss = torch.nn.functional.mse_loss(yhat, y)
    return loss

def generate_data():
    # Generate some training data based on the true function
    x = torch.rand(100) * 20 - 10
    y = 5 * x * x + 3
    return x, y

def train_step():
    x, y = generate_data()

    yhat = model(x)
    loss = compute_loss(y, yhat)

    opt.zero_grad()
    loss.backward()
    opt.step()

for _ in range(1000):
    train_step()

print(w.detach().numpy())
```

运行上述代码，可以得到和下面相近的结果：

```
[4.9924135, 0.00040895029, 3.4504161]
```

这和我们的参数非常接近。

上述只是 PyTorch 可以做的事情的冰山一角。很多问题，比如优化一个带有上百万参数的神经网络，都可以用 PyTorch 高效的用几行代码实现，PyTorch 可以跨多个设备和线程进行拓展，并且支持多个平台。



------

### 2. 将模型封装为模块

在之前的例子中，我们构建模型的方式是直接实现张量间的运算操作。但为了让代码看起来更加有组织，推荐采用 PyTorch 的 `modules` 模块。一个模块实际上是一个包含参数和压缩模型运算的容器。

比如，如果想实现一个线性模型 $y = ax + b$，那么实现的代码可以如下所示：

```python
import torch

class Net(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.a = torch.nn.Parameter(torch.rand(1))
    self.b = torch.nn.Parameter(torch.rand(1))

  def forward(self, x):
    yhat = self.a * x + self.b
    return yhat
```

使用的例子如下所示，需要实例化声明的模型，并且像调用函数一样使用它：

```python
x = torch.arange(100, dtype=torch.float32)

net = Net()
y = net(x)
```

参数都是设置 `requires_grad` 为 `true` 的张量。通过模型的 `parameters()` 方法可以很方便的访问和使用参数，如下所示：

```python
for p in net.parameters():
    print(p)
```

现在，假设是一个未知的函数 `y=5x+3+n` ，注意这里的 `n` 是表示噪音，然后希望优化模型参数来拟合这个函数，首先可以简单从这个函数进行采样，得到一些样本数据：

```python
x = torch.arange(100, dtype=torch.float32) / 100
y = 5 * x + 3 + torch.rand(100) * 0.3
```

和上一个例子类似，需要定义一个损失函数并优化模型的参数，如下所示：

```python
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for i in range(10000):
  net.zero_grad()
  yhat = net(x)
  loss = criterion(yhat, y)
  loss.backward()
  optimizer.step()

print(net.a, net.b) # Should be close to 5 and 3
```



在 PyTorch 中已经实现了很多预定义好的模块。比如 `torch.nn.Linear` 就是一个类似上述例子中定义的一个更加通用的线性函数，所以我们可以采用这个函数来重写我们的模型代码，如下所示：

```python
class Net(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(1, 1)

  def forward(self, x):
    yhat = self.linear(x.unsqueeze(1)).squeeze(1)
    return yhat
```

这里用到了两个函数，`squeeze` 和 `unsqueeze` ，主要是`torch.nn.Linear` 会对一批向量而不是数值进行操作。

同样，默认调用 `parameters()` 会返回其所有子模块的参数：

```python
net = Net()
for p in net.parameters():
    print(p)
```

当然也有一些预定义的模块是作为包容其他模块的容器，最常用的就是 `torch.nn.Sequential` ，它的名字就暗示了它主要用于堆叠多个模块（或者网络层），例如堆叠两个线性网络层，中间是一个非线性函数 `ReLU` ，如下所示：

```python
model = torch.nn.Sequential(
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 10),
)
```



------

### 3. 广播机制的优缺点

#### 优点

**PyTorch 支持广播的元素积运算**。正常情况下，当想执行类似加法和乘法操作的时候，你需要确认操作数的形状是匹配的，比如无法进行一个 `[3, 2]` 大小的张量和 `[3, 4]` 大小的张量的加法操作。

但是存在一种特殊的情况：只有单一维度的时候，PyTorch 会隐式的根据另一个操作数的维度来拓展只有单一维度的操作数张量。因此，实现 `[3,2]` 大小的张量和 `[3,1]` 大小的张量相加的操作是合法的。

如下代码展示了一个加法的例子：

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[1.], [2.]])
# c = a + b.repeat([1, 2])
c = a + b

print(c)
```

广播机制可以实现隐式的维度复制操作（`repeat` 操作），并且代码更短，内存使用上也更加高效，因为不需要存储复制的数据的结果。这个机制非常适合用于结合多个维度不同的特征的时候。

为了拼接不同维度的特征，通常的做法是先对输入张量进行维度上的复制，然后拼接后使用非线性激活函数。整个过程的代码实现如下所示：

```python
a = torch.rand([5, 3, 5])
b = torch.rand([5, 1, 6])

linear = torch.nn.Linear(11, 10)

# concat a and b and apply nonlinearity
tiled_b = b.repeat([1, 3, 1]) # b shape:  [5, 3, 6]
c = torch.cat([a, tiled_b], 2) # c shape: [5, 3, 11]
d = torch.nn.functional.relu(linear(c))

print(d.shape)  # torch.Size([5, 3, 10])
```

但实际上通过广播机制可以实现得更加高效，即 `f(m(x+y))` 是等同于 `f(mx+my)` 的，也就是我们可以先分别做线性操作，然后通过广播机制来做隐式的拼接操作，如下所示：

```python
a = torch.rand([5, 3, 5])
b = torch.rand([5, 1, 6])

linear1 = torch.nn.Linear(5, 10)
linear2 = torch.nn.Linear(6, 10)

pa = linear1(a) # pa shape: [5, 3, 10]
pb = linear2(b) # pb shape: [5, 1, 10]
d = torch.nn.functional.relu(pa + pb)

print(d.shape)  # torch.Size([5, 3, 10])
```

实际上这段代码非常通用，可以用于任意维度大小的张量，只要它们之间是可以实现广播机制的，如下所示：

```python
class Merge(torch.nn.Module):
    def __init__(self, in_features1, in_features2, out_features, activation=None):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features1, out_features)
        self.linear2 = torch.nn.Linear(in_features2, out_features)
        self.activation = activation

    def forward(self, a, b):
        pa = self.linear1(a)
        pb = self.linear2(b)
        c = pa + pb
        if self.activation is not None:
            c = self.activation(c)
        return c
```

#### 缺点

到目前为止，我们讨论的都是广播机制的优点。但它的缺点是什么呢？原因也是出现在**隐式的操作**，这种做法非常不利于进行代码的调试。

这里给出一个代码例子：

```python
a = torch.tensor([[1.], [2.]])
b = torch.tensor([1., 2.])
c = torch.sum(a + b)

print(c)
```

所以上述代码的输出结果 `c` 是什么呢？你可能觉得是 6，但这是错的，正确答案是 12 。这是因为当两个张量的维度不匹配的时候，PyTorch 会自动将维度低的张量的第一个维度进行拓展，然后在进行元素之间的运算，所以这里会将`b`  先拓展为 `[[1, 2], [1, 2]]`，然后 `a+b` 的结果应该是 `[[2,3], [3, 4]]` ，然后`sum` 操作是将所有元素求和得到结果 12。

那么避免这种结果的方法就是显式的操作，比如在这个例子中就需要指定好想要求和的维度，这样进行代码调试会更简单，代码修改后如下所示：

```python
a = torch.tensor([[1.], [2.]])
b = torch.tensor([1., 2.])
c = torch.sum(a + b, 0)

print(c)
```

这里得到的 `c` 的结果是 `[5, 7]`，而我们基于结果的维度可以知道出现了错误。

这有个通用的做法，就是在做累加（ `reduction` ）操作或者使用 `torch.squeeze` 的时候总是指定好维度。



------

### 4. 使用好重载的运算符

和 NumPy 一样，PyTorch 会重载 python 的一些运算符来让 PyTorch 代码更简短和更有可读性。

例如，切片操作就是其中一个重载的运算符，可以更容易的对张量进行索引操作，如下所示：

```python
z = x[begin:end]  # z = torch.narrow(0, begin, end-begin)
```

但需要谨慎使用这个运算符，它和其他运算符一样，也有一些副作用。正因为它是一个非常常用的运算操作，如果过度使用可以导致代码变得低效。

这里给出一个例子来展示它是如何导致代码变得低效的。这个例子中我们希望对一个矩阵手动实现行之间的累加操作：

```python
import torch
import time

x = torch.rand([500, 10])

z = torch.zeros([10])

start = time.time()
for i in range(500):
    z += x[i]
print("Took %f seconds." % (time.time() - start))
```

上述代码的运行速度会非常慢，因为总共调用了 500 次的切片操作，这就是过度使用了。一个更好的做法是采用 `torch.unbind` 运算符在每次循环中将矩阵切片为一个向量的列表，如下所示：

```python
z = torch.zeros([10])
for x_i in torch.unbind(x):
    z += x_i
```

这个改进会提高一些速度（在作者的机器上是提高了大约30%）。

但正确的做法应该是采用 `torch.sum` 来一步实现累加的操作：

```python
z = torch.sum(x, dim=0)
```

这种实现速度就非常的快（在作者的机器上提高了100%的速度）。

其他重载的算数和逻辑运算符分别是：

```python
z = -x  # z = torch.neg(x)
z = x + y  # z = torch.add(x, y)
z = x - y
z = x * y  # z = torch.mul(x, y)
z = x / y  # z = torch.div(x, y)
z = x // y
z = x % y
z = x ** y  # z = torch.pow(x, y)
z = x @ y  # z = torch.matmul(x, y)
z = x > y
z = x >= y
z = x < y
z = x <= y
z = abs(x)  # z = torch.abs(x)
z = x & y
z = x | y
z = x ^ y  # z = torch.logical_xor(x, y)
z = ~x  # z = torch.logical_not(x)
z = x == y  # z = torch.eq(x, y)
z = x != y  # z = torch.ne(x, y)
```

还可以使用这些运算符的递增版本，比如 `x += y ` 和 `x **=2` 都是合法的。

另外，Python 并不允许重载 `and` 、`or` 和 `not` 三个关键词。



------

### 5. 采用 TorchScript 优化运行时间

PyTorch 优化了维度很大的张量的运算操作。**在 PyTorch 中对小张量进行太多的运算操作是非常低效的**。所以有可能的话，将计算操作都重写为**批次（batch）的形式**，可以减少消耗和提高性能。而如果没办法自己手动实现批次的运算操作，那么可以采用 `TorchScript` 来提升代码的性能。

`TorchScript` 是一个 Python 函数的子集，但经过了 PyTorch 的验证，PyTorch 可以通过其 `just in time(jtt)` 编译器来自动优化 `TorchScript` 代码，提高性能。

下面给出一个具体的例子。在机器学习应用中非常常见的操作就是 `batch gather` ，也就是 `output[i] = input[i, index[i]]`。其代码实现如下所示：

```python
import torch
def batch_gather(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)
```

通过 `torch.jit.script` 装饰器来使用 TorchScript 的代码：

```python
@torch.jit.script
def batch_gather_jit(tensor, indices):
    output = []
    for i in range(tensor.size(0)):
        output += [tensor[i][indices[i]]]
    return torch.stack(output)
```

这个做法可以提高 10% 的运算速度。

但更好的做法还是手动实现批次的运算操作，下面是一个向量化实现的代码例子，提高了 100 倍的速度：

```python
def batch_gather_vec(tensor, indices):
    shape = list(tensor.shape)
    flat_first = torch.reshape(
        tensor, [shape[0] * shape[1]] + shape[2:])
    offset = torch.reshape(
        torch.arange(shape[0]).cuda() * shape[1],
        [shape[0]] + [1] * (len(indices.shape) - 1))
    output = flat_first[indices + offset]
    return output
```



------

### 6. 构建高效的自定义数据加载类

上一节介绍了如何写出更加高效的 PyTorch 的代码，但为了让你的代码运行更快，将数据更加高效加载到内存中也是非常重要的。幸运的是 PyTorch 提供了一个很容易加载数据的工具，即 ` DataLoader` 。一个 ` DataLoader` 会采用多个 `workers` 来同时将数据从 `Dataset` 类中加载，并且可以选择使用 `Sampler` 类来对采样数据和组成 `batch` 形式的数据。

如果你可以随时访问你的数据，那么使用 ` DataLoader` 会非常简单：只需要继承 `Dataset` 类别并实现 `__getitem__` (读取每个数据)和 `__len__`（返回数据集的样本数量）这两个方法。下面给出一个代码例子，如何从给定的文件夹中加载图片数据：

```python
import glob
import os
import random
import cv2
import torch

class ImageDirectoryDataset(torch.utils.data.Dataset):
    def __init__(path, pattern):
        self.paths = list(glob.glob(os.path.join(path, pattern)))

    def __len__(self):
        return len(self.paths)

    def __item__(self):
        path = random.choice(paths)
        return cv2.imread(path, 1)
```

比如想将文件夹内所有的 `jpeg` 图片都加载，代码实现如下所示：

```python
dataloader = torch.utils.data.DataLoader(ImageDirectoryDataset("/data/imagenet/*.jpg"), num_workers=8)
for data in dataloader:
    # do something with data
```

这里采用了 8 个 `workers` 来并行的从硬盘中读取数据。这个数量可以根据实际使用机器来进行调试，得到一个最佳的数量。

当你的数据都很大或者你的硬盘读写速度很快，采用` DataLoader`进行随机读取数据是可行的。但也可能存在一种情况，就是使用的是一个很慢的连接速度的网络文件系统，请求单个文件的速度都非常的慢，而这可能就是整个训练过程中的瓶颈。

**一个更好的做法就是将数据保存为一个可以连续读取的连续文件格式**。例如，当你有非常大量的图片数据，可以采用 `tar` 命令将其压缩为一个文件，然后用 python 来从这个压缩文件中连续的读取图片。要实现这个操作，需要用到 PyTorch 的 `IterableDataset`。创建一个 `IterableDataset` 类，只需要实现 `__iter__` 方法即可。

下面给出代码实现的例子：

```python
import tarfile
import torch

def tar_image_iterator(path):
    tar = tarfile.open(self.path, "r")
    for tar_info in tar:
        file = tar.extractfile(tar_info)
        content = file.read()
        yield cv2.imdecode(content, 1)
        file.close()
        tar.members = []
    tar.close()

class TarImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def __iter__(self):
        yield from tar_image_iterator(self.path)
```

不过这个方法有一个问题，**当使用 `DataLoader` 以及多个 `workers` 读取这个数据集的时候，会得到很多重复的数据：**

```python
dataloader = torch.utils.data.DataLoader(TarImageDataset("/data/imagenet.tar"), num_workers=8)
for data in dataloader:
    # data contains duplicated items
```

这个问题主要是因为每个 `worker` 都会创建一个单独的数据集的实例，并且都是从数据集的起始位置开始读取数据。一种避免这个问题的办法就是不是压缩为一个`tar` 文件，而是将数据划分成 `num_workers` 个单独的 `tar` 文件，然后每个 `worker` 分别加载一个，代码实现如下所示：

```python
class TarImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # For simplicity we assume num_workers is equal to number of tar files
        if worker_info is None or worker_info.num_workers != len(self.paths):
            raise ValueError("Number of workers doesn't match number of files.")
        yield from tar_image_iterator(self.paths[worker_info.worker_id])
```

所以使用例子如下所示：

```python
dataloader = torch.utils.data.DataLoader(
    TarImageDataset(["/data/imagenet_part1.tar", "/data/imagenet_part2.tar"]), num_workers=2)
for data in dataloader:
    # do something with data
```

这是一种简单的避免重复数据的问题。而 `tfrecord` 则用了比较复杂的办法来共享数据，具体可以查看：

https://github.com/vahidk/tfrecord



------

### 7. PyTorch 的数值稳定性

当使用任意一个数值计算库，比如 NumPy 或者 PyTorch ，都需要知道一点，**编写数学上正确的代码不一定会得到正确的结果，你需要确保这个计算是稳定的。**

首先以一个简单的例子开始。从数学上来说，对任意的非零 `x` ，都可以知道式子 $x * y / y = x$ 是成立的。 但看看具体实现的时候，是不是总是正确的：

```python
import numpy as np

x = np.float32(1)

y = np.float32(1e-50)  # y would be stored as zero
z = x * y / y

print(z)  # prints nan
```

代码的运行结果是打印 `nan` ，原因是 `y` 的数值对于 `float32` 类型来说非常的小，这导致它的实际数值是 0 而不是 1e-50。

另一种极端情况就是 `y` 非常的大：

```python
y = np.float32(1e39)  # y would be stored as inf
z = x * y / y

print(z)  # prints nan
```

输出结果依然是 `nan` ，因为 `y` 太大而被存储为 `inf` 的情况，对于 `float32` 类型来说，其范围是 `1.4013e-45  ~ 3.40282e+38`，当超过这个范围，就会被置为 0 或者 inf。

下面是如何查看一种数据类型的数值范围：

```python
print(np.nextafter(np.float32(0), np.float32(1)))  # prints 1.4013e-45
print(np.finfo(np.float32).max)  # print 3.40282e+38
```

**为了让计算变得稳定，需要避免过大或者过小的数值**。这看起来很容易，但这类问题是很难进行调试，特别是在 PyTorch 中进行梯度下降的时候。这不仅因为需要确保在前向传播过程中的所有数值都在使用的数据类型的取值范围内，还要保证在反向传播中也做到这一点。

下面给出一个代码例子，计算一个输出向量的 softmax，一种不好的代码实现如下所示：

```python
import torch

def unstable_softmax(logits):
    exp = torch.exp(logits)
    return exp / torch.sum(exp)

print(unstable_softmax(torch.tensor([1000., 0.])).numpy())  # prints [ nan, 0.]
```

这里计算 `logits` 的指数数值可能会得到超出 `float32` 类型的取值范围，即过大或过小的数值，这里最大的 `logits` 数值是 `ln(3.40282e+38) = 88.7`，超过这个数值都会导致 `nan` 。

那么应该如何避免这种情况，做法很简单。因为有 $\frac{exp(x-c) }{\sum exp(x-c)} = \frac{exp(x)}{\sum exp(x)}$，也就是我们可以对 `logits` 减去一个常量，但结果保持不变，所以我们选择`logits` 的最大值作为这个常数，这种做法，指数函数的取值范围就会限制为 `[-inf, 0]` ，然后最终的结果就是 `[0.0, 1.0]` 的范围，代码实现如下所示：

```python
import torch

def softmax(logits):
    exp = torch.exp(logits - torch.reduce_max(logits))
    return exp / torch.sum(exp)

print(softmax(torch.tensor([1000., 0.])).numpy())  # prints [ 1., 0.]
```

接下来是一个更复杂点的例子。

假设现在有一个分类问题。我们采用 softmax 函数对输出值 `logits` 计算概率。接着定义采用预测值和标签的交叉熵作为损失函数。对于一个类别分布的交叉熵可以简单定义为 ：
$$
xe(p, q) = -\sum p_i log(q_i)
$$
所以有一个不好的实现交叉熵的代码实现为：

```python
def unstable_softmax_cross_entropy(labels, logits):
    logits = torch.log(softmax(logits))
    return -torch.sum(labels * logits)

labels = torch.tensor([0.5, 0.5])
logits = torch.tensor([1000., 0.])

xe = unstable_softmax_cross_entropy(labels, logits)

print(xe.numpy())  # prints inf
```

在上述代码实现中，当 softmax 结果趋向于 0，其 `log` 输出会趋向于无穷，这就导致计算结果的不稳定性。所以可以对其进行重写，将 `softmax` 维度拓展并做一些归一化的操作：

```python
def softmax_cross_entropy(labels, logits, dim=-1):
    scaled_logits = logits - torch.max(logits)
    normalized_logits = scaled_logits - torch.logsumexp(scaled_logits, dim)
    return -torch.sum(labels * normalized_logits)

labels = torch.tensor([0.5, 0.5])
logits = torch.tensor([1000., 0.])

xe = softmax_cross_entropy(labels, logits)

print(xe.numpy())  # prints 500.0
```

可以验证计算的梯度也是正确的：

```python
logits.requires_grad_(True)
xe = softmax_cross_entropy(labels, logits)
g = torch.autograd.grad(xe, logits)[0]
print(g.numpy())  # prints [0.5, -0.5]
```

这里需要再次提醒，进行梯度下降操作的时候需要额外的小心谨慎，需要确保每个网络层的函数和梯度的范围都在合法的范围内，指数函数和对数函数在不正确使用的时候都可能导致很大的问题，它们都能将非常小的数值转换为非常大的数值，或者从很大变为很小的数值。







































