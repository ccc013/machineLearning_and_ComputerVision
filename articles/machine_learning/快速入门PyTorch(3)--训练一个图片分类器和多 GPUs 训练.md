
> 2019 第 44 篇，总第 68 篇文章

快速入门 PyTorch 教程前两篇文章：

- [快速入门Pytorch(1)--安装、张量以及梯度](https://mp.weixin.qq.com/s/WZdBm2JQ4yKjISQmXN4TMg)
- [快速入门PyTorch(2)--如何构建一个神经网络](https://mp.weixin.qq.com/s/Q8tNXsDh6cdCnvLQ3yglZQ)

这是快速入门 PyTorch 的第三篇教程也是最后一篇教程，这次将会在 CIFAR10 数据集上简单训练一个图片分类器，将会简单实现一个分类器从网络定义、数据处理和加载到训练网络模型，最后测试模型性能的流程。以及如何使用多 GPUs 训练网络模型。

本文的目录如下：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_tutorial_content5.png)

---
### 4. 训练分类器

[上一节](https://mp.weixin.qq.com/s/Q8tNXsDh6cdCnvLQ3yglZQ)介绍了如何构建神经网络、计算 `loss` 和更新网络的权值参数，接下来需要做的就是实现一个图片分类器。

#### 4.1 训练数据

在训练分类器前，当然需要考虑数据的问题。通常在处理如图片、文本、语音或者视频数据的时候，一般都采用标准的 Python 库将其加载并转成 Numpy 数组，然后再转回为 PyTorch  的张量。

- 对于图像，可以采用 `Pillow, OpenCV` 库；
- 对于语音，有 `scipy` 和 `librosa`;
- 对于文本，可以选择原生 Python 或者 Cython 进行加载数据，或者使用 `NLTK` 和 `SpaCy` 。

PyTorch 对于计算机视觉，特别创建了一个 `torchvision` 的库，它包含一个数据加载器(data loader)，可以加载比较常见的数据集，比如 `Imagenet, CIFAR10, MNIST` 等等，然后还有一个用于图像的数据转换器(data transformers)，调用的库是 `torchvision.datasets` 和 `torch.utils.data.DataLoader` 。

在本教程中，将采用 `CIFAR10` 数据集，它包含 10 个类别，分别是飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。数据集中的图片都是 `3x32x32`。一些例子如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/cifar10.png)

#### 4.2 训练图片分类器

训练流程如下：

1. 通过调用 `torchvision` 加载和归一化 `CIFAR10` 训练集和测试集；
2. 构建一个卷积神经网络；
3. 定义一个损失函数；
4. 在训练集上训练网络；
5. 在测试集上测试网络性能。

##### 4.2.1 加载和归一化 CIFAR10

首先导入必须的包：

```python
import torch
import torchvision
import torchvision.transforms as transforms
```

`torchvision` 的数据集输出的图片都是 `PILImage` ，即取值范围是 `[0, 1]` ，这里需要做一个转换，变成取值范围是 `[-1, 1]` , 代码如下所示：

```python
# 将图片数据从 [0,1] 归一化为 [-1, 1] 的取值范围
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

这里下载好数据后，可以可视化部分训练图片，代码如下：

```python
import matplotlib.pyplot as plt
import numpy as np

# 展示图片的函数
def imshow(img):
    img = img / 2 + 0.5     # 非归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取训练集图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 展示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片类别标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

展示图片如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_tutorial_cifar10.png)

其类别标签为：

```
 frog plane   dog  ship
```

##### 4.2.2 构建一个卷积神经网络

这部分内容其实直接采用上一节定义的网络即可，除了修改 `conv1` 的输入通道，从 1 变为 3，因为这次接收的是 3 通道的彩色图片。

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

##### 4.2.3 定义损失函数和优化器

这里采用类别交叉熵函数和带有动量的 SGD 优化方法：

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

##### 4.2.4 训练网络

第四步自然就是开始训练网络，指定需要迭代的 epoch，然后输入数据，指定次数打印当前网络的信息，比如 `loss` 或者准确率等性能评价标准。

```python
import time
start = time.time()
for epoch in range(2):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data
        # 清空梯度缓存
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:
            # 每 2000 次迭代打印一次信息
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training! Total cost time: ', time.time()-start)
```

这里定义训练总共 2 个 epoch，训练信息如下，大概耗时为 77s。

```python
[1,  2000] loss: 2.226
[1,  4000] loss: 1.897
[1,  6000] loss: 1.725
[1,  8000] loss: 1.617
[1, 10000] loss: 1.524
[1, 12000] loss: 1.489
[2,  2000] loss: 1.407
[2,  4000] loss: 1.376
[2,  6000] loss: 1.354
[2,  8000] loss: 1.347
[2, 10000] loss: 1.324
[2, 12000] loss: 1.311

Finished Training! Total cost time:  77.24696755409241
```

##### 4.2.5 测试模型性能

训练好一个网络模型后，就需要用测试集进行测试，检验网络模型的泛化能力。对于图像分类任务来说，一般就是用准确率作为评价标准。

首先，我们先用一个 `batch` 的图片进行小小测试，这里 `batch=4` ，也就是 4 张图片，代码如下：

```python
dataiter = iter(testloader)
images, labels = dataiter.next()

# 打印图片
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

图片和标签分别如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/pytorch_tutorial_test.png)

```
GroundTruth:    cat  ship  ship plane
```

然后用这四张图片输入网络，看看网络的预测结果：

```python
# 网络输出
outputs = net(images)

# 预测结果
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
```

输出为：

```python
Predicted:    cat  ship  ship  ship
```

前面三张图片都预测正确了，第四张图片错误预测飞机为船。

接着，让我们看看在整个测试集上的准确率可以达到多少吧！

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
```

输出结果如下

```python
Accuracy of the network on the 10000 test images: 55 %
```

这里可能准确率并不一定一样，教程中的结果是 `51%` ，因为权重初始化问题，可能多少有些浮动，相比随机猜测 10 个类别的准确率(即 10%)，这个结果是不错的，当然实际上是非常不好，不过我们仅仅采用 5 层网络，而且仅仅作为教程的一个示例代码。

然后，还可以再进一步，查看每个类别的分类准确率，跟上述代码有所不同的是，计算准确率部分是 `c = (predicted == labels).squeeze()`，这段代码其实会根据预测和真实标签是否相等，输出 1 或者 0，表示真或者假，因此在计算当前类别正确预测数量时候直接相加，预测正确自然就是加 1，错误就是加 0，也就是没有变化。

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
```

输出结果，可以看到猫、鸟、鹿是错误率前三，即预测最不准确的三个类别，反倒是船和卡车最准确。

```python
Accuracy of plane : 58 %
Accuracy of   car : 59 %
Accuracy of  bird : 40 %
Accuracy of   cat : 33 %
Accuracy of  deer : 39 %
Accuracy of   dog : 60 %
Accuracy of  frog : 54 %
Accuracy of horse : 66 %
Accuracy of  ship : 70 %
Accuracy of truck : 72 %
```

#### 4.3 在 GPU 上训练

深度学习自然需要 GPU 来加快训练速度的。所以接下来介绍如果是在 GPU 上训练，应该如何实现。

首先，需要检查是否有可用的 GPU 来训练，代码如下：

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

输出结果如下，这表明你的第一块 GPU 显卡或者唯一的 GPU 显卡是空闲可用状态，否则会打印 `cpu` 。

```python
cuda:0
```

既然有可用的 GPU ，接下来就是在 GPU 上进行训练了，其中需要修改的代码如下，分别是需要将网络参数和数据都转移到 GPU 上：

```python
net.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```

修改后的训练部分代码：

```python
import time
# 在 GPU 上训练注意需要将网络和数据放到 GPU 上
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

start = time.time()
for epoch in range(2):
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 清空梯度缓存
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 == 1999:
            # 每 2000 次迭代打印一次信息
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training! Total cost time: ', time.time() - start)
```

注意，这里调用 `net.to(device)` 后，需要定义下优化器，即传入的是 CUDA 张量的网络参数。训练结果和之前的类似，而且其实因为这个网络非常小，转移到 GPU 上并不会有多大的速度提升，而且我的训练结果看来反而变慢了，也可能是因为我的笔记本的 GPU 显卡问题。

如果需要进一步提升速度，可以考虑采用多 GPUs，这里可以查看数据并行教程，这是一个可选内容。

https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

本小节教程：

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

本小节的代码：

https://github.com/ccc013/DeepLearning_Notes/blob/master/Pytorch/practise/train_classifier_example.ipynb


---
### 5. 数据并行

这部分教程将学习如何使用 `DataParallel` 来使用多个 GPUs 训练网络。

首先，在 GPU 上训练模型的做法很简单，如下代码所示，定义一个 `device` 对象，然后用 `.to()` 方法将网络模型参数放到指定的 GPU 上。

```python
device = torch.device("cuda:0")
model.to(device)
```

接着就是将所有的张量变量放到 GPU 上：

```python
mytensor = my_tensor.to(device)
```

注意，这里 `my_tensor.to(device)` 是返回一个 `my_tensor` 的新的拷贝对象，而不是直接修改 `my_tensor` 变量，因此你需要将其赋值给一个新的张量，然后使用这个张量。

Pytorch 默认只会采用一个 GPU，因此需要使用多个 GPU，需要采用 `DataParallel` ，代码如下所示：

```python
model = nn.DataParallel(model)
```

这代码也就是本节教程的关键，接下来会继续详细介绍。

#### 5.1 导入和参数

首先导入必须的库以及定义一些参数：

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

这里主要定义网络输入大小和输出大小，`batch` 以及图片的大小，并定义了一个 `device` 对象。

#### 5.2 构建一个假数据集

接着就是构建一个假的(随机)数据集。实现代码如下：

```python
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
```

#### 5.3 简单的模型

接下来构建一个简单的网络模型，仅仅包含一层全连接层的神经网络，加入 `print()` 函数用于监控网络输入和输出 `tensors` 的大小：

```python
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
```

#### 5.4 创建模型和数据平行

这是本节的核心部分。首先需要定义一个模型实例，并且检查是否拥有多个 GPUs，如果是就可以将模型包裹在 `nn.DataParallel` ，并调用 `model.to(device)` 。代码如下：

```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```

#### 5.5 运行模型

接着就可以运行模型，看看打印的信息：

```python
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

输出如下：

```python
In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

#### 5.6 运行结果

如果仅仅只有 1 个或者没有 GPU ，那么 `batch=30` 的时候，模型会得到输入输出的大小都是 30。但如果有多个 GPUs，那么结果如下：

##### 2 GPUs

```python
# on 2 GPUs
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

##### 3 GPUs

```python
Let's use 3 GPUs!
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

##### 8 GPUs

```python
Let's use 8 GPUs!
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

#### 5.7 总结

`DataParallel` 会自动分割数据集并发送任务给多个 GPUs 上的多个模型。然后等待每个模型都完成各自的工作后，它又会收集并融合结果，然后返回。

更详细的数据并行教程：

https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

本小节教程：

https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html


---
### 小结

第三篇主要是简单实现了一个图像分类的流程，选择数据集，构建网络模型，定义损失函数和优化方法，训练网络，测试网络性能，并检查每个类别的准确率，当然这只是很简单的过一遍流程。

然后就是使用多 GPUs 训练网络的操作。

接下来你可以选择：

- [训练一个神经网络来玩视频游戏](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [在 imagenet 上训练 ResNet](https://github.com/pytorch/examples/tree/master/imagenet)
- [采用 GAN 训练一个人脸生成器](https://github.com/pytorch/examples/tree/master/dcgan)
- [采用循环 LSTM 网络训练一个词语级别的语言模型](https://github.com/pytorch/examples/tree/master/word_language_model)
- [更多的例子](https://github.com/pytorch/examples)
- [更多的教程](https://pytorch.org/tutorials)
- [在 Forums 社区讨论 PyTorch](https://discuss.pytorch.org/)



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
