原文：https://medium.com/@karan_jakhar/keras-vs-pytorch-dilemma-dc434e5b5ae0

作者：Karan Jakhar

### 前言

> 上一篇[2020年计算机视觉学习指南](https://mp.weixin.qq.com/s/px_Chm-iQTfTkaw96rGFrA) 介绍了两种深度学习框架--Keras 和 PyTorch ，这篇文章的作者就对这两个框架进行了对比，分别通过实现一个简单的模型来对比两个不同的代码风格，最后还给出了他的个人建议。

当你决定开始学习深度学习，那么应该选择使用什么工具呢？目前有很多深度学习的框架或者库，但本文会对比两个框架，Keras 和 PyTorch ，这是两个非常好开始使用的框架，并且它们都有一个很低的学习曲线，初学者可以很快就学会它们，因此在本文，我将分享一个办法来解决如何选择其中一个框架进行使用。

**最好的办法就是查看两个框架各自的代码风格**。设计任何方案的前提和最重要的事情就是你的工具，当你开始一个项目前必须安装配置好你的工具，并且一旦开始项目后，就不应该更改时用的工具。它会影响到你的生产力。作为一个初学者，你应该尽量尝试不同的工具，并且找到合适你的，但如果你正在参加一个非常正式的项目工作，那么这些事情都应该提早计划好。

每天都会有新的框架和工具面世，对你最好的工具应该是在个性化和抽象做好平衡的，它应该可以同步你的思考和代码风格，那么如何找到这样合适的工具呢，**答案就是你需要尝试不同的工具**。

接下来，让我们分别用 Keras 和 PyTorch 训练一个简单的模型吧。如果你是深度学习的初学者，那么不要担心理解不了某些名词概念，目前你只需要关注这两个框架的代码风格，并思考哪个才是最合适你的，也就是让你感觉更舒适并且更容易上手的。

这两个框架的主要不同点是 PyTorch 默认是 `eager` 模式，而 Keras 是在 TensorFlow 和其他框架的基础上进行工作，但目前主要是基于 TensorFlow 框架的，因此其默认是图(`graph` )模式。当然，最新版本的 TensorFlow 也提供了和 PyTorch 一样的 `eager` 模式。如果你对 NumPy 很熟悉的话，你可以把 PyTorch 看作是有 GPU 支持的 NumPy 。此外，也有不少类似 Keras 一样的第三方库作为高级 API 接口，它们使用 PyTorch 作为后端支持，比如 `Fastai`(提供了免费的很好的课程)、`Lightning`, `Ignite` 等等。也可以去了解这些框架，如果你发现它们很有趣，那你就多了一个理由使用 PyTorch 。

这两种框架都有不同的方法来实现一个模型。这里都分别选择了一种简单的实现方式。下面是分别在谷歌的 Colab 上实现的代码的链接，打开链接并运行代码，这更加有助于找到更合适你的框架：

Keras: https://colab.research.google.com/drive/1QH6VOY_uOqZ6wjxP0K8anBAXmI0AwQCm?usp=sharing#forceEdit=true&sandboxMode=true

PyTorch: https://colab.research.google.com/drive/1irYr0byhK6XZrImiY4nt9wX0fRp3c9mx?usp=sharing#scrollTo=FoKO0mEScvXi&forceEdit=true&sandboxMode=true

本文并不会介绍太细节的东西，因为我们的目标只是对两个框架的代码结构和风格进行查看和了解。

------

### 基于 Keras 的模型实现

下面是实现数字识别的代码实现。代码非常容易理解，你最好在 colab 中查看并且进行实验，至少要开始运行起来。

```python
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

img_rows, img_cols = 28, 28
num_classes = 10
batch_size = 128
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train/255
x_test  = x_test/255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
```

在 Keras 中有一些作为样例的数据集，其中一个就是 MNIST 手写数字数据集，上述代码主要是实现加载数据集的功能，图片是 NumPy 的数组格式。另外，上述代码也做了一点的图像处理来将数据可以应用到模型中。

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
     activation='relu',
     input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])
```

上述代码就是模型的代码实现。在 Keras(TensorFlow) 中，我们需要先定义想使用的所有东西，然后它们会只运行一次。我们不能对它们进行实验，但是在 PyTorch 中是可以做到的。

```python
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("test_model.h5")

# load the model
from keras.models import load_model
model = load_model("test_model.h5")

# predict digit
prediction = model.predict(gray)
print(prediction.argmax())
```

上述代码就是训练和验证模型，可以使用 `save()` 方法来保存模型，然后通过 `load_model()` 方法来加载保存的模型文件，`predict()` 方法是用于对测试数据进行预测得到预测结果。

这就是使用 Keras 简单实现一个模型的概览，下面看看 PyTorch 是怎么实现模型的吧。



------

### 基于 PyTorch 的模型实现

研究者主要用 PyTorch ，因为它的灵活性以及偏实验的代码风格，这包括可以对 PyTorch 的一切都进行修改调整，对 也就是可以完全控制一切，进行实验也是非常容易。在 PyTorch 中，不需要先定义所有的事情再运行，对每个单独的步骤的测试都非常容易。因此，它比 Keras 更容易调试。

下面也是利用 PyTorch 实现一个简单的数字识别模型。

```python
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
```

上述代码主要是导入需要的库以及定义了一些变量，这些变量如 `n_epochs, momentum` 等都是必须设置的超参数，但这里不会详细展开说明，因为我们也说过本文的目标是理解框架的代码结构和风格。

```python
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape
```

这段代码则是声明了一个数据加载器用于加载训练数据集进行训练和测试。数据集有多种下载数据的方法，这和框架没有关系。当然上面这段代码对于深度学习的初学者可能是有些复杂了。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
```

接下来这段代码就是定义模型。这是一个很通用的创建一个网络模型的方法，定义一个类继承 `nn.Module`，`forward()`方法是实现网络的前向传播。PyTorch 的实现是非常直接，并且可以根据需要进行修改。

```python
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'model.pth')
      torch.save(optimizer.state_dict(), 'optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
```

接下来这段代码，我们分别定义了训练和测试函数，`train()` 和 `test()`。在 Keras 中直接调用 `fit()` 函数，然后所有事情都给我们实现好了，但是在 PyTorch 中我们需要手动实现这些步骤。当然，在一些高级 API 库，比如 `Fastai` 里将这部分也变得很简单，减少了需要的代码量。

```python
#loading the model
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)
network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)
```

最后就是保存和加载模型用于再次训练或者进行预测的代码。PyTorch 的模型文件通常是以 `pt` 或者`pth` 为后缀名。



------

### 个人的建议

当你开始学习一个模型，并且理解它的理念后，从一个框架转移到另一个并不困难，这只需要几天的工作。**我的建议就是两个框架都要学习，但不需要学得非常深入。你应该选择一个框架并开始实现你的模型代码，但同时也需要对另一个框架有所了解。这有助于你阅读用另一个框架实现的模型代码。**你不应该被框架所约束，它们都是很好的框架。

我最初开始使用的是 Keras，但现在我在工作中使用 PyTorch，因为它可以更好的进行实验。我喜欢 PyTorch 的 python 风格。所以首先使用一个你觉得更适合你的框架，然后同时也尝试去学习另一个框架，如果学习后发现它使用更加舒适，那就改为使用它，并且这两个框架的核心概念都是非常相似的，两者的相互转换都非常容易。

最后祝你在深度学习之旅中好运。你应该更专注算法的理论概念以及它们在现实生活中如何使用和实现的。

最后再次给出两份模型代码实现的 colab 链接：

- PyTorch: https://colab.research.google.com/drive/1irYr0byhK6XZrImiY4nt9wX0fRp3c9mx?usp=sharing
- Keras: https://colab.research.google.com/drive/1QH6VOY_uOqZ6wjxP0K8anBAXmI0AwQCm?usp=sharing



































