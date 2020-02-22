

原文链接：https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/

作者：Adrian Rosebrock

今天介绍的是基于 Keras 实现多标签图像分类，主要分为四个部分：

- 介绍采用的多标签数据集
- 简单介绍使用的网络模型 `SmallerVGGNet`，一个简化版的 `VGGNet`
- 实现 `SmallerVGGNet` 模型并训练
- 利用训练好的模型，对测试样例进行分类测试

接下来就开始本文的内容。

------

#### 1. 多标签图像数据集

我们将采用如下所示的多标签图像数据集，一个服饰图片数据集，总共是 2167 张图片，六大类别：

- 黑色牛仔裤(Black Jeans, 344张)
- 蓝色连衣裙(Blue Dress，386张)
- 蓝色牛仔裤(Blue Jeans, 356张)
- 蓝色衬衫(Blue Shirt, 369张)
- 红色连衣裙(Red Dress，380张)
- 红色衬衫(Red Shirt，332张)

因此我们的 CNN 网络模型的目标就是同时预测衣服的颜色以及类型。



![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/keras_multi_label_dataset.jpg)

关于如何收集和建立这个数据集，可以参考这篇文章：

https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/

这篇文章会介绍如何采用微软的 Bing 服务接口进行图片下载，然后删除不相关的图片。

#### 2. 多标签分类项目结构

整个多标签分类的项目结构如下所示：

```
├── classify.py
├── dataset
│   ├── black_jeans [344 entries
│   ├── blue_dress [386 entries]
│   ├── blue_jeans [356 entries]
│   ├── blue_shirt [369 entries]
│   ├── red_dress [380 entries]
│   └── red_shirt [332 entries]
├── examples
│   ├── example_01.jpg
│   ├── example_02.jpg
│   ├── example_03.jpg
│   ├── example_04.jpg
│   ├── example_05.jpg
│   ├── example_06.jpg
│   └── example_07.jpg
├── fashion.model
├── mlb.pickle
├── plot.png
├── pyimagesearch
│   ├── __init__.py
│   └── smallervggnet.py
├── search_bing_api.py
└── train.py
```

简单介绍每份代码和每个文件夹的功能作用：

- `search_bing_api.py` ：主要是图片下载，但本文会提供好数据集，所以可以不需要运行该代码；
- `train.py` ：最主要的代码，处理和加载数据以及训练模型；
- `fashion.model` ：保存的模型文件，用于 `classify.py` 进行对测试图片的分类；
- `mlb.pickle`：由 `scikit-learn` 模块的 `MultiLabelBinarizer` 序列化的文件，将所有类别名字保存为一个序列化的数据结构形式
- `plot.png` ：绘制训练过程的准确率、损失随训练时间变化的图
- `classify.py` ：对新的图片进行测试

三个文件夹：

- `dataset`：数据集文件夹，包含六个子文件夹，分别对应六个类别
- `pyimagesearch` ：主要包含建立 Keras 的模型代码文件--`smallervggnet.py` 
- `examples`：7张测试图片

#### 3. 基于 Keras 建立的网络结构

本文采用的是一个简化版本的 `VGGNet`，`VGGNet` 是 2014 年由 Simonyan 和 Zisserman 提出的，论文--[*Very Deep Convolutional Networks for Large Scale Image Recognition*](https://arxiv.org/pdf/1409.1556/)。

这里先来展示下 `SmallerVGGNet` 的实现代码，首先是加载需要的 Keras 的模块和方法：

```python
# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
```

接着开始定义网络模型--`SmallerVGGNet` 类，它包含 `build` 方法用于建立网络，接收 5 个参数，`width, height, depth` 就是图片的宽、高和通道数量，然后 `classes` 是数据集的类别数量，最后一个参数 `finalAct` 表示输出层的激活函数，注意一般的图像分类采用的是 `softmax` 激活函数，但是**多标签图像分类需要采用 `sigmoid`** 。

```python
class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
 
		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
```

接着，就开始建立网络模型了，总共是 5 层的卷积层，最后加上一个全连接层和输出层，其中卷积层部分可以说是分为三个部分，每一部分都是基础的卷积层、RELU 层、BatchNormalization 层，最后是一个最大池化层(MaxPoolingLayer)以及 Dropout 层。

```python
		# CONV => RELU => POOL
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
 
		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
         # first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
 
		# use a *softmax* activation for single-label classification
		# and *sigmoid* activation for multi-label classification
		model.add(Dense(classes))
		model.add(Activation(finalAct))
 
		# return the constructed network architecture
		return model
```

#### 4. 实现网络模型以及训练

现在已经搭建好我们的网络模型`SmallerVGGNet` 了，接下来就是 `train.py` 这份代码，也就是实现训练模型的代码。

首先，同样是导入必须的模块，主要是 `keras` ，其次还有绘图相关的 `matplotlib`、`cv2`，处理数据和标签的 `sklearn` 、`pickle` 等。

```python
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
```

注意，这里需要提前安装的第三方模块包括 `Keras, scikit-learn, matplotlib, imutils, OpenCV`，安装命令如下：

```shell
pip install keras, scikit-learn, matplotlib, imutils, opencv-python
```

当然，还需要安装 `tensorflow` ，如果仅仅采用 CPU 版本，可以直接 `pip install tensorflow` ，而如果希望采用 GPU ，那就需要安装 CUDA，具体教程可以看看如下教程:

https://www.pyimagesearch.com/2017/09/27/setting-up-ubuntu-16-04-cuda-gpu-for-deep-learning-with-python/

接着，继续设置命令行参数：

```python
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
```

这里主要是四个参数：

- `--dataset`: 数据集路径
- `--model` : 保存的模型路径
- `--labelbin` : 保存的多标签二进制对象路径
- `--plot` : 保存绘制的训练准确率和损失图

然后，设置一些重要的参数，包括训练的总次数 `EPOCHS` 、初始学习率  `INIT_LR`、批大小 `BS`、输入图片大小 `IMAGE_DIMS` ：

```python
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
```

然后就开始数据处理部分的代码，首先是加载数据的代码：

```python
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
 
# initialize the data and labels
data = []
labels = []
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
 
	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
```

这部分代码，首先是将所有数据集的路径都保存到 `imagePaths` 中，接着进行 `shuffle` 随机打乱操作，然后循环读取图片，对图片做尺寸调整操作，并处理标签，得到 `data` 和 `labels` 两个列表，其中处理标签部分的实现结果如下所示：

```shell
$ python
>>> import os
>>> labels = []
>>> imagePath = "dataset/red_dress/long_dress_from_macys_red.png"
>>> l = label = imagePath.split(os.path.sep)[-2].split("_")
>>> l
['red', 'dress']
>>> labels.append(l)
>>>
>>> imagePath = "dataset/blue_jeans/stylish_blue_jeans_from_your_favorite_store.png"
>>> l = label = imagePath.split(os.path.sep)[-2].split("_")
>>> labels.append(l)
>>>
>>> imagePath = "dataset/red_shirt/red_shirt_from_target.png"
>>> l = label = imagePath.split(os.path.sep)[-2].split("_")
>>> labels.append(l)
>>>
>>> labels
[['red', 'dress'], ['blue', 'jeans'], ['red', 'shirt']]
```

因此，`labels` 就是一个嵌套列表的列表，每个子列表都包含两个元素。

然后就是数据的预处理，包括转换为 `numpy` 的数组，对数据进行归一化操作，以及采用 `scikit-learn` 的方法 `MultiLabelBinarizer` 将标签进行 `One-hot` 编码操作：

```python
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
 
# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))
```

同样，这里也看看对标签进行 `One-hot` 编码操作的结果是怎样的：

```shell
$ python
>>> from sklearn.preprocessing import MultiLabelBinarizer
>>> labels = [
...     ("blue", "jeans"),
...     ("blue", "dress"),
...     ("red", "dress"),
...     ("red", "shirt"),
...     ("blue", "shirt"),
...     ("black", "jeans")
... ]
>>> mlb = MultiLabelBinarizer()
>>> mlb.fit(labels)
MultiLabelBinarizer(classes=None, sparse_output=False)
>>> mlb.classes_
array(['black', 'blue', 'dress', 'jeans', 'red', 'shirt'], dtype=object)
>>> mlb.transform([("red", "dress")])
array([[0, 0, 1, 0, 1, 0]])
```

`MultiLabelBinarizer` 会先统计所有类别的数量，然后按顺序排列，即对每个标签分配好其位置，然后进行映射得到 `One-hot` 变量，如上所示，总管六个类别，依次是 `'black', 'blue', 'dress', 'jeans', 'red', 'shirt'`，而 `red` 和 `dress` 分别是第 5 和 3 个位置，所以得到的 `One-hot` 变量是 `[0, 0, 1, 0, 1, 0]`

数据处理最后一步，划分训练集和测试集，以及采用 `keras` 的数据增强方法 `ImageDataGenerator` ：

```python
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)
 
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
```

训练集和测试集采用`scikit-learn` 的方法 `train_test_split` ，按照比例 8:2 划分。

然后就是初始化模型对象、优化方法，开始训练：

```python
# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")
 
# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
```

这里采用的是 `Adam` 优化方法，损失函数是 `binary cross-entropy` 而非图像分类常用的 `categorical cross-entropy`，原因主要是多标签分类的目标是将每个输出的标签作为一个独立的伯努利分布，并且希望单独惩罚每一个输出节点。

最后就是保存模型，绘制曲线图的代码了：

```python
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
 
# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
```

训练代码写好了，那么就可以开始进行训练了，训练的命令如下：

```shell
$ python train.py --dataset dataset --model fashion.model \
	--labelbin mlb.pickle
```

训练的部分记录如下所示：

```shell
Using TensorFlow backend.
[INFO] loading images...
[INFO] data matrix: 2165 images (467.64MB)
[INFO] class labels:
1. black
2. blue
3. dress
4. jeans
5. red
6. shirt
[INFO] compiling model...
[INFO] training network...
Epoch 1/75
name: GeForce GTX TITAN X
54/54 [==============================] - 4s - loss: 0.3503 - acc: 0.8682 - val_loss: 0.9417 - val_acc: 0.6520
Epoch 2/75
54/54 [==============================] - 2s - loss: 0.1833 - acc: 0.9324 - val_loss: 0.7770 - val_acc: 0.5377
Epoch 3/75
54/54 [==============================] - 2s - loss: 0.1736 - acc: 0.9378 - val_loss: 1.1532 - val_acc: 0.6436
...
Epoch 73/75
54/54 [==============================] - 2s - loss: 0.0534 - acc: 0.9813 - val_loss: 0.0324 - val_acc: 0.9888
Epoch 74/75
54/54 [==============================] - 2s - loss: 0.0518 - acc: 0.9833 - val_loss: 0.0645 - val_acc: 0.9784
Epoch 75/75
54/54 [==============================] - 2s - loss: 0.0405 - acc: 0.9857 - val_loss: 0.0429 - val_acc: 0.9842
[INFO] serializing network...
[INFO] serializing label binarizer...
```

在训练结束后，训练集和测试集上的准确率分别是 `98.57%` 和 `98.42` ，绘制的训练损失和准确率折线图图如下所示，上方是训练集和测试集的准确率变化曲线，下方则是训练集和测试集的损失图，从这看出，训练的网络模型并没有遭遇明显的过拟合或者欠拟合问题。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/plot.png)

#### 5. 测试网络模型

训练好模型后，就是测试新的图片了，首先先完成代码 `classify.py` ，代码如下：

```python
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)
 
# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())
 
# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:2]

# loop over the indexes of the high confidence class labels
for (i, j) in enumerate(idxs):
	# build the label and draw the label on the image
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
	print("{}: {:.2f}%".format(label, p * 100))
 
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
```

这部分代码也不难，主要是加载图片和模型，然后进行预测，得到结果会输出前两个概率最大的结果，然后用 `OpenCV` 展示出来，调用的命令如下：

```shell
$ python classify.py --model fashion.model --labelbin mlb.pickle \
	--image examples/example_01.jpg
```

实验结果如下，给出的预测结果是红色连衣裙，展示出来的也的确是红色连衣裙的图片。

```shell
Using TensorFlow backend.
[INFO] loading network...
[INFO] classifying image...
black: 0.00%
blue: 3.58%
dress: 95.14%
jeans: 0.00%
red: 100.00%
shirt: 64.02%
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/keras_multi_label_output_01-702x1024.png)

其他的样例图片都可以通过相同的命令，只需要修改输入图片的名字即可，然后就是其中最后一张图片，是比较特殊的，输入命令如下所示：

```shell
$ python classify.py --model fashion.model --labelbin mlb.pickle \
	--image examples/example_07.jpg
Using TensorFlow backend.
[INFO] loading network...
[INFO] classifying image...
black: 91.28%
blue: 7.70%
dress: 5.48%
jeans: 71.87%
red: 0.00%
shirt: 5.92%
```

展示的结果，这是一条黑色连衣裙，但预测结果给出黑色牛仔裤的结果。

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/keras_multi_label_output_07-655x1024.png)

这里的主要原因就是黑色连衣裙并不在我们的训练集类别中。这其实也是目前图像分类的一个问题，无法预测未知的类别，因为训练集并不包含这个类别，因此 CNN 没有见过，也就预测不出来。

#### 6. 小结

本文介绍了如何采用 Keras 实现多标签图像分类，主要的两个关键点：

1. 输出层采用 `sigmoid` 激活函数，而非 `softmax` 激活函数；
2. 损失函数采用 `binary cross-entropy` ，而非 `categorical cross-entropy`。











