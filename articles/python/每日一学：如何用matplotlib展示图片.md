### 前言

今天简单介绍如何通过 matplotlib 展示图片，分为以下几种情况：

1. 直接用 matplotlib 读取图片，然后展示图片；
2. 采用 opencv 读取图片，然后用 matplotlib 来展示图片；
3. 采用 PIL 读取图片，然后用 matplotlib 来展示图片。

首先是需要安装需要的库，主要是 `opencv` 、 `matplotlib`、`Pillow`  两个库：

```shell
pip install opencv-python matplotlib Pillow
```

此外，在 `jupyter` 中运行代码。

另外，本次代码例子中展示所用的图片为：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/matplotlib_show_img_0.png)

代码和图片都上传到 GitHub 上了：

https://github.com/ccc013/CodesNotes/blob/master/PythonNotes/matplotlib_notes.ipynb

### 1. matplotlib 读取并展示图片

首先是导入需要的库：

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
```

第一行就是导入用于展示图片的函数 `matplotlib` 的 `pyplot` ，第二行则是用于读取图片的 `image` ，第三行是因为在 `jupyter` 中用 `matplotlit` 展示图片需要加入的一行代码。

接下来就是读取并展示图片，如下所示：

```python
# 采用 matplotlib 展示图片
image = mpimg.imread('plane.jpg')
plt.imshow(image)
```

结果如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/matplotlib_show_img_01.png)

这里我们发现展示的图片，出现了坐标轴，可以通过添加一行代码，来关闭坐标轴：

```python
plt.axis('off')
```

结果如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/matplotlib_show_img_02.png)

### 2. 展示通过 opencv 读取的图片

不过，对于图像库，使用更多的还是 `opencv` ，所以如何通过 `matplotlib` 展示 `opencv` 读取的图片呢？

代码其实很简单，如下所示：

```python
import cv2
image = cv2.imread("plane.jpg")
plt.imshow(image)
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/matplotlib_show_img_03.png)

但这里发现展示的图片颜色不对，和原图出现了很大的区别，这是为什么呢？

原因其实是 `opencv` 对于 RGB 图片是将其表示为一个多维的 `NumPy` 的多维数组，但排列顺序是反序的，也就是BGR 的顺序，因此这里需要对通道顺序进行调整，代码应该这么修改：

```python
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/matplotlib_show_img_04.png)

通过进行通道的转换后，再次展示图片，就能显示原图了。

### 3. 展示通过 PIL 读取的图片

另外一个非常常用的图像处理库就是 PIL 了，这里展示的代码也很简单，如下所示：

```python
# 展示 PIL 读取的图片
from PIL import Image
image = Image.open('plane.jpg')
plt.imshow(image)
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/matplotlib_show_img_05.png)



### 小结

今天简单介绍了如何通过 `matplotlib` 来展示图片，分别是三种情况，直接用 `matplotlib` 读取图片，用 `opencv` 读取图片，用 `PIL` 读取图片，其中需要注意的是 `opencv` 读取图片的情况，因为其对于 RGB 通道的排列是反序的。











