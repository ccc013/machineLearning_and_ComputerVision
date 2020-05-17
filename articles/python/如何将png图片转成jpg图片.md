### 背景

最近在工作中会遇到需要将 png 图片转换为 jpg 图片的需求，主要原因也是 png 图片占的空间太大，如果图片数量上万张，可能就需要十几G的存储空间，所以希望转换为更小的 jpg 图片。

当然，这里并不是直接修改图片后缀为 jpg 即可，这样直接粗暴的转换可能会对图片质量有所损失，包括背景颜色会出现问题；

### 解决思路

实际上要将 png 图片转换为 jpg 格式的图片，其实就是从 4 通道的 png 转换为 3通道的 jpg 格式，也就是能保留原始的 RGB 三通道，只是去掉第四个通道的 alpha 通道，也就是我们需要将 RGB 通道的像素部分提取出来，然后贴到一个空白的新图片上，再保存为 jpg 图片即可。

### 代码实现

这里使用的是 Pillow 库来进行转换，然后这里需要注意不同模式的图片，处理方式还是有所不同的。

这里简单介绍，通过 Pillow 打开的图片，有以下几种模式：

- 1：1位像素，表示黑和白，但是存储的时候每个像素存储为8bit。
- L：8位像素，表示黑和白。
- P：8位像素，使用调色板映射到其他模式。
- RGB：3x8位像素，为真彩色。
- RGBA：4x8位像素，有透明通道的真彩色。
- CMYK：4x8位像素，颜色分离。
- YCbCr：3x8位像素，彩色视频格式。
- I：32位整型像素。
- F：32位浮点型像素。

通过 `mode` 即可查看图片的模式。

这里介绍比较常见的几种模式转换为 jpg 的方法，首先是 `L` 模式的转换：

```python
from PIL import Image
im = Image.open("test.png")
bg = Image.new("RGB", im.size, (255,255,255))
bg.paste(im,im)
bg.save("test.jpg")
```

然后是 `RGBA`  和 `P`  模式的转换，其转换方法也是一样的：

```python
from PIL import Image
img_pil = Image.open('test.png').convert('RGBA')
x, y = img_pil.size
p = Image.new('RGBA', img_pil.size, (255, 255, 255))
p.paste(img_pil, (0, 0, x, y), img_pil)
p = p.convert("RGB")
p.save('test.jpg')
```

下面给出一个代码例子：

首先是导入需要的库：

```python
import os
from PIL import Image
%matplotlib inline
import matplotlib.pyplot as plt
```

接着读取图片：

```python
# 原始的 png 图片
ori_img = 'plane.png'
# 读取图片
img_png = Image.open(ori_img)
print(img_png.mode, img_png.size)
plt.imshow(img_png)
```

开始转换：

```python
# 转 jpg
img_pil = img_png.convert('RGBA')
x, y = img_pil.size
img_jpg = Image.new('RGBA', img_pil.size, (255, 255, 255))
img_jpg.paste(img_pil, (0, 0, x, y), img_pil)
img_jpg = img_jpg.convert("RGB")
print(img_jpg.mode, img_jpg.size)
plt.imshow(img_jpg)
```

保存图片：

```python
img_jpg.save('plane.jpg')
```



代码输出结果如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/%E5%A6%82%E4%BD%95%E5%B0%86png%E5%9B%BE%E7%89%87%E8%BD%AC%E6%88%90jpg%E5%9B%BE%E7%89%87_example.png )

通过这种操作，原本是 128kb 的 png 图片转换为 38kb 左右的 jpg 图片，减少了接近 4 倍的存储空间；

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/%E5%A6%82%E4%BD%95%E5%B0%86png%E5%9B%BE%E7%89%87%E8%BD%AC%E6%88%90jpg%E5%9B%BE%E7%89%87_example2.png)

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/%E5%A6%82%E4%BD%95%E5%B0%86png%E5%9B%BE%E7%89%87%E8%BD%AC%E6%88%90jpg%E5%9B%BE%E7%89%87_example3.png)

所以如果对图片质量要求不高，可以接受一定的质量损失，可以将图片保存为 jpg 格式进行保存，这样可以保存更多数量的图片。



参考：

- http://www.voidcn.com/article/p-rbpllhah-btp.html



### 小结

这只是一种解决 png 转换为 jpg 图片的方法，是从采用 Pillow 库的代码实现方法，如果是采用其他的图片库，比如 opencv 等，也有相应的解决方法，这里大家可以自己探索一下，网上应该也是有相应的解决方法的。

















