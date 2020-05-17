整理自知乎问题：**CNN 模型所需的计算力（flops）和参数（parameters）数量是怎么计算的？**

链接：https://www.zhihu.com/question/65305385

首先是给出两个定义：

- **FLOPS**：全大写，是floating point operations per second的缩写，**意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标**。

- **FLOPs**：注意 s 小写，是floating point operations的缩写（s表复数），**意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。**

这里说的自然是第二种 FLOPs，计算量，也就是模型的复杂度。

### 卷积层的 FLOPs

不考虑激活函数，对于单个输入特征图的计算公式为（没有考虑 batch ）:
$$
(2 \times C_i \times K^2 - 1)\times H\times W\times C_o
$$
这里每个参数的含义：$C_i$ 是输入通道数量， K 表示卷积核的大小，H 和 W 是输出特征图(feature map)的大小，$C_o$ 是输出通道。

因为是乘法和加法，所以括号内是 2 ，表示两次运算操作。另外，不考虑 bias 的时候，有-1，而考虑 bias 的时候是没有 -1。

对于括号内的理解是这样的：

$2*C_i*K^2-1 = (C_i * K^2) + (C_i*K^2-1)$，第一项是乘法的运算数量，第二项是加法运算数量，因为 n 个数相加，是执行 n-1 次的加法次数，如果考虑 bias，就刚好是 n 次，也就是变成 $2*C_i*K^2$

对于整个公式来说就是分两步计算：

1. 括号内是计算得到输出特征图的一个像素的数值；
2. 括号外则是乘以整个输出特征图的大小，拓展到整个特征图。

举个例子，如下图所示是一个输出特征图的计算，其中输入特征图是 5*5 ，卷积核是 3 * 3，输出的特征图大小也是 3 * 3，所以这里对应公式中的参数，就是 K=3, H=W=3, 假设输入和输出通道数量都是 1，那么下图得到右边的特征图的一个像素的数值的计算量就是 (3*3)次乘法+（3 * 3-1）次加法 = 17，然后得到整个输出特征图的计算量就是 17 * 9 = 153.

![此处输入图片的描述](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/convolution_schematic.gif)

### 全连接层的 FLOPs

计算公式为：
$$
（2 \times I-1)\times O
$$
每个参数的含义：I 是输入数量，O 是输出数量。

同样 2 也是表示乘法和加法，然后不考虑 bias 是 -1，考虑的时候没有 -1。

对于这个公式也是和卷积层的一样，括号内考虑一个输出神经元的计算量，然后扩展到所有的输出神经元。



### 相关实现代码库

GitHub 上有几个实现计算模型的 FLOPs 的库：

- https://github.com/Lyken17/pytorch-OpCounter
- https://github.com/sagartesla/flops-cnn
- https://github.com/sovrasov/flops-counter.pytorch

非常简单的代码实现例子，来自 https://github.com/sagartesla/flops-cnn/blob/master/flops_calculation.py

```python
input_shape = (3 ,300 ,300) # Format:(channels, rows,cols)
conv_filter = (64 ,3 ,3 ,3)  # Format: (num_filters, channels, rows, cols)
stride = 1
padding = 1
activation = 'relu'

if conv_filter[1] == 0:
    n = conv_filter[2] * conv_filter[3] # vector_length
else:
    n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length

flops_per_instance = n + ( n -1)    # general defination for number of flops (n: multiplications and n-1: additions)

num_instances_per_filter = (( input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # for rows
num_instances_per_filter *= ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # multiplying with cols

flops_per_filter = num_instances_per_filter * flops_per_instance
total_flops_per_layer = flops_per_filter * conv_filter[0]  # multiply with number of filters

if activation == 'relu':
    # Here one can add number of flops required
    # Relu takes 1 comparison and 1 multiplication
    # Assuming for Relu: number of flops equal to length of input vector
    total_flops_per_layer += conv_filter[0] * input_shape[1] * input_shape[2]


if total_flops_per_layer / 1e9 > 1:   # for Giga Flops
    print(total_flops_per_layer/ 1e9 ,'{}'.format('GFlops'))
else:
    print(total_flops_per_layer / 1e6 ,'{}'.format('MFlops'))
```

