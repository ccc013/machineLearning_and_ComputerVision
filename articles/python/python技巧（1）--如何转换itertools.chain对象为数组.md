
> 总第 115 篇文章，本文大约 900 字，阅读大约需要 3 分钟

之前做1月总结的时候说过希望每天或者每2天开始的更新一些学习笔记，这是开始的第一篇。

这篇介绍的是如何把一个 `itertools.chain` 对象转换为一个数组。

参考 stackoverflow 上的一个回答：Get an array back from an itertools.chain object，链接如下：

https://stackoverflow.com/questions/26853860/get-an-array-back-from-an-itertools-chain-object

例子：

```python
list_of_numbers = [[1, 2], [3], []]
import itertools
chain = itertools.chain(*list_of_numbers)
```

解决方法有两种：

第一种比较简单，直接采用 `list` 方法，如下所示：

```python
list(chain)
```

但缺点有两个：

- 会在外层多嵌套一个列表
- 效率并不高


第二个就是利用 `numpy` 库的方法 `np.fromiter`，示例如下：

```python
>>> import numpy as np
>>> from itertools import chain
>>> list_of_numbers = [[1, 2], [3], []]
>>> np.fromiter(chain(*list_of_numbers), dtype=int)
array([1, 2, 3])

```

对比两种方法的运算时间，如下所示：


```
>>> list_of_numbers = [[1, 2]*1000, [3]*1000, []]*1000
>>> %timeit np.fromiter(chain(*list_of_numbers), dtype=int)
10 loops, best of 3: 103 ms per loop
>>> %timeit np.array(list(chain(*list_of_numbers)))
1 loops, best of 3: 199 ms per loop
```

可以看到采用 `numpy` 方法的运算速度会更快。

---

欢迎关注我的微信公众号--**算法猿的成长**，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_0601.png)

**如果觉得不错，在看、转发就是对小编的一个支持！**

