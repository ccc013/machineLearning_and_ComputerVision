
> 总第 116 篇文章，本文大约 1000 字，阅读大约需要 3 分钟

今天介绍和列表相关的两个小技巧：

- 碾平列表（flatten list），也就是列表里的元素也带有列表的情况；
- 列表去重，保留原始顺序和不保留顺序的做法


---

### 1. 碾平列表

**碾平列表（flatten list ）**，即当列表里面嵌套列表，如何将这些子列表给取出来，得到一个不包含子列表的列表，示例如下：

```python
list1 = [1, [2, [3,4]], 5]

=>new_list = [1, 2, 3, 4, 5]
```

这里介绍 3 种方法，分别如下。

**方法1：利用递归的思想**，代码如下：

```python
list1 = [1, [2, [3,4]], 5]
res = []

def fun(s):
    for i in s:
        if isinstance(i, list):
            fun(i)
        else:
            res.append(i)

fun(list1)
print(res)
```

接着是两种比较高级的写法，用 `lambda` 实现一个匿名函数

**方法2：**

```python
flat = lambda L: sum(map(flat, L), []) if isinstance(L, list) else [L]

print(flat(list1))
```

**方法3：**

```python
a = [1, 2, [3, 4], [[5, 6], [7, 8]]]

flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

print(flatten(a))
```


---

### 2. 列表去重

列表去重可能会破坏原有的顺序，所以下面分别介绍保留顺序和不保留顺序的做法。

#### 去重，但改变顺序

去重但改变顺序，两种方法

方法1 就是利用 `set` 进行去重

```
l1 = ['b','c','d','b','c','a','a']
l2 = list(set(l1))
print l2
```

方法2 是利用字典的键不重复的特性，将列表的元素作为一个字典的键，然后返回这个字典的所有键，即可实现去重的操作。

```
l1 = ['b','c','d','b','c','a','a']
l2 = {}.fromkeys(l1).keys()
print l2
```

#### 去重，不改变顺序

利用 `sorted` 和 `set` 方法实现去重并保留原始顺序，这里 `sorted` 指定排序的规则就是按照原列表的索引顺序

```
l1 = ['b','c','d','b','c','a','a']
l2 = sorted(set(l1),key=l1.index)
print l2
```

---

欢迎关注我的微信公众号--**算法猿的成长**，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_0601.png)

**如果觉得不错，在看、转发就是对小编的一个支持！**


