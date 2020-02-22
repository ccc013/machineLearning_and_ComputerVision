
> 2019 年第 49 篇文章，总第 73 篇文章


我们在采用 `dict` 的时候，一般都需要判断键是否存在，如果不存在，设置一个默认值，存在则采取其他的操作，但这个做法其实需要多写几行代码，那么是否有更高效的写法，可以减少代码，但可读性又不会降低呢，毕竟作为程序员，我们都希望写出可用并且高效简洁的代码。

今天看到一篇文章，作者介绍可以使用 `defaultdict` 和 `Counter` 来代替 `dictionary` 可以写出比更加简洁和可读性高的代码，因此今天就简单翻译这篇文章，并后续简单介绍这两种数据类型。

文章链接

https://towardsdatascience.com/python-pro-tip-start-using-python-defaultdict-and-counter-in-place-of-dictionary-d1922513f747

关于字典的介绍，也可以查看我之前写的[Python基础入门_2基础语法和变量类型](https://mp.weixin.qq.com/s/Cw1TyTLKP_6271Sgx6gIBA)。

本文目录：

- Counter 和 defaultdict
- 为何要用 defaultdict 呢？
- defaultdict 的定义和使用
- Counter 的定义和使用

---

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/0_c3GbEX7Wao3oHm_0.jpg)

学习一门编程语言很简单，在学习一门新语言的时候，我会专注于以下顺序的知识点，并且开始用新语言写代码其实很简单：

- **运算符和数据类型**：+，-，int，float，str
- **条件语句**：if，else，case，switch
- **循环语句**：For，while
- **数据结构**：List，Array，Dict，Hashmaps
- **定义函数**

但能写代码和写出优雅高效的代码是两件事情，每种语言都有其独特的地方。

因此，一门编程语言的新手总是会写出比较过度的代码，比如，对于 Java 开发者，在学习 Python 后，要写一段实现对一组数字的求和代码，会是下面这样子：

```
x=[1,2,3,4,5]
sum_x = 0
for i in range(len(x)):
    sum_x+=x[i]
```

但对于一名 Python 老手来说，他的代码是：

```
sum_x = sum(x)
```

所以接下来会开启一个名为“Python Shorts”的系列文章，主要介绍一些 Python 提供的简单概念以及有用的技巧和使用例子，这个系列的目标就是写出高效并且可读的代码。

#### Counter 和 defaultdict

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/0_pxVbW7V5iXP4VBfU.jpg)

这里的要代码实现的例子就是统计一段文字中单词出现的次数，比如《哈姆雷特》，应该如何做呢？

Python 中可以有多种实现方法，但只有一种是比较优雅的，那就是采用原生 Python 的实现--`dict` 数据类型。

代码如下所示：


```
# count the number of word occurrences in a piece of text
text = "I need to count the number of word occurrences in a piece of text. How could I do that? " \
       "Python provides us with multiple ways to do the same thing. But only one way I find beautiful."

word_count_dict = {}
for w in text.split(" "):
    if w in word_count_dict:
        word_count_dict[w] += 1
    else:
        word_count_dict[w] = 1
```

这里还可以应用 `defaultdict` 来减少代码行数：

```
from collections import defaultdict
word_count_dict = defaultdict(int)
for w in text.split(" "):
    word_count_dict[w] += 1
```

利用 `Counter` 也可以做到：

```
from collections import Counter
word_count_dict = Counter()
for w in text.split(" "):
    word_count_dict[w] += 1
```

`Counter` 还有另一种写法，更加简洁：

```
word_counter = Counter(text.split(" "))
```

`Counter` 其实就是一个计数器，它本身就应用于统计给定的变量对象的次数，因此，我们还可以获取出现次数最多的单词：


```
print('most common word: ', word_count_dict.most_common(10))
```
输出如下：

```
most common word:  [('I', 3), ('the', 2), ('of', 2), ('do', 2), ('to', 2), ('multiple', 1), ('in', 1), ('way', 1), ('us', 1), ('occurrences', 1)]
```

其他的一些应用例子：

```
# Count Characters
print(Counter('abccccccddddd'))  
# Count List elements
print(Counter([1, 2, 3, 4, 5, 1, 2]))  
```

输出结果：

```
Counter({'c': 6, 'd': 5, 'a': 1, 'b': 1})
Counter({1: 2, 2: 2, 3: 1, 4: 1, 5: 1})
```

#### 为何要用 defaultdict 呢？

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/fruits.jpg)

既然 `Counter` 这么好用，是不是只需要 `Counter` 就可以了？答案当然是否定的，因为 `Counter` 的问题就是其数值必须是整数，本身就是用于统计数量，因此如果我们需要的数值是字符串，列表或者元组，那么就不能继续用它。

这个时候，`defaultdict` 就派上用场了。它相比于 `dict` 的最大区别就是可以设置默认的数值，即便 `key` 不存在。例子如下：


```
s = [('color', 'blue'), ('color', 'orange'), ('color', 'yellow'), ('fruit', 'banana'), ('fruit', 'orange'),
     ('fruit', 'banana')]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)
print(d)  
```
输出结果：

```
defaultdict(<class 'list'>, {'color': ['blue', 'orange', 'yellow'], 'fruit': ['banana', 'orange', 'banana']})
```
这里就是事先将字典的所有数值都初始化一个空列表，而如果是传入集合 `set`：

```
s = [('color', 'blue'), ('color', 'orange'), ('color', 'yellow'), ('fruit', 'banana'), ('fruit', 'orange'),
     ('fruit', 'banana')]
d = defaultdict(set)
for k, v in s:
    d[k].add(v)
print(d)
```
输出结果：

```
defaultdict(<class 'set'>, {'color': {'blue', 'yellow', 'orange'}, 'fruit': {'banana', 'orange'}})
```

这里需要注意的就是列表和集合的添加元素方法不相同，列表是`list.append()`，而集合是`set.add()`。


---

接着是补充下，这两个数据类型的一些定义和方法，主要是参考官方文档的解释。

#### defaultdict 的定义和使用

关于 `defaultdict`，在官方文档的介绍有：

> class collections.defaultdict([default_factory[, ...]])

> 返回一个新的类似字典的对象。 defaultdict 是内置 dict 类的子类。它重载了一个方法并添加了一个可写的实例变量。其余的功能与 dict 类相同，此处不再重复说明。

> 第一个参数 default_factory 提供了一个初始值。它默认为 None 。所有的其他参数都等同与 dict 构建器中的参数对待，包括关键词参数。

在 `dict` 有一个方法`setdefault()`，利用它也可以实现比较简洁的代码：

```
s = [('color', 'blue'), ('color', 'orange'), ('color', 'yellow'), ('fruit', 'banana'), ('fruit', 'orange'),('fruit', 'banana')]
a = dict()
for k, v in s:
    a.setdefault(k, []).append(v)
print(a)
```

但官方文档也说了，`defaultdict` 的实现要比这种方法更加快速和简单。


#### Counter 的定义和使用

中文官方文档的说明：

> class collections.Counter([iterable-or-mapping])

> 一个 Counter 是一个 dict 的子类，用于计数可哈希对象。它是一个集合，元素像字典键(key)一样存储，它们的计数存储为值。计数可以是任何整数值，包括0和负数。 Counter 类有点像其他语言中的 bags或multisets。

这里，应该不只是可哈希对象，还有可迭代对象，否则列表属于不可哈希对象，是否可哈希，其实是看该数据类型是否实现了 `__hash__` 方法：

```
a = (2, 1)
a.__hash__()
```
输出：

```
3713082714465905806
```
而列表：

```
b=[1,2]
b.__hash__()
```
报错：

```
TypeError: 'NoneType' object is not callable
```

当然，之前也提过，调用`hash()` 方法，也同样可以判断一个数据类型是否可哈希，而可哈希的数据类型都是不可变的数据类型。

对于 `Counter` ，还可以通过关键字来初始化：

```
c = Counter(cats=4, dogs=8)
print(c)
```
输出：

```
Counter({'dogs': 8, 'cats': 4})
```

`Counter` 的一些方法，除了上述介绍的`most_common()`外，还有：

- `elements()`：返回一个迭代器，将所有出现元素按照其次数来重复 `n` 个，并且返回任意顺序，但如果该元素统计的次数少于 1 ，则会忽略，例子如下：


```
c = Counter(a=4, b=2, c=0, d=-2)
sorted(c.elements())
# ['a', 'a', 'a', 'a', 'b', 'b']
```

- `subtract()`:减法操作，输入输出可以是 0 或者负数

```
c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
c.subtract(d)
print(c)
# Counter({'a': 3, 'b': 0, 'c': -3, 'd': -6})
```

此外，还有以下这些方法：


```
# 求和
sum(c.values())                
# 清空 Counter
c.clear()                      
# 转换为 列表
list(c)                         
# 转换为 集合
set(c)                          
# 转换为 字典
dict(c)                        
# 键值对
c.items()                       
# 
Counter(dict(list_of_pairs))    
# 输出 n 个最少次数的元素
c.most_common()[:-n-1:-1]       
# 返回非零正数
+c      
# 返回负数
-c
```

此外，也可以采用运算符`+,-,&,|`，各有各不同的实现作用：

```
c = Counter(a=3, b=1)
d = Counter(a=1, b=2)
# 加法操作 c[x] + d[x]
print(c + d)    # Counter({'a': 4, 'b': 3})                 
# 减法，仅保留正数
print(c - d )   # Counter({'a': 2})                 
# 交集:  min(c[x], d[x]) 
print(c & d)    # Counter({'a': 1, 'b': 1})             
# 并集:  max(c[x], d[x])
print(c | d)    # Counter({'a': 3, 'b': 2})
```

---

参考：
- [collections--- 容器数据类型](https://docs.python.org/zh-cn/3/library/collections.html)
- [What does “hashable” mean in Python?](https://stackoverflow.com/questions/14535730/what-does-hashable-mean-in-python)


---
#### 小结

如果需要进行计数，比如计算单词出现次数，采用 `Counter` 是一个不错的选择，非常简洁，可读性也高；而如果需要保存的数据不是整数，并且都是统一的某个类型，比如都是列表，那么直接采用 `defaultdict` 来定义一个变量对象，会比用 `dict` 的选择更好。

最后，本文的代码例子已经上传到 Github 上了：

https://github.com/ccc013/Python_Notes/blob/master/Python_tips/defaultdict_and_counter.ipynb


欢迎关注我的微信公众号--**算法猿的成长**，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_0601.png)

**如果觉得不错，在看、转发就是对小编的一个支持！**

