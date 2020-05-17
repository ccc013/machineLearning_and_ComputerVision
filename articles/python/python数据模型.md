最近开始阅读《流畅的python》，也会开始更新这本书的学习笔记

第一篇的内容是第一章 python 数据模型，主要是介绍 python 类中的特殊方法（或者说魔术方法），这类特殊方法的实现可以让我们自定义的类对象能够使用 python 标准库的方法，同时也有助于接口方法的一致性。

本文的代码例子：
https://github.com/ccc013/CodesNotes/blob/master/FluentPython/1_Python%E6%95%B0%E6%8D%AE%E6%A8%A1%E5%9E%8B.ipynb

------

### 前言

**数据模型其实是对 Python 框架的描述，它规范了这门语言自身构建模块的接口**，这些模块包括但不限于序列、迭代器、函数、类和上下文管理器。

通常在不同框架下写程序，都需要花时间来实现那些会被框架调用的方法，python 当然也包含这些方法，**当 python 解释器碰到特殊的句法的时候，会使用特殊方法来激活一些基本的对象操作**，这种特殊方法，也叫做魔术方法（magic method），通常以两个下划线开头和结尾，比如最常见的 `__init__`, `__len__` 以及 `__getitem__` 等，而 `obj[key]` 这样的操作背后的特殊方法是 `__getitem__`，初始化一个类示例的时候，如 `obj= Obj()` 的操作背后，特殊方法就是 `__init__`。

通过实现 python 的这些特殊方法，可以让自定义的对象实现和支持下面的操作：

- 迭代
- 集合类
- 属性访问
- 运算符重载
- 函数和方法的调用
- 对象的创建和销毁
- 字符串表示形式和格式化
- 管理上下文（也就是 with 块）

### 一摞 Python 风格的纸牌

接下来尝试自定义一个类，并实现两个特殊方法：`__getitem__` 和 `__len__` ，看看实现它们后，可以对自定义的类示例实现哪些操作。

这里自定义一个纸牌类，并定义了数字和花色，代码如下所示：

```python
import collections
# 用 nametuple 构建一个类来表示纸牌
Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
    
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]
```

其中辅助用到 `collections` 库的 `nametuple` ，用来表示一张纸牌，其属性包括数字 `rank` 和 花色 `suit` ，下面是对这个 `Card` 的简单测试：

```python
# 测试 Card
beer_card = Card('7', 'diamonds')
beer_card
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_1_card.png)

接着就是测试自定义的 `FrenchDeck` 类，这里会调用 `len()` 方法看看一摞纸牌有多少张：

```python
# 测试 FrenchDeck
deck = FrenchDeck()
len(deck)
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_1_card2.png)

然后是进行索引访问的操作，这里测试从正序访问第一张，以及最后一张纸牌的操作：

```python
print(deck[0], deck[-1])
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_1_card3.png)

如果想进行随机抽取卡牌，可以结合 `random.choice` 来实现：

```python
# 随机抽取，结合 random.choice
from random import choice

choice(deck)
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_1_card4.png)

由于我们实现 `__getitem__`  方法是获取纸牌，所以也可以支持切片（slicing）的操作，例子如下所示：

```python
# 切片
print(deck[:3])
print(deck[12::13])
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_1_card5.png)

另外，实现 `__getitem__` 方法就可以支持迭代操作：

```python
# 可迭代的读取
for card in deck:
    print(card)
    
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_1_card6_1.png)

反向迭代也自然可以做到：

```python
# 反向迭代
for card in reversed(deck):
    print(card)
    break
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_1_card7.png)

另外，当然也可以自定义排序规则，如下所示：

```python
# 制定排序的规则
suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]
 
# 对卡牌进行升序排序
for card in sorted(deck, key=spades_high):
    print(card)
```

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_1_card8.png)



总结一下，实现 python 的特殊方法的好处包括：

- 统一方面的名称，如果有别人采用你自定义的类，不用花更多精力记住不同的名称，比如获取数量都是 `len()` 方法，而不会是 `size` 或者 `length` 
- 可以更加方便利用 python 的各种标准库，比如 `random.choice` 、`reversed`、`sorted` ，不需要自己重新发明轮子



### 如何使用特殊方法

这里分两种情况来说明对于特殊方法的调用：

1. **python 内置的类型**：比如列表（list）、字典（dict）等，那么 CPython 会抄近路，即 `__len__` 实际上会直接返回 `PyVarObject` 里的 `ob_size` 属性。 `PyVarObject` **是表示内存中长度可变的内置对象的 C 语言结构体，直接读取这个值比调用一个方法要快很多**。
2. **自定义的类**：通过内置函数（如 `len, iter, str` 等）调用特殊方法是最好的选择。

对于特殊方法的调用，这里还要补充说明几点：

- 特殊方法的存在是为了被 Python 解释器调用的。我们不需要调用它们，即不需要这么写 `my_object.__len__()`，而应该是 `len(my_object)`，这里的 `my_object` 表示一个自定义类的对象。

- **通常对于特殊方法的调用都是隐式的**。比如 `for i in x` 循环语句是用 `iter(x)` ，也就是调用 `x.__iter__()`  方法。

- 除非有大量元编程存在，否则都不需要直接使用特殊方法；



接下来是实现一个自定义的二维向量类，然后自定义加号的特殊方法，实现运算符重载。

代码例子如下所示：

```python
# 一个简单的二维向量类
from math import hypot

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return 'Vector(%r, %r)' % (self.x, self.y)
    
    def __abs__(self):
        return hypot(self.x, self.y)
    
    def __bool__(self):
        return bool(abs(self))
        
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
```

这里除了必须实现的 `__init__`外，还实现了几个特殊方法：

- `__add__`: 加法运算符；
- `__bool__` ：用于判断是否真假，也就是在调用`bool（）` 方法；默认情况下是自定义类的实例总是被认为是真的，但如果实现了 `__bool__`或者 `__len__` ，则会返回它们的结果，`bool()`首先尝试返回 `__bool__` 方法，如果没有实现，则会尝试调用 `__len__` 方法
- `__mul__` ：实现的是标量乘法，即向量和数的乘法；
- `__abs__` ：如果输入是整数或者浮点数，返回输入值的绝对值；如果输入的是复数，返回这个复数的模；如果是输入向量，返回的是它的模；
- `__repr__` : 可以将对象用字符串的形式表达出来；

这里要简单介绍下 `__repr__` 和 `__str__` 两个方法的区别：

- `__repr__` ：交互式控制台、调试程序(debugger)、`%` 和 `str.format` 方法都会调用这个方法来获取字符串形式；
- `__str__` ：主要是在 `str()` 和 `print()` 方法中会调用该方法，它返回的字符串会对终端用户更加友好；
- 如果只想实现其中一个方法，`__repr__`  是更好的选择，因为默认会调用 `__repr__` 方法。

接下来就是简单测试这个类，测试结果如下所示：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/1_2_vector.png)



### 特殊方法一览

下面分别根据是否和运算符相关分为两类的特殊方法：

#### 和运算符无关的特殊方法

|          类别           |                            方法名                            |
| :---------------------: | :----------------------------------------------------------: |
| 字符串/字节序列表现形式 |           `__repr__, __str__,__format__,__bytes__`           |
|        数值转换         | `__abs__,__bool__,__complex__,__int__,__float__,__hash__,__index__` |
|        集合模拟         |  `__len__,__getitem__,__setitem__,__delitem__,__contains__`  |
|        迭代枚举         |              ` __iter__,__reversed__,__next__ `              |
|       可调用模拟        |                          `__call__`                          |
|       上下文管理        |                    `__enter__, __exit__`                     |
|     实例创建和销毁      |                  `__new__,__init__,__del__`                  |
|        属性管理         | `__getattr__,__getattribute__,__setattr__,__delattr__,__dir__` |
|       属性描述符        |                 `__get__,__set__,__delete__`                 |
|     跟类相关的服务      |      `__prepare__,__instancecheck__,__subclasscheck__`       |



#### 和运算符相关的特殊方法

|        类别        |                     方法名和对应的运算符                     |
| :----------------: | :----------------------------------------------------------: |
|     一元运算符     |             `__neg__ -, __pos__ +,__abs__ abs()`             |
|   众多比较运算符   | `__lt__ <, __le__ <=, __eq__ ==, __ne__ !=, __gt__ >, __ge__ >=` |
|     算术运算符     | `__add__ +, __sub__ -, __mul__ *, __truediv__ /, __floordiv__ //, __mod__ %, __divmod__ divmod(), __pow__ **或者pow(), __round__ round()` |
|   反向算法运算符   | `__radd__, __rsub__, __rmul__, __rtruediv__, __rfloordiv__, __rmod__, __rdivmod__, __rpow__` |
| 增量赋值算术运算符 | `__iadd__, __isub__, __imul__, __itruediv__, __ifloordiv__, __imod__, __ipow__` |
|      位运算符      | `__invert__ ~, __lshift__ <<, __rshift__ >>, __and__ &, __or__ |, __xor__ ^` |
|    反向位运算符    |   `__rlshift__, __rrshift__, __rand__, __rxor__, __ror__`    |
|  增量赋值位运算符  |   `__ilshift__, __irshift__, __iand__, __ixor__, __ior__`    |

这里有两类运算符要解释一下：

- **反向运算符**：交换两个操作数的位置的时候会调用反向运算符，比如 `b * a` 而不是 `a * b` ；
- **增量赋值运算符**：把一种中缀运算符变成赋值运算的捷径，即是 `a *= b` 的操作



### 为什么 len 不是普通方法

`len` 之所以不是普通方法，是为了让 Python 自带的数据结构变得高效，前面也提到内置类型在使用 `len` 方法的时候，CPython 会直接从一个 C 结构体里读取对象的长度，完全不会调用任何方法，因此速度会非常快。而在 python 的内置类型，比如列表 list、字符串 str、字典 dict 等查询数量是非常常见的操作。

这种处理方式实际上是在保持内置类型的效率和保证语言的一致性之间找到一个平衡点。



------

### 小结

本文介绍了两个代码例子，说明了在自定义类的时候，实现特殊方法，可以实现和内置类型（比如列表、字典、字符串等）一样的操作，包括实现迭代、运算符重载、打印类实例对象等，然后还根据是否和运算符相关将特殊方法分为两类，并列举出来了，最后也介绍了 `len` 方法的例子来说明 python 团队是如何保持内置类型的效率和保证语言一致性的。

