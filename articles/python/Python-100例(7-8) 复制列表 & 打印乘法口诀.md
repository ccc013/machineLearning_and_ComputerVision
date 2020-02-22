
> 2019 年第 27 篇，总 51 篇文章

前面分享的六道题目如下：

- [Python-100 | 练习题 01 & 列表推导式](https://mp.weixin.qq.com/s/qSUJKYjGLkGcswdBpA8KLg)
- [Python-100 练习题 02](https://mp.weixin.qq.com/s/w2pmPqp_dmPNFfoaZP95JQ)
- [Python-100 练习题 03 完全平方数](https://mp.weixin.qq.com/s/iHGr6vCJHgALPoHj8koy-A)
- [Python-100 练习题 04 判断天数](https://mp.weixin.qq.com/s/2hXJq1k-BTCcHAR1tG_o3w)
- [Python-100例(5-6) 排序&斐波那契数列](https://mp.weixin.qq.com/s/0MGLyYfBfHhhAyZ0r1RqHQ)

这次是分享 Python-100 例的第 7-8 题，分别是复制列表和打印乘法口诀，这两道题目都比较简单。


---
### Example-7 复制列表

> **题目**：将一个列表的数据复制到另一个列表

#### 思路

直接采用切片操作，即 `[:]`

#### 代码实现

这道题目比较简单，代码如下：

```python
print('original list: {}'.format(input_list))
copyed_list = input_list[:]
print('copyed_list: {}'.format(copyed_list))
```

输出结果如下：

```python
original list: [3, 2, '1', [1, 2]]
copyed_list: [3, 2, '1', [1, 2]]
```

这道题目只要知道列表的切片操作，就非常简单，当然如果不知道这个操作，也可以通过 for 循环来遍历实现复制的操作，就是没有这么简洁，一行代码搞定。

### Example-8 乘法口诀

> **题目**：输出 9*9 乘法口诀

#### 思路

最简单就是通过两层的 for 循环，两个参数，一个控制行，一个控制列，然后注意每行输出个数，即每层循环的起始和结束条件。

#### 代码实现

两种实现方法如下：

```python
# 第一种，for 循环实现
def multiplication_table1():
    for i in range(1, 10):
        for j in range(1, i + 1):
            print('%d*%d=%-2d ' % (i, j, i * j), end='')
        print('')


# 第二种，一行代码实现
def multiplication_table2():
    print('\n'.join([' '.join(['%s*%s=%-2s' % (y, x, x * y) for y in range(1, x + 1)]) for x in range(1, 10)]))
```

结果如下：

```
1*1=1 
1*2=2  2*2=4 
1*3=3  2*3=6  3*3=9 
1*4=4  2*4=8  3*4=12 4*4=16
1*5=5  2*5=10 3*5=15 4*5=20 5*5=25
1*6=6  2*6=12 3*6=18 4*6=24 5*6=30 6*6=36
1*7=7  2*7=14 3*7=21 4*7=28 5*7=35 6*7=42 7*7=49
1*8=8  2*8=16 3*8=24 4*8=32 5*8=40 6*8=48 7*8=56 8*8=64
1*9=9  2*9=18 3*9=27 4*9=36 5*9=45 6*9=54 7*9=63 8*9=72 9*9=81
```

练习代码已经上传到我的 GitHub 上了：

https://github.com/ccc013/CodesNotes/tree/master/Python_100_examples


---
### 小结

今天分享的两道题目就到这里，如果你有更好的解决方法，也可以后台留言，谢谢！

---


欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)


#### 往期精彩推荐

##### Python-100 练习系列

- [Python-100 | 练习题 01 & 列表推导式](https://mp.weixin.qq.com/s/qSUJKYjGLkGcswdBpA8KLg)
- [Python-100 练习题 02](https://mp.weixin.qq.com/s/w2pmPqp_dmPNFfoaZP95JQ)
- [Python-100 练习题 03 完全平方数](https://mp.weixin.qq.com/s/iHGr6vCJHgALPoHj8koy-A)
- [Python-100 练习题 04 判断天数](https://mp.weixin.qq.com/s/2hXJq1k-BTCcHAR1tG_o3w)
- [Python-100例(5-6) 排序&斐波那契数列](https://mp.weixin.qq.com/s/0MGLyYfBfHhhAyZ0r1RqHQ)

##### 机器学习系列

- [机器学习入门系列（1）--机器学习概览](https://mp.weixin.qq.com/s/r_UkF_Eys4dTKMH7DNJyTA)
- [机器学习入门系列(2)--如何构建一个完整的机器学习项目(一)](https://mp.weixin.qq.com/s/nMG5Z3CPdwhg4XQuMbNqbw)
- [机器学习数据集的获取和测试集的构建方法](https://mp.weixin.qq.com/s/HxGO7mhxeuXrloN61sDGmg)
- [特征工程之数据预处理（上）](https://mp.weixin.qq.com/s/BnTXjzHSb5-4s0O0WuZYlg)
- [特征工程之数据预处理（下）](https://mp.weixin.qq.com/s/Npy1-zrRmqETN8GydnIb8Q)
- [特征工程之特征缩放&特征编码](https://mp.weixin.qq.com/s/WYPUJbcT6UHvEFMJe8vteg)
- [特征工程(完)](https://mp.weixin.qq.com/s/0QkAOXg9nw8UwpnKuYdC-g)
- [常用机器学习算法汇总比较(上）](https://mp.weixin.qq.com/s/4Ban_TiMKYUBXTq4WcMr5g)
- [常用机器学习算法汇总比较(中）](https://mp.weixin.qq.com/s/ELQbsyxQtZYdtHVrfOFBFw)
- [常用机器学习算法汇总比较(完）](https://mp.weixin.qq.com/s/V2C4u9mSHmQdVl9ZYs1-FQ)

##### Github项目 & 资源教程推荐

- [[Github 项目推荐] 一个更好阅读和查找论文的网站](https://mp.weixin.qq.com/s/ImQcGt8guLKZawNLS-_HzA)
- [[资源分享] TensorFlow 官方中文版教程来了](https://mp.weixin.qq.com/s/Si1YaYLfhL1upbjQkvireQ)
- [必读的AI和深度学习博客](https://mp.weixin.qq.com/s/0J2raJqiYsYPqwAV1MALaw)
- [[教程]一份简单易懂的 TensorFlow 教程](https://mp.weixin.qq.com/s/vXIM6Ttw37yzhVB_CvXmCA)
- [[资源]推荐一些Python书籍和教程，入门和进阶的都有！](https://mp.weixin.qq.com/s/jkIQTjM9C3fDvM1c6HwcQg)
- [[Github项目推荐] 机器学习& Python 知识点速查表](https://mp.weixin.qq.com/s/kn2DUJHL48UyuoUEhcfuxw)





