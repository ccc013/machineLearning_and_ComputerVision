**原题** | 10 Python Pandas tricks that make your work more efficient

**作者** | Shiu-Tang Li

**原文** | https://towardsdatascience.com/10-python-pandas-tricks-that-make-your-work-more-efficient-2e8e483808ba

#### 导读

Pandas 是一个广泛应用于数据分析等领域的 Python 库。关于它的教程有很多，但这里会一些比较冷门但是非常有用的技巧。

#### read_csv

这是一个大家都应该知道的函数，因为它就是读取 csv 文件的方法。

但如果需要读取数据量很大的时候，可以添加一个参数--`nrows=5`，来先加载少量数据，这可以避免使用错误的分隔符，因为并不是所有的都采用逗号分隔，然后再加载整个数据集。

Ps. 在 Linux 的终端，可以采用 `head` 命令来查看文件的前 5 行数据，命令示例如下所示：

```shell
head -n 5 data.txt
```

加载数据后，可以通过方法`df.columns.tolist()`获取所有的列名字，再采用参数`usecols=['c1','c2',...]` 来读取真正需要的列。如果想读取速度更快并且知道一些列的数据类型，可以使用参数 `dtype={'c1':str, 'c2':int,...}`，使用这个参数的另一个好处是对于包含不同类型的列，比如同时包含字符串和整型的列，这个参数可以指定该列就是字符串或者整型的类型，避免在采用该列作为键进行融合不同表的时候出现错误。



#### Select_dtypes

如果必须用 Python 进行数据预处理，采用这个方法可以节省一些时间。在读取表后，默认数据类型可以能是 `bool, int64, float64, object, category, timedelta64, datetime64`，首先可以用下面的方法来查看分布情况和知道 `dataframe` 中包含哪些数据类型：

`df.dtypes.value_counts()`

接着使用下面的方法来选择特定类型的数据，比如说数字特征：

`df.select_dtypes(include=['float64', 'int64'])`



#### copy

这个方法很重要，首先先看看下面这个例子：

```python
import pandas as pd
df1 = pd.DataFrame({ 'a':[0,0,0], 'b': [1,1,1]})
df2 = df1
df2['a'] = df2['a'] + 1
df1.head()
```

运行上述代码后，会发现`df1` 的数值被改变了，这是因为 `df2=df1` 这段代码并不是对 `df1` 进行拷贝，然后赋给 `df2`，而是设置了一个指向 `df1` 的指针。 因此任何对 `df2` 的改变都会改变 `df1`，如果要修改这个问题，可以采用下面的代码：

```python
df2 = df1.copy()
```

或者

```python
from copy import deepcopy
df2 = deepcopy(df1)
```

#### map

这是一个非常酷的命令，可以用于做简单的数据转化操作。首先需要定义一个字典，它的键是旧数值，而其值是新的数值，如下所示：

```python
level_map = {1: 'high', 2: 'medium', 3: 'low'}
df['c_level'] = df['c'].map(level_map)
```

还有一些例子：

- 布尔值的 True，False 转化为 1，0
- 定义层次
- 用户定义的词典编码



#### apply or not apply

如果我们想创建一个新的采用其他列作为输入的列，`apply` 方法是一个非常有用的方法：

```python
def rule(x, y):
    if x == 'high' and y > 10:
         return 1
    else:
         return 0
df = pd.DataFrame({ 'c1':[ 'high' ,'high', 'low', 'low'], 'c2': [0, 23, 17, 4]})
df['new'] = df.apply(lambda x: rule(x['c1'], x['c2']), axis =  1)
df.head()
```

上面这段代码我们先定义了一个两个输入参数的方法，然后采用`apply` 方法将其应用到 `df` 的两列 `c1, c2`。

`apply` 的问题是有时候速度太慢了。如果是希望计算 `c1` 和 `c2` 两列的最大值，可以这么写：

```python
df['maximum'] = df.apply(lambda x: max(x['c1'], x['c2']), axis = 1)
```

但你会发现比下面这段代码要慢很多：

```python
df['maximum'] = df[['c1','c2']].max(axis=1)
```

**要点**：如果可以采用其他内置函数实现的工作，就不要采用`apply` 方法啦。比如，想对列`c` 的数值进行取舍为整数值，可以采用方法 `round(df['c'], o)` 或者 `df['c'].round(o)`，而不是使用`apply` 方法的代码：`df.apply(lambda x: round(x['c'], 0), axis=1)`

#### value_counts

这个方法用于检查数值的分布情况。比如，你想知道`c`列的每个唯一数值出现的频繁次数和可能的数值，可以如下所示：

```python
df['c'].value_counts()
```

这里还有一些有趣的技巧或者参数：

1. **normalize=True**：如果想看频率而不是次数，可以使用这个参数设置；
2. **dropna=False**：查看包含缺失值的统计
3. `df['c'].value_counts().reset_index()`：如果想对这个统计转换为一个 `dataframe` 并对其进行操作
4. `df['c'].value_counts().reset_index().sort_values(by='index')` 或者是 `df['c'].value_counts().sort_index()` : 实现根据列的每个取值对统计表进行排序

#### number of missing values

当构建模型的时候，我们希望可以删除掉带有太多缺失值的行，或者都是缺失值的行。这可以通过采用`.isnull()` 和 `.sum()` 来计算特定列的缺失值数量：

```python
import pandas as pd
import numpy as np
df = pd.DataFrame({ 'id': [1,2,3], 'c1':[0,0,np.nan], 'c2': [np.nan,1,1]})
df = df[['id', 'c1', 'c2']]
df['num_nulls'] = df[['c1', 'c2']].isnull().sum(axis=1)
df.head()
```

#### select rows with specific IDs

在 SQL 中这个操作可以通过`SELECT * FROM … WHERE ID in (‘A001’, ‘C022’, …) `来获取特定 IDs 的记录。而在 pandas 中，可以如下所示：

```python
df_filter = df['ID'].isin(['A001','C022',...])
df[df_filter]
```

#### Percentile groups

假设有一个都是数值类型的列，然后希望对这些数值划分成几个组，比如前 5% 是第一组，5-20%是第二组，20%-50%是第三组，最后的50%是第四组。这可以采用`.cut` 方法，但这有另外一个选择：

```python
import numpy as np
cut_points = [np.percentile(df['c'], i) for i in [50, 80, 95]]
df['group'] = 1
for i in range(3):
    df['group'] = df['group'] + (df['c'] < cut_points[i])
# or <= cut_points[i]
```

这个方法的速度非常快。

#### to_csv

最后是一个非常常用的方法，保存为 `csv` 文件。这里也有两个小技巧：

第一个就是`print(df[:5].to_csv())`，这段代码可以打印前5行，并且也是会保存到文件的数据。

另一个技巧是处理混合了整数和缺失值的情况。当某一列同时有缺失值和整数，其数据类型是 `float` 类型而不是 `int` 类型。所以在导出该表的时候，可以添加参数`float_format='%.of'` 来将 `float` 类型转换为整数。如果只是想得到整数，那么可以去掉这段代码中的 `.o` 









































