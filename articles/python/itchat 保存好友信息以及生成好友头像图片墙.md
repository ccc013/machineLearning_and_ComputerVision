>  2019  第 41  篇，总第  65 篇文章

最近简单运用 itchat 这个库来实现一些简单的应用，主要包括以下几个应用：

- 统计保存好友的数量和信息
- 统计和保存关注的公众号数量和信息
- 简单生成好友头像的图片墙，利用一个第三方库生成马赛克风格的图片墙

itchat 的 github 项目地址如下，这是一个开源的微信个人接口：

https://github.com/littlecodersh/ItChat

这个库的安装也很简单，直接用 `pip install itchat` 即可安装

接下来就开始介绍如何利用这个库来实现上述操作。

#### 1. 统计保存好友的数量和信息

首先是微信登录，简单的几行代码即可实现：

```python
import itchat

# 避免频繁扫描二维码登录
itchat.auto_login(hotReload=True)
itchat.dump_login_status()
```

运行这段代码后，就会弹出一个二维码，进行扫描登录，其中 `hotReload=True` 是保证不用每次运行程序都需要弹出二维码扫描登录。

然后是获取好友的信息：

```python
we_friend = itchat.get_friends(update=True)[:]
```

这里 `we_friend` 就是保存了好友信息的一个字典，并且 `we_friend[0]` 是保存用户自己的信息，从`we_friend[1]` 开始才是真正的好友的信息，这里我们将主要保存以下信息：

|    key     | 含义 |
| :--------: | :--: |
|  NickName  | 昵称 |
| RemarkName | 备注 |
|    Sex     | 性别 |
|  Province  | 省份 |
|    City    | 城市 |
| Signature  | 签名 |

保存好友的信息代码如下：

```python
friends = we_friend[1:]
total_numbers = len(friends)
print('你的好友数量为: {}'.format(total_numbers))
friend_infos_dict = {}
for fri_info in friends:
	for key in friend_key:
		if friend_infos_dict.get(key, False):
			friend_infos_dict[key].append(fri_info[key])
		else:
			friend_infos_dict[key] = [fri_info[key]]
# 保存信息
fri_save_file_name = os.path.join(save_file_path, '好友信息.csv')
df = pd.DataFrame(friend_infos_dict)
df.to_csv(fri_save_file_name, sep=',')
```

其中 `save_file_path` 是指定保存好友信息文件的文件夹路径，

#### 2. 保存公众号信息

获取公众号信息并保存的代码如下：

```python
# 公众号获取的信息内容，分别是昵称、城市、城市、签名
mps_key = ['NickName', 'City', 'Province', 'Signature']
# 获取公众号信息
mps = itchat.get_mps(update=True)
mps_num = len(mps)
print('你关注的公众号数量: {}'.format(mps_num))

mps_save_file_name = os.path.join(save_file_path, '公众号信息.csv')
mps_dict = {}
for mp in mps:
	for key in mps_key:
		if mps_dict.get(key, False):
			mps_dict[key].append(mp[key])
		else:
			mps_dict[key] = [mp[key]]

df = pd.DataFrame(mps_dict)
df.to_csv(mps_save_file_name, sep=',', encoding='utf-8')
```

#### 3. 生成好友头像图片墙

首先同样需要获取好友的头像，并保存到本地，代码如下：

```python
def save_head_photo(save_photo_dir):
    itchat.auto_login(hotReload=True)
    itchat.dump_login_status()
    friends = itchat.get_friends(update=True)[1:]

    # 采集好友头像并保存到本地
    num = 0
    for fri in friends:
        img = itchat.get_head_img(userName=fri['UserName'])
        img_path = os.path.join(save_photo_dir, str(num) + '.jpg')
        if not os.path.exists(img_path):
            file_image = open(img_path, 'wb')
            file_image.write(img)
            file_image.close()
        num += 1

    print('完成好友头像保存至路径: ', save_photo_dir)
```

其中获取头像的函数是 `itchat.get_head_image()` 。

接着就是生成好友头像的图片墙，这里有两种方式，第一种是比较常规的生成方法。首先需要导入以下库

```python
import itchat
import math
import PIL.Image as Image
import os
```

接着是设置画布大小及每行的头像数量，头像的大小，代码是：

```python
 # 画布大小
 image_size = 1280
 # 算出每张图片的大小多少合适
 each_size = int(math.sqrt(float(image_size * image_size) / len(ls)))
 # 每行图片数量
 lines = int(image_size / each_size)
 print('each_size={}, lines={}'.format(each_size, lines))
 # 创建 1280*1280 的画布
 image = Image.new('RGBA', (image_size, image_size))
```

利用的是 `pillow` 库，安装方式是 `pip install pillow` 。这里我设置的画布大小就是 1280 * 1280。

然后就是读取保存的头像，并逐一粘贴到画布上，代码如下：

```python
# 读取保存的好友头像图片
ls = os.listdir(save_photo_dir)
for i in range(0, len(ls)):
    try:
    	img_path = os.path.join(save_photo_dir, str(i) + ".jpg")
    	img = Image.open(img_path)
    except IOError:
    	print("Error for image: {}".format(img_path))
    else:
    	img = img.resize((each_size, each_size), Image.ANTIALIAS)
    	image.paste(img, (x * each_size, y * each_size))  # 粘贴位置
    	x += 1
    	if x == lines:  # 换行
    		x = 0
    		y += 1

image.save(os.path.join(os.getcwd(), "好友头像拼接图.jpg"))
```

第二种是参考了 [当 Python 遇上你的微信好友](https://mp.weixin.qq.com/s?__biz=Mzg2ODAyNTgyMQ==&mid=2247483938&idx=1&sn=26b997990527449a458eb8ffd50710dd&chksm=ceb3d690f9c45f862f96f2ca27e5bb6a560c631924abf1fe04010547e2d35ff458bb13b1406c&xtrack=1&scene=90&subscene=93&sessionid=1555895785&clicktime=1555895789&ascene=56&devicetype=android-26&version=27000436&nettype=WIFI&abtest_cookie=BAABAAoACwASABMABQAjlx4AW5keAMmZHgDTmR4A3JkeAAAA&lang=zh_CN&pass_ticket=c3+wIVsrgpMqsd7FdMAbhwdUHGsmG6vOhu9VTCWktleDHxWipn+RTzL79RlV8+gj&wx_header=1) 介绍的第三方库 `photomosaic` ，安装方法也很简单：

```
pip install photomosaic
```

这个第三方库可以生成蒙太奇马赛克风格的图片或者视频。

实现代码如下：

```python
import photomosaic as pm

def create_photomosaic(save_photo_dir, background_photo):
    # 读取背景图片
    bg_photo = pm.imread(background_photo)
    # 读取好友头像图片，定义图片库
    pool = pm.make_pool(os.path.join(save_photo_dir, '*.jpg'))
    # 制作 50*50 的拼图马赛克
    image = pm.basic_mosaic(bg_photo, pool, (50, 50))
    # 保存结果
    pm.imsave('马赛克好友头像图片.jpg', image)
```

其中上述的四行代码也是最基本的使用代码，包括：

- 选择背景图片
- 定义图片库
- 制作马赛克拼图
- 保存图片

这里我简单选择了下面这张背景图片：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/mosaic_bg.jpg)

生成结果如下：

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/%E9%A9%AC%E8%B5%9B%E5%85%8B%E5%A5%BD%E5%8F%8B%E5%A4%B4%E5%83%8F%E5%9B%BE%E7%89%87.jpg)



#### 小结

简单运用 itchat 实现了以上三个小应用，实际上还可以有更多的应用，比如再根据好友信息分析性别比例、好友区域分布、签名的情感分析、关注的公众号类别、给特定的好友发送信息，以及制作微信机器人等。

本文的代码已经上传到 github 上：

https://github.com/ccc013/Python_Notes/tree/master/Projects/wechatProjects/itchat

也可以按如下操作获取代码：

1.关注公众号“机器学习与计算机视觉”
2.在公众号后台回复“itchat"，即可获取代码



------

参考：

- [手把手教你用itchat统计好友信息，了解一下？](https://mp.weixin.qq.com/s/fWU1J_h235I76gavm69IYw)
- [当 Python 遇上你的微信好友](https://mp.weixin.qq.com/s?__biz=Mzg2ODAyNTgyMQ==&mid=2247483938&idx=1&sn=26b997990527449a458eb8ffd50710dd&chksm=ceb3d690f9c45f862f96f2ca27e5bb6a560c631924abf1fe04010547e2d35ff458bb13b1406c&xtrack=1&scene=90&subscene=93&sessionid=1555895785&clicktime=1555895789&ascene=56&devicetype=android-26&version=27000436&nettype=WIFI&abtest_cookie=BAABAAoACwASABMABQAjlx4AW5keAMmZHgDTmR4A3JkeAAAA&lang=zh_CN&pass_ticket=c3+wIVsrgpMqsd7FdMAbhwdUHGsmG6vOhu9VTCWktleDHxWipn+RTzL79RlV8+gj&wx_header=1)



欢迎关注我的微信公众号--机器学习与计算机视觉，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_new.jpg)


#### 往期精彩推荐

##### 机器学习系列

- [初学者的机器学习入门实战教程！](https://mp.weixin.qq.com/s/HoFiD0ItcO5_pVMspni_xw)
- [模型评估、过拟合欠拟合以及超参数调优方法](https://mp.weixin.qq.com/s/1NxVNtKNsZFWYI62KzL1GA)
- [常用机器学习算法汇总比较(完）](https://mp.weixin.qq.com/s/V2C4u9mSHmQdVl9ZYs1-FQ)
- [常用机器学习算法汇总比较(上）](https://mp.weixin.qq.com/s/4Ban_TiMKYUBXTq4WcMr5g)
- [机器学习入门系列(2)--如何构建一个完整的机器学习项目(一)](https://mp.weixin.qq.com/s/nMG5Z3CPdwhg4XQuMbNqbw)
- [特征工程之数据预处理（上）](https://mp.weixin.qq.com/s/BnTXjzHSb5-4s0O0WuZYlg)


##### Github项目 & 资源教程推荐

- [[Github 项目推荐] 一个更好阅读和查找论文的网站](https://mp.weixin.qq.com/s/ImQcGt8guLKZawNLS-_HzA)
- [[资源分享] TensorFlow 官方中文版教程来了](https://mp.weixin.qq.com/s/Si1YaYLfhL1upbjQkvireQ)
- [必读的AI和深度学习博客](https://mp.weixin.qq.com/s/0J2raJqiYsYPqwAV1MALaw)
- [[教程]一份简单易懂的 TensorFlow 教程](https://mp.weixin.qq.com/s/vXIM6Ttw37yzhVB_CvXmCA)
- [[资源]推荐一些Python书籍和教程，入门和进阶的都有！](https://mp.weixin.qq.com/s/jkIQTjM9C3fDvM1c6HwcQg)
- [[Github项目推荐] 机器学习& Python 知识点速查表](https://mp.weixin.qq.com/s/kn2DUJHL48UyuoUEhcfuxw)
- [[Github项目推荐] 推荐三个助你更好利用Github的工具](https://mp.weixin.qq.com/s/Mtijg-AXN4zCeZktkr7nqQ)
- [Github上的各大高校资料以及国外公开课视频](https://mp.weixin.qq.com/s/z3XS6fO303uVZSPmFKBKlQ)
- [这些单词你都念对了吗？顺便推荐三份程序员专属英语教程！](https://mp.weixin.qq.com/s/yU3wczdePK9OvMgjNg5bug)





