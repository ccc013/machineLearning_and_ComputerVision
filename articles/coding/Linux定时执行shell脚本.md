
> 2019年第 85 篇文章，总第 109 篇文章

`crontab` 可以在指定的时间执行一个shell脚本以及执行一系列 Linux 命令。

### 定时执行shell 脚本

简单给出执行 shell 脚本的步骤。

1. 首先是编写一个测试脚本--`test.sh` 

```shell
# 创建脚本
$ vim test.sh
# 在脚本中做写入文件操作
date >> data.txt
```

2. 修改脚本的权限，确保脚本有执行的权限

```shell
chmod 777 test.sh
```

3. 设置 crontab 定时任务

```sh
# 打开定时任务配置文件
crontab -e
# 在配置文件中写入定时任务的操作, 这里就是指定每天12点定时执行脚本，并把执行脚本的日志写入文件 test.log
0 12 * * * sh test.sh > test.log
```

4. 保存退出，也就是 `:wq`
5. 如果有错，linux 会在执行的时候发送一份邮件给你

```shell
cat /var/spool/mail/root
```

注意：

crontab 是运行在系统默认环境里，如果运行的脚本是执行 python 代码，即脚本的内容可能是：

```shell
python test.py
```

这里的 `python` 会是系统默认的 python 版本，而如果你是运行在 conda 环境里，那么这里就需要采用当前环境里 python 版本的执行文件的绝对路径，即先用以下命令查找当前 python 版本的执行文件位置：

```shell
$ which python
# 假设输出的文件位置为：
/root/anaconda3/py3/bin/python
```

这里输出的路径，直接替换脚本里的 `python`:

```shell
/root/anaconda3/py3/bin/python test.py
```

这样才能保证运行不出错，否则可能因为版本问题出错；



------

### crontab命令详解

#### 常用命令

```shell
crontab –e     //修改 crontab 文件，如果文件不存在会自动创建。 
crontab –l      //显示 crontab 文件。 
crontab -r      //删除 crontab 文件。
crontab -ir     //删除 crontab 文件前提醒用户。

service crond status     //查看crontab服务状态
service crond start     //启动服务 
service crond stop     //关闭服务 
service crond restart     //重启服务 
service crond reload     //重新载入配置
```

所以如果需要取消某个定时任务，就是可以删除在配置文件中的对应命令，即 `crontab -e` 打开文件，然后删除对应哪行的命令即可



#### 基本格式

```
*　　*　　*　　*　　*　　command
分  时　 日　 月　 周　  命令
```

第1列表示分钟 00～59 每分钟用`*`或者 `*/1`表示

第2列表示小时 00～23（0表示0点）

第3列表示日期 01～31

第4列表示月份 01～12

第5列标识号星期 0～6（0表示星期天）

第6列要运行的命令

此外每一列除了数字，还可以有这些符号，其含义如下所示：

```
*        代表任何时间，比如第一个 * 就代表一小时中的每分钟都执行
,        代表不连续的时间，比如 0 8,12,16 * * * 代表每天8，12，16点0分执行
-        代表连续的时间范围，比如0 5 * * 1-6 代表在周一到周六凌晨5点0分执行
*/n     代表每个多久执行一次，比如*/10 * * * *代表每隔10分钟执行一次
```

#### 示例

1、在 凌晨00:10运行

```
10 0 * * * sh test.sh
```

2、每个工作日23:59都进行备份作业。

```
59 23 * * 1,2,3,4,5 sh test.sh   
或者  
59 23 * * 1-5 sh test.sh
```

3、每分钟运行一次命令

```
*/1 * * * * sh test.sh
```

4、每个月的1号 14:10 运行

```
10 14 1 * * sh test.sh
```

5、每10分钟定时请求一个地址

```
0 */10 * * * /usr/bin/curl http://www.aaa.com/index.php
```

注意，一般最好不要同时采用几号和每周几，可能会出现混淆；

 

**正确、错误日志的输出是否写入到文件方法**

1.**不输出任何内容**（建议使用方法一）

```
*/1 * * * * /root/XXXX.sh >/dev/null 2>&1 
或
*/1 * * * * /root/XXXX.sh &>/dev/null    //&表示任何内容
```

2.**将正确和错误日志都输出到** /tmp/load.log

```
*/1 * * * * /root/XXXX.sh > /tmp/load.log 2>&1
```

3.**只输出正确日志到** /tmp/load.log

```
*/1 * * * * /root/XXXX.sh > /tmp/load.log
或
*/1 * * * * /root/XXXX.sh 1> /tmp/load.log    //1可以省略
```

4.**只输出错误日志到** /tmp/load.log

```
*/1 * * * * /root/XXXX.sh 2> /tmp/load.log
```

部分解释:

```
/dev/null 代表空设备文件
> 代表重定向到哪里
1 表示stdout标准输出，系统默认值是1，所以">/dev/null"等同于"1>/dev/null"
2 表示stderr标准错误
& 表示等同于的意思，2>&1，表示2的输出重定向等同于1
```



---

参考文章：

- [Linux 定时执行shell脚本命令之crontab](https://www.cnblogs.com/wenzheshen/p/8432588.html)

- [linux定时执行sh文件](https://blog.csdn.net/IT_xiaocao/article/details/78206364)


---
欢迎关注我的微信公众号--**算法猿的成长**，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_0601.png)

**如果觉得不错，在看、转发就是对小编的一个支持！**
