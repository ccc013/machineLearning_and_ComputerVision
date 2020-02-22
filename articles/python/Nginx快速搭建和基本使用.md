
> 2019年第 83 篇文章，总第 107 篇文章

最近在工作中项目需要上线，所以也了解到关于一些部署上线的知识内容，Nginx 就是其中一个知识点，主要是可以用它来进行负载均衡，本文的目录如下：

- 简介
- 安装配置
- 基本使用



### 简介

关于Nginx，来自维基百科的介绍（https://zh.wikipedia.org/wiki/Nginx）：

> **Nginx**（发音同“engine X”）是异步框架的[网页服务器](https://zh.wikipedia.org/wiki/網頁伺服器)，也可以用作[反向代理](https://zh.wikipedia.org/wiki/反向代理)、[负载平衡器](https://zh.wikipedia.org/wiki/负载均衡)和 HTTP 缓存。

Nginx 使用异步事件驱动的方法来处理请求，相比于 Apache、lighttpd 具有占有内存少，稳定性高、并发服务能力强等优势，根据官方测试结果，可以支持五万个并行连接，而在实际的运作中，可以支持两万至四万个并行连接。

------

### 安装配置

#### 安装gcc和gcc-c++

首先需要安装 gcc 和 gcc-c++，在centos下安装的命令：

```
yum install gcc gcc-c++
```

#### 安装 PCRE 库


```
wget http://jaist.dl.sourceforge.net/project/pcre/pcre/8.33/pcre-8.33.tar.gz
tar -zxvf pcre-8.33.tar.gz
cd pcre-8.33
./configure
make && make install
```

#### 安装 Perl 5

参考文章：https://blog.csdn.net/qq_20678155/article/details/68926562


```
// 下载安装包
wget http://www.cpan.org/src/5.0/perl-5.16.1.tar.gz

// 解压源码包
tar -xzf perl-5.16.1.tar.gz

// 进入源码目录
cd perl-5.16.1

//自定义安装目录
./Configure -des -Dusethreads -Dprefix=/usr/local/perl

// 下面这三个命令要依次都执行，这是在编译源码
make
make test
make install


// 查看版本
perl -v

```

#### 安装 openssl


```
wget http://www.openssl.org/source/openssl-1.0.1j.tar.gz
tar -zxvf openssl-1.0.1j.tar.gz
cd openssl-1.0.1j
./config
make && make install
```

#### 安装 zlib 

```
wget http://zlib.net/zlib-1.2.11.tar.gz
tar -zxvf zlib-1.2.11.tar.gz
./configure
make && make install
```

#### 安装 nginx

```
# nginx
wget http://nginx.org/download/nginx-1.8.0.tar.gz
tar -zxvf nginx-1.8.0.tar.gz
cd nginx-1.8.0
./configure --prefix=/usr/local/nginx
make && make install
```

#### nginx 测试

```
/usr/local/nginx/sbin/nginx -t # 测试一下配置文件是否正确
/usr/local/nginx/sbin/nginx # 启动
curl -X GET localhost:80 # 出现 Welcome to nginx! 则表示 Nginx 已经安装并运行成功
# /usr/local/nginx/sbin/nginx –s reload
/usr/local/nginx/sbin/nginx –s stop
```

### 基本使用

#### 常用命令

```
nginx -v # version info
rpm -ql nginx
nginx -V

systemctl start/stop/status/restart/reload nginx
# reload, restart 都是重复服务，但 reload 并不需要关闭服务
```

#### 配置

使用的话，需要修改在 `/usr/local/nginx/conf`文件夹的配置文件 `nginx.conf` 中下面的内容：


```
http {
    ...
    
  upstream ip0 {
      server ip1:port1;
      server ip2:port2;
      ...
  }
  
  server {
      listen port0;
      server_name ip0;
      ...
      location / {
          proxy_pass ip0;
      }
      ...
  }
  ... 
}
```

这里是请求 `ip0:port0` ，然后 nginx 会将请求转发到 `ip1:port1, ip2:port2,...` 上，也就是说，配置 nginx 的机器的ip就是 ip1，然后设置一个端口 port0，而 ip1，ip2 等则是运行服务的机器，由于 nginx 也是需要占用 cpu 的，所以建议单独用一台机器配置nginx，并且在配置文件中可以设置开启多进程，只需要修改 `work_process` 后的数字，通常设置为机器的cpu的核数量-1的数量。

------

欢迎关注我的微信公众号--**算法猿的成长**，或者扫描下方的二维码，大家一起交流，学习和进步！

![](https://cai-images-1257823952.cos.ap-beijing.myqcloud.com/qrcode_0601.png)

**如果觉得不错，在看、转发就是对小编的一个支持！**


