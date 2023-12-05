# object tracking using KF

使用卡尔曼滤波器实现单目标跟踪的算法。

## 版本更新信息

- v1.0:

  - 实现基于最大IOU匹配的单目标跟踪
  - 增加KF预测轨迹模块，解决检测漏帧问题
- v2.0

  - 增加EKF预测模块

  - 将状态矩阵A维数拓展到8，解决长宽估计不准确

  - 调整函数初试运行使用方式

  - 增加rmse误差计算
- v2.1
  - 将yolo模型检测结果作为检测值，调整mask rcnn检查结果为真值，方便rmse计算
  - 改进rmse计算方式，改为计算最近window_len长度数据的rmse，即moving_rmse
- v2.2
  - 增加图像显示
  
  - 调整EKF状态方程建模，对部分初始值做出修改
- v2.3
  - 显示图像修改
  - 评估方式从仅考虑x改为xywh的均值，即4个变量权重相同
  - EFK状态向量修改到10维，调整程序初试化代码
  - 文件夹新增不同方式输出结果






## how to use 使用：

```
python ./main [path/to/video] --filter [filter_type]
例如：
python ./main ./data/output.avi --filter KF
python ./main ./data/output.avi --filter EKF
```

## 需要自行下载：

部分资料由于大小不能直接上传github（限制100M）

​	output.avi

​	maskrcnn模型

​	mask_rcnn_coco.h5

​	yolo v7模型

## reference 参考教程：

1、【目标跟踪原理】 https://www.bilibili.com/video/BV1Qf4y1J7D4

2、【超详细讲解无迹卡尔曼（UKF）滤波】http://t.csdnimg.cn/RyHV6

3、【【卡尔曼滤波器】6_扩展卡尔曼滤波器_Extended Kalman Filter】 https://www.bilibili.com/video/BV1jK4y1U78V



