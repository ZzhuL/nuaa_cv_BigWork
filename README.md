# nuaa_cv_BigWork || U-Net(Convolutional Networks for Biomedical Image Segmentation)
# 该项目参考一下开源仓库
<https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_segmentation/unet>  
[Attention U-Net]<https://zhuanlan.zhihu.com/p/114471013>
# 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10
* Ubuntu或Centos(Windows暂不支持多GPU训练)
# 文件结构:
  ├── src: 搭建U-Net模型代码  
      ├──unet.py  
      └──Attention U-Net.py  
  ├── train_utils: 训练、验证以及多GPU训练相关模块  
  ├── BCdataset.py: 自定义dataset用于读取DRIVE数据集(视网膜血管分割)  
  ├── train.py: 以单GPU为例进行训练  
  └── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试  
