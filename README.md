# PV_MCSTF_demo
A time series data imaging technique and spatio-temporal fusion network for anomaly diagnosis in photovoltaic power generation systems.

# Dataset
The dataset utilised in this project may be downloaded from [Synthetic Anomaly Dataset of Photovoltaic Power Generation](https://doi.org/10.5281/zenodo.8248701).  
This dataset, published by the Key Laboratory of Artificial Intelligence at the Institute of Artificial Intelligence, Shanghai Jiao Tong University, comprises one normal class and six abnormal classes.  
Each system collects observational data exclusively within the time window from 03:45 to 21:30 Beijing time, with a sampling frequency of 15 minutes. 
As a result, each time series sample obtained contains 72 successive time points.
每个系统仅在03:45至21:30北京时间的时间窗口内采集观测数据，采样频率为15分钟。
因此，每次获取的时间序列样本包含72个连续时间点。
本项目对原始训练集和测试集分别汇总到两个文件中，并进行了重新划分。
从原测试集中选取三个非重叠站点(30100119、30100137、30100146)的样本作为新测试集。
同时将其余六个站点的样本重新分配至训练集。
样本比例约为8:2，训练集含8796个样本，测试集含2256个样本。

# 环境需求
torch 1.9.0+cu111
torchvission 0.10.0+cu111

# How to run
首先，运行dataset_MCSTF.py获得MCSTF图像数据集；
然后，设置正确的图像与时序数据读取路径；
最后，运行main.py文件，获得训练、验证和测试结果。
