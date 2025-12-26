# PV_MCSTF_demo
A time series data imaging technique and spatio-temporal fusion network for anomaly diagnosis in photovoltaic power generation systems.

# Dataset
The dataset utilised in this project may be downloaded from [Synthetic Anomaly Dataset of Photovoltaic Power Generation](https://doi.org/10.5281/zenodo.8248701).  
This dataset, published by the Key Laboratory of Artificial Intelligence at the Institute of Artificial Intelligence, Shanghai Jiao Tong University, comprises one normal class and six abnormal classes.  

Each system collects observational data exclusively within the time window from 03:45 to 21:30 Beijing time, with a sampling frequency of 15 minutes. 
As a result, each time series sample obtained contains 72 successive time points.  

This project consolidated the original train set and test set into two separate files and redistributed them.
Three non-overlapping sites (30100119, 30100137, 30100146) were selected from the original test set to form the new test set.
The samples from the remaining six sites were reassigned to the train set.
The sample ratio is approximately 8:2, with the training set containing 8,796 samples and the test set containing 2,256 samples.  

# 环境需求
torch 1.9.0+cu111  
torchvission 0.10.0+cu111

# How to run
First, run the dataset_MCSTF.py to obtain the MCSTF image dataset;  
Then, configure the correct paths for reading image and time series data;  
Finally, run the main.py file to obtain training, validation, and testing results.
