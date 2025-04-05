# WiFi Signal-Based Human Activity Recognition Using Bidirectional Mamba Models

## Project Introduction

This repository provides the accompanying code for the paper titled "WiFi Signal-Based Human Activity Recognition Using Bidirectional Mamba Models". The paper explores the application of Bidirectional Mamba Models (BMM) for Human Activity Recognition (HAR) using channel state information obtained from multi-antenna WiFi signals. The codebase includes the implementation of the proposed model, along with the links to necessary datasets and evaluation metrics.

**Paper Abstract**: WiFi signals, due to their non-contact, low-cost, and privacy-preserving nature, have become an important data source for Human Activity Recognition (HAR). In this paper, we propose a state space model-based deep learning method for HAR using channel state information obtained from multi-antenna WiFi signals. Specifically, the model is designed by combining depthwise separable convolution and a bidirectional Mamba model, which aim to extract the spatial and temporal features jointly along long sequential WiFi data frame. Experimental results show that our proposed model achieves significant improvements in accuracy on three widely-used open datasets of NTU-Fi HAR, NTU-Fi Human-ID and UT-HAR, when compared to the reference state-of-the-art algorithms including BiLSTM, ProbSparse Transformer and ViT. Experimental results also show that our proposed model significantly reduces parameter size compared to reference models with comparable latent states while ensuring near-perfect test accuracy.

## Method Description

The model first uses 1D depthwise separable convolution to extract local spatiotemporal features, and then uses the improved bidirectional Mamba module to extract temporal features (method inspired by paper "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model," code link: <https://github.com/hustvl/Vim>). The bidirectional Mamba module is optimized for the WiFi human identification task, by improving the data input part and removing redundant parameters.

Use the following command line to select the running mode for differrent datasets:
```bash
python run.py --dataset NTU-Fi-HumanID
python run.py --dataset NTU-Fi_HAR
python run.py --dataset UT
```

**Note**: The project doesnot include the datasets. Please download the datasets via the following links:
s. Please download the datasets from the following links:
- <https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark> 

The work can be cited as follows:

[1] H. Tan, Y. Dao, H. Zhang, T. Guo and W. Wang, "WiFi Signal-Based Human Activity Recognition Using Bidirectional Mamba Models," submitted to 2025 IEEE 11th World Forum on Internet of Things (WF-IoT), Chengdu, 2025.
