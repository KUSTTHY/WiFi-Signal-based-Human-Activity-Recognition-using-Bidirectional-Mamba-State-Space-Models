# WiFi-Signal-based-Human-Activity-Recognition-using-Bidirectional-Mamba-State-Space-Models

The model first uses 1D depthwise separable convolution to extract spatial features, and then uses the improved bidirectional Mamba module to extract temporal features (code paper "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model," paper link: https://arxiv.org/pdf/2401.09417.pdf, code link: https://github.com/hustvl/Vim). The bidirectional Mamba module is optimized for the WiFi human identification task, improving the data input part and removing redundant parameters.

Use the command line to select the running mode:：

python run.py --dataset NTU-Fi-HumanID

python run.py --dataset NTU-Fi_HAR

python run.py --dataset UT
