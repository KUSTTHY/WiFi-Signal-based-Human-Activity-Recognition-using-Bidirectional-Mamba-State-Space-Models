import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from dataset import NTU_HAR_Dataset, UT_HAR_dataset  # 请根据实际情况修改
from vision_mamba import FusionModel
from train_and_test import train, test, val

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # 数据集路径和参数设置
    if args.dataset == 'NTU':
        # NTU 数据集的路径
        train_root_dir = '/home/aip/Vim-main/DATA/NTU-Fi-HumanID/test_amp'
        test_root_dir = '/home/aip/Vim-main/DATA/NTU-Fi-HumanID/train_amp'

        # 实例化 NTU 数据集
        train_dataset = NTU_HAR_Dataset(root_dir=train_root_dir, modal='CSIamp')
        test_dataset = NTU_HAR_Dataset(root_dir=test_root_dir, modal='CSIamp')

        # 设置 FusionModel 参数
        depth = 2
        embed_dim = 342
        channels = 1000
        num_classes = 14
        in_channels = 342
        out_channels = 342
        kernel_size = 5
    elif args.dataset == 'UT':
        # UT 数据集的路径
        root = '/home/aip/Vim-main/DATA'
        data = UT_HAR_dataset(root)

        train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'], data['X_test']), 0),
                                                  torch.cat((data['y_val'], data['y_test']), 0))

        # 设置 FusionModel 参数
        depth = 8
        embed_dim = 90
        channels = 250
        num_classes = 7
        in_channels = 90
        out_channels = 90
        kernel_size = 5
    else:
        raise ValueError("Unsupported dataset. Choose either 'NTU' or 'UT'.")

    # 创建 DataLoader
    if args.dataset == 'NTU':
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=32)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=32)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, num_workers=32)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=32)

    # 实例化融合模型
    fusion_model = FusionModel(depth=depth, embed_dim=embed_dim, channels=channels,
                               num_classes=num_classes, in_channels=in_channels,
                               out_channels=out_channels, kernel_size=kernel_size).to(device)
    criterion = nn.CrossEntropyLoss()

    # 训练和测试
    train_epoch = 100
    train(
        model=fusion_model,
        tensor_loader=train_loader,
        val_loader=test_loader,
        num_epochs=train_epoch,
        learning_rate=1e-4 if args.dataset == 'NTU' else 1e-3,
        criterion=criterion,
        device=device
    )
    test(
        model=fusion_model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )
    val(
        model=fusion_model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with different datasets.")
    parser.add_argument('--dataset', choices=['NTU', 'UT'], required=True, help="Choose dataset: 'NTU' or 'UT'.")
    args = parser.parse_args()

    main(args)
