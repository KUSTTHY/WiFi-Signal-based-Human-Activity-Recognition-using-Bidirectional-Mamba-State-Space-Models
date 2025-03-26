import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CSI_Dataset
from vision_mamba import VisionMamba,FusionModel
from train_and_test import train, test, val

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    print(f'A-S')
    # 数据集路径
    train_root_dir = '/home/aip/Vim-main/DATA/NTU-Fi-HumanID/test_amp'
    test_root_dir = '/home/aip/Vim-main/DATA/NTU-Fi-HumanID/train_amp'

    # 实例化数据集
    train_dataset = CSI_Dataset(root_dir=train_root_dir, modal='CSIamp')
    test_dataset = CSI_Dataset(root_dir=test_root_dir, modal='CSIamp')

    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=32)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=32)

    # 实例化融合模型
    fusion_model = FusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    model = fusion_model
    train_epoch = 100

    train(
        model=model,
        tensor_loader=train_loader,
        val_loader=test_loader,
        num_epochs=train_epoch,
        learning_rate=1e-4,
        criterion=criterion,
        device=device
    )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )
    val(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )


if __name__ == "__main__":
    main()