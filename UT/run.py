import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UT_HAR_dataset
from vision_mamba import FusionModel
from train_and_test import train, test, val

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    root = '/home/aip/Vim-main/DATA'
    data = UT_HAR_dataset(root)

    train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
    test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, num_workers=32)  # drop_last=True
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=32)

    # 实例化融合模型
    fusion_model = FusionModel().to(device)
    model=fusion_model
    train_epoch = 100

    criterion = nn.CrossEntropyLoss()
    train(
        model=model,
        tensor_loader=train_loader,
        val_loader=test_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device
    )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )
    return


if __name__ == "__main__":
    main()