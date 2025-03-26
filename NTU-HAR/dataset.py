import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import glob
import torch.nn.functional as F


class CSI_Dataset(Dataset):
    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform
        self.data_list = glob.glob(root_dir + '/*/*.mat')
        self.folder = glob.glob(root_dir + '/*/')
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        # normalize
        x = (x - 42.3199) / 4.9802
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = x.squeeze(0)
        # x = x[:, ::2]
        # x = x.reshape(342, 1000)
        if self.transform:
            x = self.transform(x)
        x = torch.FloatTensor(x)
        return x, y