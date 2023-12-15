import torch

from typing import Dict

import models.tenet
from utils.data_utils import data_split
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class TEPDataset(Dataset):
    def __init__(self, src_path: str, split_ratio: Dict, data_domains: Dict,
                 dataset_mode: str, data_dim: int = 4, transform=None):
        _data = data_split(src_path, split_ratio, data_domains)
        self.data = None
        self.labels = None
        if dataset_mode == 'train':
            self.data = _data['source_train'][:, :, :-1]
            self.labels = _data['source_train'][:, :, -1]
        elif dataset_mode == 'eval':
            self.data = _data['source_eval'][:, :, :-1]
            self.labels = _data['source_eval'][:, :, -1]
        elif dataset_mode == 'test':
            self.data = _data['target_test'][:, :, :-1]
            self.labels = _data['target_test'][:, :, -1]
        else:
            assert "data is not exist!"

        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.labels = torch.from_numpy(self.labels).to(torch.long)
        offset = data_dim - len(self.data.shape)
        if offset > 0:
            for _ in range(offset):
                self.data = torch.unsqueeze(self.data, dim=1)

        self.length = self.data.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            self.data = self.transform(self.data)
            self.labels = self.transform(self.labels)
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = TEPDataset(src_path=r'../data/TEP',
                         split_ratio={'train': 0.8, 'eval': 0.2},
                         data_domains={'source': 1, 'target': 3},
                         dataset_mode='train',
                         transform=None)
    data_iter = DataLoader(dataset, batch_size=32, shuffle=True)
    for _, (data, label) in enumerate(data_iter):
        print(data)
