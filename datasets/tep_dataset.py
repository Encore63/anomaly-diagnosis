from collections import defaultdict

import torch

from typing import Dict, List
from dataclasses import dataclass
from utils.data_utils import data_split
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


@dataclass
class DatasetConfig:
    data_domains: Dict
    dataset_mode: str
    seed: int = 2024,
    data_dim: int = 3,
    src_path: str = './data/TEP',
    split_ratio: Dict = {'train': 0.7, 'eval': 0.2},
    neglect: List = None,
    time_win: int = 10,
    num_classes: int = 10,
    overlap: bool = True,
    transform = None


class TEPDataset(Dataset):
    def __init__(self, src_path: str, split_ratio: Dict, data_domains: Dict,
                 dataset_mode: str, seed: int = 2024, data_dim: int = 4, neglect: list = None,
                 time_win: int = 10, num_classes: int = 10, overlap: bool = True, transform=None):
        self.raw_data = data_split(src_path, split_ratio, data_domains, random_seed=seed,
                                   neglect=neglect, num_classes=num_classes, time_win=time_win, overlap=overlap)
        self.labels = None
        self.domains = data_domains
        self.transform = transform
        self._select_dataset_mode(dataset_mode)

        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.labels = torch.from_numpy(self.labels).to(torch.long)
        self._adjust_data_dim(time_win, data_dim)

        self.length = self.data.shape[0]

    def _adjust_data_dim(self, time_win: int, data_dim: int):
        if time_win != 0:
            offset = data_dim - len(self.data.shape)
            if offset > 0:
                for _ in range(offset):
                    self.data = torch.unsqueeze(self.data, dim=1)
        else:
            self.data = torch.squeeze(self.data, dim=1)

    def _select_dataset_mode(self, mode_choice: str):
        if mode_choice == 'train':
            self.data = self.raw_data['source_train'][:, :, :-1]
            self.labels = self.raw_data['source_train'][:, :, -1]
        elif mode_choice == 'eval':
            self.data = self.raw_data['source_eval'][:, :, :-1]
            self.labels = self.raw_data['source_eval'][:, :, -1]
        elif mode_choice == 'test' and self.domains['target'] is not None:
            self.data = self.raw_data['target_test'][:, :, :-1]
            self.labels = self.raw_data['target_test'][:, :, -1]
        else:
            assert self.data is not None or self.labels is not None, 'Given data does not exist!'

    def __getitem__(self, index):
        if self.transform:
            self.data = self.transform(self.data)
            self.labels = self.transform(self.labels)
        return self.data[index], self.labels[index, 0]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = TEPDataset(src_path=r'../data/TEP',
                         split_ratio={'train': 0.7, 'eval': 0.2},
                         data_domains={'source': 1, 'target': 2},
                         dataset_mode='test',
                         data_dim=4,
                         transform=None,
                         overlap=True)
    data_iter = DataLoader(dataset, batch_size=128, shuffle=True)
    for x, y in data_iter:
        print(x.shape, y.shape)
        break
