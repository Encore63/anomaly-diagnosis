import torch
import numpy as np

from typing import List
from utils.data_utils import data_concat
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split


class _TimeSeriesDataset(Dataset):
    def __init__(self, list_data, time_win=10, data_dim=3, transform=None):
        self.data = list_data[:, :, :-1]
        self.labels = list_data[:, :, -1]

        self.transform = transform
        self.length = self.data.shape[0]

        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.labels = torch.from_numpy(self.labels).to(torch.long)
        self._adjust_data_dim(time_win, data_dim)

    def _adjust_data_dim(self, time_win: int, data_dim: int):
        if time_win != 0:
            offset = data_dim - len(self.data.shape) + 1
            if offset >= 0:
                for _ in range(offset):
                    self.data = torch.unsqueeze(self.data, dim=1)
        else:
            self.data = torch.squeeze(self.data, dim=1)

    def __getitem__(self, index):
        if self.transform:
            self.data = self.transform(self.data)
            self.labels = self.transform(self.labels)
        return self.data[index], self.labels[index, 0]

    def __len__(self):
        return self.length


class TEPDataset(object):
    def __init__(self, src_path: str, transfer_task: List, transfer=True, seed: int = 2024, data_dim: int = 3,
                 neglect: list = None, time_win: int = 10, num_classes: int = 10, overlap: bool = True, transform=None):
        self.source = transfer_task[0]
        self.target = transfer_task[1]

        source_raw, target_raw = [], []
        for source_mode in self.source:
            source_raw.append(data_concat(src_path, source_mode, time_win=time_win, neglect=neglect,
                                          num_classes=num_classes, overlap=overlap))
        for target_mode in self.target:
            target_raw.append(data_concat(src_path, target_mode, time_win=time_win, neglect=neglect,
                                          num_classes=num_classes, overlap=overlap))
        self.source_data = np.concatenate(source_raw, axis=0)
        self.target_data = np.concatenate(target_raw, axis=0)

        self.transfer = transfer
        self.random_seed = seed
        self.data_dim = data_dim
        self.time_win = time_win
        self.transform = transform

        '''
        Construct uniform distribution for raw data.
        '''
        self._data_shuffle()

    def _data_shuffle(self):
        np.random.seed(self.random_seed)
        np.random.shuffle(self.source_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(self.target_data)

    def get_subset(self, choice=None):
        _subset = {}
        if self.transfer:
            source_train, source_val = train_test_split(self.source_data, test_size=0.2,
                                                        random_state=self.random_seed)
            if choice == 'train':
                _subset.setdefault('train', _TimeSeriesDataset(source_train, time_win=self.time_win,
                                                               data_dim=self.data_dim, transform=self.transform))
                _subset.setdefault('val', _TimeSeriesDataset(source_val, time_win=self.time_win,
                                                             data_dim=self.data_dim, transform=self.transform))
            elif choice == 'test':
                _subset = _TimeSeriesDataset(self.target_data, time_win=self.time_win,
                                             data_dim=self.data_dim, transform=self.transform)
            else:
                _subset.setdefault('train', _TimeSeriesDataset(source_train, time_win=self.time_win,
                                                               data_dim=self.data_dim, transform=self.transform))
                _subset.setdefault('val', _TimeSeriesDataset(source_val, time_win=self.time_win,
                                                             data_dim=self.data_dim, transform=self.transform))
                _subset.setdefault('test', _TimeSeriesDataset(self.target_data, time_win=self.time_win,
                                                              data_dim=self.data_dim, transform=self.transform))
            return _subset
        else:
            source_train_val, source_test = train_test_split(self.source_data, test_size=0.1,
                                                             random_state=self.random_seed)
            source_train, source_val = train_test_split(source_train_val, test_size=0.2,
                                                        random_state=self.random_seed)
            if choice == 'train':
                _subset.setdefault('train', _TimeSeriesDataset(source_train, time_win=self.time_win,
                                                               data_dim=self.data_dim, transform=self.transform))
                _subset.setdefault('val', _TimeSeriesDataset(source_val, time_win=self.time_win,
                                                             data_dim=self.data_dim, transform=self.transform))
            elif choice == 'test':
                _subset = _TimeSeriesDataset(source_test, time_win=self.time_win,
                                             data_dim=self.data_dim, transform=self.transform)
            else:
                _subset.setdefault('train', _TimeSeriesDataset(source_train, time_win=self.time_win,
                                                               data_dim=self.data_dim, transform=self.transform))
                _subset.setdefault('val', _TimeSeriesDataset(source_val, time_win=self.time_win,
                                                             data_dim=self.data_dim, transform=self.transform))
                _subset.setdefault('test', _TimeSeriesDataset(source_test, time_win=self.time_win,
                                                              data_dim=self.data_dim, transform=self.transform))
            return _subset


if __name__ == '__main__':
    tep_dataset = TEPDataset(src_path=r'../data/TEP', transfer_task=[[1], [2]])
    subset = tep_dataset.get_subset('test')
    print(len(subset))
