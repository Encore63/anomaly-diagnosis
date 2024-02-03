import torch

from typing import Dict

from models.tenet import TENet
from utils.data_utils import data_split
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class TEPDataset(Dataset):
    def __init__(self, src_path: str, split_ratio: Dict, data_domains: Dict,
                 dataset_mode: str, seed: int = None, data_dim: int = 4, neglect: list = None,
                 time_win: int = 10, num_classes: int = 10, overlap: bool = True, transform=None):
        _data = data_split(src_path, split_ratio, data_domains, random_seed=seed,
                           neglect=neglect, num_classes=num_classes, time_win=time_win, overlap=overlap)
        self.data = None
        self.labels = None
        if dataset_mode == 'train':
            self.data = _data['source_train'][:, :, :-1]
            self.labels = _data['source_train'][:, :, -1]
        elif dataset_mode == 'eval':
            self.data = _data['source_eval'][:, :, :-1]
            self.labels = _data['source_eval'][:, :, -1]
        elif dataset_mode == 'test' and data_domains['target'] is not None:
            self.data = _data['target_test'][:, :, :-1]
            self.labels = _data['target_test'][:, :, -1]
        else:
            assert self.data is not None or self.labels is not None, 'Given data does not exist!'

        self.data = torch.from_numpy(self.data).to(torch.float32)
        self.labels = torch.from_numpy(self.labels).to(torch.long)
        if time_win != 0:
            offset = data_dim - len(self.data.shape)
            if offset > 0:
                for _ in range(offset):
                    self.data = torch.unsqueeze(self.data, dim=1)
        else:
            self.data = torch.squeeze(self.data, dim=1)

        self.length = self.data.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            self.data = self.transform(self.data)
            self.labels = self.transform(self.labels)
        return self.data[index], self.labels[index, 0]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = TEPDataset(src_path=r'../data/TEP',
                         split_ratio={'train': 0.7, 'eval': 0.3},
                         data_domains={'source': 1, 'target': 3},
                         dataset_mode='test',
                         data_dim=3,
                         transform=None,
                         overlap=False)
    data_iter = DataLoader(dataset, batch_size=128, shuffle=True)
    print(len(data_iter) * data_iter.batch_size, list(data_iter)[0][0].shape)
