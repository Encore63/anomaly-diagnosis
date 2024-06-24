import os
import pandas as pd

from tqdm import tqdm
from scipy.io import loadmat
from utils.seq_augmentor import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class SequenceDataset(Dataset):

    def __init__(self, list_data, test=False, data_dim=2, transform=None):
        self.test = test
        self.data_dim = data_dim
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            for _ in range(self.data_dim - len(seq.shape)):
                seq = np.expand_dims(seq, 0)
            return seq, item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            for _ in range(self.data_dim - len(seq.shape)):
                seq = np.expand_dims(seq, 0)
            return seq, label


signal_size = 1024
root = '../data/CWRU'
data_name = {
    0: ["97.mat", "105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat", "234.mat"],
    # 1797rpm
    1: ["98.mat", "106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat", "235.mat"],
    # 1772rpm
    2: ["99.mat", "107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat", "236.mat"],
    # 1750rpm
    3: ["100.mat", "108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
        "237.mat"]}  # 1730rpm

dataset_name = ["12k_Drive_End_Bearing_Fault_Data", "12k_Fan_End_Bearing_Fault_Data",
                "48k_Drive_End_Bearing_Fault_Data",
                "Normal_Baseline_Data"]
axis = ["_DE_time", "_FE_time", "_BA_time"]

label = [i for i in range(0, 10)]


def get_files(root, N):  # N为转速，N传进来是一个代表负载
    """
    This function is used to generate the final training set and test set.
    root:The location of the data set
    """
    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(data_name[N[k]]))):
            if n == 0:
                path1 = os.path.join(root, dataset_name[3], data_name[N[k]][n])
            else:
                path1 = os.path.join(root, dataset_name[0], data_name[N[k]][n])
            data1, lab1 = data_load(path1, data_name[N[k]][n], _label=label[n])
            data += data1
            lab += lab1

    return [data, lab]


def data_load(filename, axis_name, _label):
    """
    This function is mainly used to generate test data and training data.
    filename:Data location
    axis_name:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    """
    data_number = axis_name.split(".")
    if eval(data_number[0]) < 100:
        real_axis = "X0" + data_number[0] + axis[0]
    else:
        real_axis = "X" + data_number[0] + axis[0]
    fl = loadmat(filename)[real_axis]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(_label)
        start += signal_size
        end += signal_size

    return data, lab


class CWRUDataset(object):
    num_classes = 10
    input_channel = 1

    def __init__(self, data_dir, transfer_task, normalize_type="0-1", data_dim=2):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normalize_type = normalize_type
        self.data_dim = data_dim
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normalize_type),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normalize_type),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = SequenceDataset(list_data=train_pd, data_dim=self.data_dim, transform=self.data_transforms['train'])
            source_val = SequenceDataset(list_data=val_pd, data_dim=self.data_dim, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = SequenceDataset(list_data=train_pd, data_dim=self.data_dim, transform=self.data_transforms['train'])
            target_val = SequenceDataset(list_data=val_pd, data_dim=self.data_dim, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = SequenceDataset(list_data=train_pd, data_dim=self.data_dim, transform=self.data_transforms['train'])
            source_val = SequenceDataset(list_data=val_pd, data_dim=self.data_dim, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = SequenceDataset(list_data=data_pd, data_dim=self.data_dim, transform=self.data_transforms['val'])
            return source_train, source_val, target_val


if __name__ == '__main__':
    from easydict import EasyDict as eDict
    dataset_tool = CWRUDataset(data_dir=root, transfer_task=[[0], [1]], normalize_type="0-1", data_dim=3)
    source, target = eDict(d={'train': None, 'val': None}), eDict(d={'train': None, 'val': None})
    source['train'], source['val'], target['train'], target['val'] = dataset_tool.data_split(transfer_learning=True)
    print(type(source.val))
