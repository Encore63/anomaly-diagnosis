import os

import pathlib
import numpy as np
import pandas as pd

from typing import Dict
from sklearn.preprocessing import StandardScaler


def digit_extract(file_name) -> int:
    return int(''.join(filter(str.isdigit, file_name)))


def data_concat(src_path: str, mode: int, num_data=600, time_win=10, neglect=None, num_classes=10) -> np.ndarray:
    """
    数据预处理及标准化
    :param src_path: 数据路径
    :param mode: 数据模式
    :param neglect: 忽略的类别 (choice 1 2 3 4 5 6)
    :param num_data: 数据量 (default 600)
    :param time_win: 时间窗口大小 (default 10)
    :param num_classes: 类别数量 (default 10)
    :return: 经预处理后的数据
    """
    scaler = StandardScaler()
    src_path = pathlib.Path(src_path).joinpath('mode{}_new'.format(mode))
    normal_data = pd.read_csv(src_path.joinpath('mode{}_d00.csv'.format(mode)))
    normal_data = normal_data.to_numpy()
    scaler.fit(normal_data[1:num_data, :])
    dataset = np.zeros((1, time_win, 51))
    count, idx_class = 0, 0
    class_ignore = set(neglect) if neglect is not None else set()
    for root, dirs, files in os.walk(src_path):
        sorted_files = sorted(files, key=digit_extract)
        for file in sorted_files:
            if file.split('.')[0][-2:] == '00':
                continue
            if idx_class in class_ignore:
                idx_class += 1
                continue
            file_path = src_path.joinpath(file)
            fault_data = pd.read_csv(file_path).to_numpy()
            fault_data = scaler.transform(fault_data)
            fault_data = np.concatenate((fault_data[0:num_data, :45],
                                         fault_data[0:num_data, 46:49],
                                         fault_data[0:num_data, 50:52]), axis=1)
            label = np.array([[idx_class]] * len(fault_data))
            fault_data = np.concatenate((fault_data, label), axis=1)
            data_temp = np.zeros(((len(fault_data) - time_win + 1), time_win, fault_data.shape[1]))
            for _i in range((len(fault_data) - time_win + 1)):
                for _j in range(time_win):
                    data_temp[_i, _j] = fault_data[_i + _j]
            dataset = np.concatenate((dataset, data_temp), axis=0)
            count += 1
            idx_class += 1
            if count >= num_classes:
                break
    dataset = dataset[1:dataset.shape[0], :, :]
    return dataset


def data_split(src_path: str, ratio: Dict, domains: Dict, random_seed: int, **kwargs) -> dict:
    """
    数据集划分
    :param src_path: 数据路径
    :param ratio: 划分比例
    :param domains: 数据域
    :param random_seed 随机数种子
    :return: 经划分后的数据集
    """
    assert sum(ratio.values()) <= 1, "Invalid ratio!"
    datasets = dict()
    data = data_concat(src_path, domains['source'], **kwargs)
    data_size = data.shape[0]
    train_size = int(data_size * ratio['train'])
    eval_size = int(data_size * ratio['eval'])
    np.random.seed(random_seed)
    np.random.shuffle(data)
    train_data = data[:train_size, :, :]
    eval_data = data[train_size:train_size + eval_size, :, :]
    datasets.setdefault('source_train', train_data)
    datasets.setdefault('source_eval', eval_data)
    if domains['target'] != domains['source'] and domains['target'] is not None:
        test_data = data_concat(src_path, domains['target'], **kwargs)
        np.random.seed(random_seed)
        np.random.shuffle(test_data)
        datasets.setdefault('target_test', test_data)
    else:
        datasets.setdefault('target_test', data[train_size + eval_size:, :, :])
    return datasets


class DataTransform(object):
    """
    Tools for data transformation or augmentation
    """
    def __init__(self, data: np.ndarray):
        self.data = data

    def gaussian(self, sigma: float = 0.01) -> np.ndarray:
        return self.data + np.random.normal(loc=0, scale=sigma, size=self.data.shape)

    def random_gaussian(self, sigma: float = 0.01):
        if np.random.randint(2):
            return self.data
        else:
            return self.data + np.random.normal(loc=0, scale=sigma, size=self.data.shape)

    def random_scale(self, sigma: float = 0.01):
        if np.random.randint(2):
            return self.data
        else:
            scale_factor = np.random.normal(loc=1, scale=sigma, size=(self.data.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, self.data.shape[1])))
            return self.data * scale_matrix


if __name__ == '__main__':
    c_data = data_concat(src_path=r'D:\PyProjects\FaultDiagnosis\data\TEP', mode=1)
    print(c_data.shape)
