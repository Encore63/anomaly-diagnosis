import os

import pathlib
import numpy as np
import pandas as pd

from typing import Dict
from sklearn.preprocessing import StandardScaler


def data_concat(src_path: str, mode: int, num_data=600, time_win=10) -> np.ndarray:
    """
    数据预处理及标准化
    :param src_path: 数据路径
    :param mode: 数据模式
    :param num_data: 数据量 (default 600)
    :param time_win: 时间窗口大小 (default 10)
    :return: 经预处理后的数据
    """
    scaler = StandardScaler()
    src_path = pathlib.Path(src_path).joinpath('mode{}_new'.format(mode))
    normal_data = pd.read_csv(src_path.joinpath('mode{}_d00.csv'.format(mode)))
    normal_data = normal_data.to_numpy()
    scaler.fit(normal_data[1:num_data, :])
    dataset = np.zeros((1, time_win, 51))
    idx_class = 0
    for root, dirs, files in os.walk(src_path):
        for file in files:
            file_path = src_path.joinpath(file)
            data = pd.read_csv(file_path).to_numpy()
            data = scaler.transform(data)
            data = np.concatenate((data[0:num_data, :45],
                                   data[0:num_data, 46:49],
                                   data[0:num_data, 50:52]), axis=1)
            label = np.array([[idx_class]] * len(data))
            data = np.concatenate((data, label), axis=1)
            data_temp = np.zeros(((len(data) - time_win + 1), time_win, data.shape[1]))
            for _i in range((len(data) - time_win + 1)):
                for _j in range(time_win):
                    data_temp[_i, _j] = data[_i + _j]
            dataset = np.concatenate((dataset, data_temp), axis=0)
            idx_class += 1
    dataset = dataset[1:dataset.shape[0], :, :]
    return dataset


def data_split(src_path: str, ratio: Dict, domains: Dict, **kwargs) -> dict:
    """
    数据集划分
    :param src_path: 数据路径
    :param ratio: 划分比例
    :param domains: 数据域
    :return: 经划分后的数据集
    """
    assert sum(ratio.values()) <= 1
    datasets = dict()
    data = data_concat(src_path, domains['source'], **kwargs)
    data_size = data.shape[0]
    train_size = int(data_size * ratio['train'])
    eval_size = int(data_size * ratio['eval'])
    train_data = data[:train_size, :, :]
    eval_data = data[train_size:train_size + eval_size, :, :]
    datasets.setdefault('source_train', train_data)
    datasets.setdefault('source_eval', eval_data)
    datasets.setdefault('target_test', data_concat(src_path, domains['target'], **kwargs))
    return datasets


if __name__ == '__main__':
    res_data = data_concat(src_path=r'D:\PyProjects\FaultDiagnosis\data\TEP', mode=1)
    print(res_data[:, :, -1].shape)
