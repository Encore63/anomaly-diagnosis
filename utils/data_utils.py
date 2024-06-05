import os
import torch
import pathlib
import numpy as np
import pandas as pd

from typing import Dict
from sklearn.preprocessing import StandardScaler


def digit_extract(file_name) -> int:
    return int(''.join(filter(str.isdigit, file_name)))


def data_concat(src_path: str, mode: int, num_data=600, time_win=10,
                neglect=None, num_classes=10, overlap=True) -> np.ndarray:
    """
    数据预处理及标准化
    :param src_path: 数据路径
    :param mode: 数据模式 (choice 1 2 3 4 5 6)
    :param neglect: 忽略的类别
    :param num_data: 数据量 (default 600)
    :param time_win: 时间窗口大小 (default 10)
    :param num_classes: 类别数量 (default 10)
    :param overlap: 数据重叠 (default True)
    :return: 经预处理后的数据
    """
    scaler = StandardScaler()
    src_path = pathlib.Path(src_path).joinpath('mode{}_new'.format(mode))
    normal_data = pd.read_csv(src_path.joinpath('mode{}_d00.csv'.format(mode)))
    normal_data = normal_data.to_numpy()
    scaler.fit(normal_data[1:num_data, :])
    dataset = np.zeros((1, time_win, 51)) if overlap else np.zeros((1, 51))
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
            if overlap:
                data_temp = np.zeros(((len(fault_data) - time_win + 1), time_win, fault_data.shape[1]))
                for _i in range((len(fault_data) - time_win + 1)):
                    for _j in range(time_win):
                        data_temp[_i, _j] = fault_data[_i + _j]
            else:
                data_temp = fault_data
            dataset = np.concatenate((dataset, data_temp), axis=0)
            count += 1
            idx_class += 1
            if count >= num_classes:
                break
    if overlap:
        dataset = dataset[1:dataset.shape[0], :, :]
    else:
        assert time_win != 0, 'Invalid size of time window.'
        dataset = dataset[1:dataset.shape[0], :]
        dataset = dataset.reshape((-1, time_win, dataset.shape[-1]))
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


def domain_division(model, data, p_threshold: float = None, use_entropy: bool = False, weighting: bool = False,
                    **kwargs):
    """
    数据域划分
    :param model: 模型
    :param data: 数据
    :param p_threshold: 概率阈值
    :param use_entropy: 是否使用谱熵
    :param weighting: 是否使用权重
    :return: 经划分后的数据集
    """
    from scipy.stats import entropy

    transfer = False
    batch_size = data.shape[0]
    with torch.no_grad():
        pred = model(data)
        max_prob, max_idx = torch.max(pred, dim=-1, **kwargs)

    if not p_threshold:
        p_threshold = max_prob.sum() / batch_size  # max_prob.mean()

    # 计算谱熵
    if use_entropy:
        from antropy import spectral_entropy

        if data.shape[1] == 1:
            transfer = True
            data = data.reshape((-1, data.shape[2], data.shape[3]))
        # e = data * 2
        # e = e / e.sum(dim=2, keepdim=True)
        # e = entropy(e.cpu(), axis=1)
        e = spectral_entropy(data.cpu().numpy(), sf=0.006, axis=1)
        e_threshold = e.mean()
        e_mask = torch.from_numpy(e).mean(1).ge(e_threshold).int().cpu()
    else:
        e_mask = torch.zeros(batch_size)

    p_mask = max_prob.ge(p_threshold).int().cpu()
    mask = p_mask if not use_entropy else p_mask | e_mask
    src_idx = torch.argwhere(mask == 1)
    tgt_idx = torch.argwhere(mask == 0)
    src_idx = src_idx.view(src_idx.shape[0])
    tgt_idx = tgt_idx.view(tgt_idx.shape[0])

    if transfer:
        data = data.unsqueeze(dim=1)
    source_data = data[src_idx]
    target_data = data[tgt_idx]
    if weighting:
        with torch.no_grad():
            src_logit, tgt_logit = model(source_data), model(target_data)
            src_w = -(torch.softmax(src_logit, dim=-1) * torch.log_softmax(src_logit, dim=-1)).sum(-1).mean(0)
            tgt_w = -(torch.softmax(tgt_logit, dim=-1) * torch.log_softmax(tgt_logit, dim=-1)).sum(-1).mean(0)
            weight = torch.softmax(torch.Tensor([src_w, tgt_w]), dim=0)
            # print(weight, src_logit.shape, tgt_logit.shape)
            source_data *= weight[0]
            target_data *= weight[1]
    return source_data, target_data, src_idx, tgt_idx


def domain_merge(source_data, target_data, source_index, target_index):
    source_index = list(source_index.cpu().numpy())
    target_index = list(target_index.cpu().numpy())
    order = source_index + target_index
    order.sort()

    s_idx, t_idx = 0, 0
    result = torch.zeros_like(torch.concat([source_data, target_data], dim=0)).cpu()
    for i, var in enumerate(order):
        if var in source_index:
            result[i] = source_data[s_idx]
            s_idx += 1
        elif var in target_index:
            result[i] = target_data[t_idx]
            t_idx += 1
    return result.cuda()


if __name__ == '__main__':
    # c_data = data_concat(src_path=r'F:\StudyFiles\PyProjects\AnomalyDiagnosis\data\TEP',
    #                      mode=1, time_win=10, overlap=True)
    # print(c_data.shape)

    model = torch.load(r'../checkpoints/best_model.pth')
    x = torch.randn((64, 1, 10, 50)).to('cuda')
    print(model(x).shape)
    # src_data, tgt_data = domain_division(model=model,
    #                                      data=x,
    #                                      use_entropy=True)
