import torch
import numpy as np


@torch.jit.script
def fuzzy_entropy(x, m=2, r=0.25, n=2):
    """
    模糊熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    n 计算模糊隶属度时的维度
    """
    # 将x转化为数组
    x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x should be a 1D tensor.")
    # 计算x的行数是否小于m+1
    if len(x) < m + 1:
        raise ValueError("length of x should be less than m + 1.")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        _x = []
        for i in range(len(x) - m + 1 - temp):
            _x.append(x[i:i + m + temp])
        _x = np.array(_x)
        # 计算X任意一行数据与其他行数据对应索引数据的差值绝对值的最大值
        d_value = []  # 存储差值
        for index1, i in enumerate(_x):
            sub = []
            for index2, j in enumerate(_x):
                if index1 != index2:
                    sub.append(max(np.abs(i - j)))
            d_value.append(sub)
        # 计算模糊隶属度
        D = np.exp(-np.power(d_value, n) / r)
        # 计算所有隶属度的平均值
        lm = np.average(D.ravel())
        entropy = abs(entropy) - lm

    return entropy
