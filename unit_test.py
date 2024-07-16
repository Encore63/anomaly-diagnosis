import os.path

import torch
import unittest as ut

from copy import deepcopy
from algorithms import norm, tent
from datasets.getter import get_dataset
from torch.utils.data.dataloader import DataLoader
from utils.data_utils import domain_division, find_thresh


def tent_init(_model):
    _model = tent.configure_model(_model)
    _params, _ = tent.collect_params(_model)
    _optimizer = torch.optim.Adam(_params, lr=1e-3)
    _model = tent.Tent(_model, _optimizer)
    return _model


# class TestAlgorithm(ut.TestCase):
#     def setUp(self):
#         self.source = 1
#         self.target = 2
#
#         '''
#         0 - uncertain set for current mini-batch data
#         1 - certain set for current mini-batch data
#         '''
#         self.data_type = 1
#         self.algorithm = 'norm'
#         self.model_name = 'resnet'
#         self.learning_rate = 1e-3
#
#     def test_negative_transfer(self):
#         model = torch.load(rf'./checkpoints/best_model_{self.model_name}_{self.source}.pth')
#         frozen = deepcopy(model)
#         if self.algorithm == 'norm':
#             model = norm.Norm(model)
#         elif self.algorithm == 'tent':
#             model = tent.configure_model(model)
#             params, param_names = tent.collect_params(model)
#             optimizer = torch.optim.Adam(params, lr=self.learning_rate)
#             model = tent.Tent(model, optimizer)
#
#         count = 0
#         tep_dataset = get_dataset('tep', [[self.source], [self.target]], 'test')
#         data_iter = DataLoader(dataset=tep_dataset, batch_size=256, shuffle=False)
#         for x, y in data_iter:
#             if torch.cuda.is_available():
#                 x, y = x.cuda(), y.cuda()
#             pred = torch.softmax(frozen(x), dim=1)
#             thresh = find_thresh(pred)
#             c_data, u_data, c_index, u_index, w = domain_division(frozen, x, use_entropy=False,
#                                                                   weighting=True, p_threshold=thresh)
#             c_data, u_data = c_data.cuda(), u_data.cuda()
#             if self.data_type == 1:
#                 output = model(c_data)
#                 index = c_index
#             else:
#                 output = model(u_data)
#                 index = u_index
#
#             count += torch.eq(torch.argmax(output, 1), y[index]).float().mean()
#         print(f'acc: {count / len(data_iter)}')
#
#     def test_param_sensitivity(self):
#         from algorithms.comp import utr
#
#         old_model = torch.load(rf'./checkpoints/best_model_{self.model_name}_{self.source}.pth')
#         pre = deepcopy(old_model.state_dict())
#         new_model = utr.capture_unc(pre, old_model)
#         print(pre['conv1.0.weight'][0], new_model.state_dict()['conv1.0.weight'][0], sep='\n')
#
#         old_count, new_count = 0, 0
#         tep_dataset = get_dataset('tep', [[self.source], [self.target]], 'test')
#         data_iter = DataLoader(dataset=tep_dataset, batch_size=256, shuffle=False)
#
#         for x, y in data_iter:
#             if torch.cuda.is_available():
#                 x, y = x.cuda(), y.cuda()
#             old_output = old_model(x)
#             new_output = new_model(x)
#
#             old_count += torch.eq(torch.argmax(old_output, 1), y).float().mean()
#             new_count += torch.eq(torch.argmax(new_output, 1), y).float().mean()
#         print(f'old_acc: {old_count / len(data_iter)}')
#         print(f'new_acc: {new_count / len(data_iter)}')
#         print(f'div: {old_count / len(data_iter) - new_count / len(data_iter)}')
#
#     def test_normalization(self):
#         model = torch.load(r'F:\StudyFiles\PyProjects\AnomalyDiagnosis\checkpoints\best_model_resnet_1.pth')
#         source = deepcopy(model)
#         c_model = deepcopy(model)
#         u_model = deepcopy(model)
#         if self.algorithm == 'norm':
#             model = norm.Norm(model)
#             c_model = norm.Norm(c_model)
#             u_model = norm.Norm(u_model)
#         elif self.algorithm == 'tent':
#             model = tent.configure_model(model)
#             params, param_names = tent.collect_params(model)
#             optimizer = torch.optim.Adam(params, lr=self.learning_rate)
#             model = tent.Tent(model, optimizer)
#
#         c_count, u_count = 0, 0
#         tep_dataset = get_dataset('tep', [[self.source], [self.target]], 'test')
#         data_iter = DataLoader(dataset=tep_dataset, batch_size=256, shuffle=False)
#         for x, y in data_iter:
#             if torch.cuda.is_available():
#                 x, y = x.cuda(), y.cuda()
#             pred = torch.softmax(source(x), dim=1)
#             thresh = find_thresh(pred)
#             c_data, u_data, c_index, u_index, w = domain_division(source, x, use_entropy=False,
#                                                                   weighting=True, p_threshold=thresh)
#             c_data, u_data = c_data.cuda(), u_data.cuda()
#
#             _, _ = c_model(c_data), u_model(u_data)
#             c_model.eval()
#             u_model.eval()
#
#             c_output = c_model(x)
#             u_output = u_model(x)
#
#             c_count += torch.eq(torch.argmax(c_output, 1), y).float().mean()
#             u_count += torch.eq(torch.argmax(u_output, 1), y).float().mean()
#         print(f'c_acc: {c_count / len(data_iter)}')
#         print(f'u_acc: {u_count / len(data_iter)}')


if __name__ == '__main__':
    algorithm = 'norm'
    model = torch.load(r'F:\StudyFiles\PyProjects\AnomalyDiagnosis\checkpoints\best_model_resnet_1.pth')
    source = deepcopy(model)
    c_model = deepcopy(model)
    u_model = deepcopy(model)
    if algorithm == 'norm':
        model = norm.Norm(model)
    elif algorithm == 'tent':
        model = tent.configure_model(model)
        params, param_names = tent.collect_params(model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        model = tent.Tent(model, optimizer)

    s_count, count, c_count, u_count = 0, 0, 0, 0
    tep_dataset = get_dataset('tep', [[1], [1]], 'test')
    data_iter = DataLoader(dataset=tep_dataset, batch_size=256, shuffle=False)
    for x, y in data_iter:
        c_model = norm.Norm(c_model)
        u_model = norm.Norm(u_model)
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        pred = torch.softmax(source(x), dim=1)
        thresh = find_thresh(pred)
        c_data, u_data, c_index, u_index, w = domain_division(source, x, use_entropy=False,
                                                              weighting=True, p_threshold=thresh)
        c_data, u_data = c_data.cuda(), u_data.cuda()

        _, _ = c_model(c_data), u_model(u_data)
        c_model.eval()
        u_model.eval()

        c_output = c_model(x)
        u_output = u_model(x)
        output = model(x)
        # s_output = source(x)

        c_count += torch.eq(torch.argmax(c_output, 1), y).float().mean()
        u_count += torch.eq(torch.argmax(u_output, 1), y).float().mean()
        # s_count += torch.eq(torch.argmax(s_output, 1), y).float().mean()
        count += torch.eq(torch.argmax(output, 1), y).float().mean()
    print(f'acc (normed by statistics of certain data): {c_count / len(data_iter)}')
    print(f'acc (normed by statistics of uncertain data): {u_count / len(data_iter)}')
    print(f'acc (normed by statistics of all data): {count / len(data_iter)}')
    print(f'acc (without adaptation): {s_count / len(data_iter)}')
