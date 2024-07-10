import torch
import unittest as ut

from copy import deepcopy
from algorithms import norm, tent
from datasets.getter import get_dataset
from torch.utils.data.dataloader import DataLoader
from utils.data_utils import domain_division, find_thresh


class TestAlgorithm(ut.TestCase):
    def setUp(self):
        self.source = 1
        self.target = 2

        '''
        0 - uncertain set for current mini-batch data
        1 - certain set for current mini-batch data
        '''
        self.data_type = 1
        self.algorithm = 'norm'
        self.model_name = 'resnet'
        self.learning_rate = 1e-3

    def test_negative_transfer(self):
        model = torch.load(rf'./checkpoints/best_model_{self.model_name}_{self.source}.pth')
        frozen = deepcopy(model)
        if self.algorithm == 'norm':
            model = norm.Norm(model)
        elif self.algorithm == 'tent':
            model = tent.configure_model(model)
            params, param_names = tent.collect_params(model)
            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
            model = tent.Tent(model, optimizer)

        count = 0
        tep_dataset = get_dataset('tep', [[self.source], [self.target]], 'test')
        data_iter = DataLoader(dataset=tep_dataset, batch_size=256, shuffle=False)
        for x, y in data_iter:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred = torch.softmax(frozen(x), dim=1)
            thresh = find_thresh(pred)
            c_data, u_data, c_index, u_index, w = domain_division(frozen, x, use_entropy=False,
                                                                  weighting=True, p_threshold=thresh)
            c_data, u_data = c_data.cuda(), u_data.cuda()
            if self.data_type == 1:
                output = model(c_data)
                index = c_index
            else:
                output = model(u_data)
                index = u_index

            count += torch.eq(torch.argmax(output, 1), y[index]).float().mean()
        print(f'acc: {count / len(data_iter)}')
