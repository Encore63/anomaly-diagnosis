import torch
import unittest as ut

from torch import nn
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


class TestAlgorithm(ut.TestCase):
    def setUp(self):
        self.source = 1
        self.target = 6

        '''
        0 - uncertain set for current mini-batch data
        1 - certain set for current mini-batch data
        '''
        self.data_type = 0
        self.algorithm = 'norm'
        self.model_name = 'resnet'
        self.learning_rate = 1e-3

    def test_negative_transfer(self):
        model = torch.load(rf'./checkpoints/best_model_{self.model_name}_{self.source}.pth')
        frozen = deepcopy(model)
        n_model = deepcopy(model)
        t_model = deepcopy(model)

        n_model = norm.Norm(n_model)

        t_model = tent.configure_model(t_model)
        params, param_names = tent.collect_params(t_model)
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        t_model = tent.Tent(t_model, optimizer)

        count, n_count, t_count = 0, 0, 0
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
                n_output = n_model(c_data)
                t_output = t_model(c_data)
                index = c_index
            else:
                output = model(u_data)
                n_output = n_model(u_data)
                t_output = t_model(u_data)
                index = u_index

            count += torch.eq(torch.argmax(output, 1), y[index]).float().mean()
            n_count += torch.eq(torch.argmax(n_output, 1), y[index]).float().mean()
            t_count += torch.eq(torch.argmax(t_output, 1), y[index]).float().mean()
        print(f'acc: {count / len(data_iter)}')
        print(f'n_acc: {n_count / len(data_iter)}')
        print(f't_acc: {t_count / len(data_iter)}')

    def test_param_sensitivity(self):
        from algorithms.comp import utr

        old_model = torch.load(rf'./checkpoints/best_model_{self.model_name}_{self.source}.pth')
        pre = deepcopy(old_model.state_dict())
        new_model = utr.capture_unc(pre, old_model)
        print(pre['conv1.0.weight'][0], new_model.state_dict()['conv1.0.weight'][0], sep='\n')

        old_count, new_count = 0, 0
        tep_dataset = get_dataset('tep', [[self.source], [self.target]], 'test')
        data_iter = DataLoader(dataset=tep_dataset, batch_size=256, shuffle=True)

        for x, y in data_iter:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            old_output = old_model(x)
            new_output = new_model(x)

            old_count += torch.eq(torch.argmax(old_output, 1), y).float().mean()
            new_count += torch.eq(torch.argmax(new_output, 1), y).float().mean()
        print(f'old_acc: {old_count / len(data_iter)}')
        print(f'new_acc: {new_count / len(data_iter)}')
        print(f'div: {old_count / len(data_iter) - new_count / len(data_iter)}')

    def test_normalization(self):
        model = torch.load(rf'./checkpoints/best_model_{self.model_name}_{self.source}.pth')
        source = deepcopy(model)
        c_model = deepcopy(model)
        u_model = deepcopy(model)
        if self.algorithm == 'norm':
            model = norm.Norm(model)
            c_model = norm.Norm(c_model)
            u_model = norm.Norm(u_model)
        elif self.algorithm == 'tent':
            model = tent_init(model)
            c_model = tent_init(c_model)
            u_model = tent_init(u_model)

        s_count, a_count, c_count, u_count = 0, 0, 0, 0
        tep_dataset = get_dataset('tep', [[self.source], [self.target]], 'test')
        data_iter = DataLoader(dataset=tep_dataset, batch_size=256, shuffle=False)
        for x, y in data_iter:
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
            # if self.algorithm == 'tent':
            #     for m in c_model.modules():
            #         if isinstance(m, nn.BatchNorm2d):
            #             m.requires_grad_(True)
            #     for m in u_model.modules():
            #         if isinstance(m, nn.BatchNorm2d):
            #             m.requires_grad_(True)
            rand_idx = torch.randperm(256)[:128]

            c_output = c_model(x[rand_idx])
            u_output = u_model(x[rand_idx])
            a_output = model(x[rand_idx])
            s_output = source(x[rand_idx])

            c_count += torch.eq(torch.argmax(c_output, 1), y[rand_idx]).float().mean()
            u_count += torch.eq(torch.argmax(u_output, 1), y[rand_idx]).float().mean()
            s_count += torch.eq(torch.argmax(s_output, 1), y[rand_idx]).float().mean()
            a_count += torch.eq(torch.argmax(a_output, 1), y[rand_idx]).float().mean()
        print(f'acc (normed by statistics of certain data): {c_count / len(data_iter)}')
        print(f'acc (normed by statistics of uncertain data): {u_count / len(data_iter)}')
        print(f'acc (normed by statistics of all data): {a_count / len(data_iter)}')
        print(f'acc (without adaptation): {s_count / len(data_iter)}')

    def test_gmm(self):
        from datasets.tep_dataset import TEPDataset
        from sklearn.mixture import GaussianMixture
        from algorithms.comp.jmds import joint_model_data_score

        model = torch.load(rf'./checkpoints/best_model_{self.model_name}_{self.source}.pth')
        source = deepcopy(model).eval()
        gmm = GaussianMixture(n_components=10)

        tep_dataset = TEPDataset(r'./data/TEP', transfer_task=[[self.source], [self.target]])
        subset = tep_dataset.get_subset()
        source_data, source_label = subset['train'].data, subset['train'].labels
        target_data = subset['test']

        data_iter = DataLoader(dataset=target_data, batch_size=128, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        gmm.fit(source_data.view(source_data.shape[0], -1), source_label)

        for x, y in data_iter:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            gmm_output = gmm.predict(x.view(x.shape[0], -1).cpu().numpy())
            # _, gmm_output = joint_model_data_score(x, source, 'avg_pool', 10)
            # model_output = model(x)
            print(gmm_output, y, sep='\n')
            print(joint_model_data_score(x, source, 'avg_pool', 10)[1])
            print(torch.tensor(gmm_output) == y.cpu())
            print((torch.tensor(gmm_output) == y.cpu()).sum(0) / 128)
            break
