import torch
import hydra
import matplotlib.pyplot as plt

from pprint import pprint
from algorithms import tent
from einops import rearrange
from algorithms import divtent
from copy import copy, deepcopy
from algorithms.norm import Norm
from algorithms.delta import DELTA
from utils.hook_manager import HookManager
from utils.visualize import plot_embedding
from utils.visualize import plot_confusion_matrix
from utils.visualize import plot_similarity_matrix
from datasets.tep_dataset import TEPDataset
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix as cm


class Analyze(object):
    def __init__(self, dataset, model, layer_name, algorithm):
        self.model = model
        self.algorithm = algorithm
        self.layer_name = layer_name
        self.dataset = dataset
        self.args = self._get_configs()
        self._init_model(self.args)

    @staticmethod
    def _get_configs():
        with hydra.initialize(version_base=None, config_path='./configs', job_name='analyze'):
            global_configs = hydra.compose(config_name='config')
        return global_configs

    def _init_model(self, args):
        if not self.algorithm:
            self.model.eval()
        elif self.algorithm == 'norm':
            self.model = Norm(self.model)
            self.layer_name = 'model.{}'.format(self.layer_name)
        elif self.algorithm == 'tent':
            params, param_names = tent.collect_params(self.model)
            optimizer = torch.optim.Adam(params, lr=args.optim.learning_rate)
            self.model = tent.Tent(self.model, optimizer, steps=args.algorithm.steps)
            self.layer_name = 'model.{}'.format(self.layer_name)
        elif self.algorithm == 'delta':
            self.model = DELTA(args.algorithm, self.model)
            self.layer_name = 'model.{}'.format(self.layer_name)
        elif self.algorithm == 'division':
            params, param_names = divtent.collect_params(self.model)
            optimizer = torch.optim.Adam(params, lr=args.optim.learning_rate)
            self.model = divtent.DivTent(self.model, optimizer, steps=args.algorithm.steps,
                                         use_entropy=args.algorithm.use_entropy,
                                         weighting=args.algorithm.weighting)
            self.layer_name = 'model.{}'.format(self.layer_name)

    def embedding_analyze(self, show_modules=False, **kwargs) -> None:
        data_iter = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        hook_manager = HookManager(self.model)
        if show_modules:
            pprint(hook_manager.get_modules())
            return None
        hook_manager.register_hook(self.layer_name)
        title = f'{self.algorithm} [{self.layer_name}]'
        for x, y in data_iter:
            if torch.cuda.is_available():
                x = x.to(torch.device('cuda'))
            if self.algorithm in {'tent', 'division', 'delta'}:
                # update model parameters
                self.model(x)
            embeddings = hook_manager.get_activations(x)
            data = embeddings[self.layer_name].cpu()
            print(data.shape, y.shape)
            plot_embedding(data, y, title=title, **kwargs)
            plt.show()
        hook_manager.remove_hooks()

    def confusion_matrix(self):
        data_iter = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        for x, y in data_iter:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            confusion_matrix = cm(pred, y)
            plot_confusion_matrix(confusion_matrix, self.args.model.num_classes)

    @staticmethod
    def domain_divergence(data_path, domains, dim):
        from torch.nn import BatchNorm1d
        from utils.data_utils import data_concat
        from torch.distributions import Normal, MultivariateNormal, kl_divergence

        data, normed_data = list(), list()
        mean_var = list()
        for domain in domains:
            data.append(data_concat(src_path=data_path, mode=domain, num_classes=10)[:, :, :-1])
        # bn = BatchNorm1d(dim, momentum=1)
        for domain_data in data:
            domain_data = torch.Tensor(domain_data)
            if dim != domain_data.shape[1]:
                domain_data = rearrange(domain_data, 'B L E -> B E L')
            # bn(domain_data)
            # mean, var = bn.running_mean.detach(), bn.running_var.detach()
            mean_var.append((deepcopy(domain_data.mean()), deepcopy(domain_data.var())))
            # bn.reset_parameters()

        distributions = list()
        for m, v in mean_var:
            distributions.append(Normal(m, v))

        sim_matrix = torch.zeros((len(domains), len(domains)))
        for dx in range(len(distributions)):
            for dy in range(len(distributions)):
                # 使用对称KL散度度量域间差异
                sim_matrix[dx, dy] = (1 / 2 * kl_divergence(distributions[dx], distributions[dy]) +
                                      1 / 2 * kl_divergence(distributions[dy], distributions[dx]))
        print(sim_matrix)
        plot_similarity_matrix(sim_matrix, cmap='Blues')


if __name__ == '__main__':
    pretrained_model = torch.load(r'checkpoints/best_model_resnet_1.pth')
    tep_dataset = TEPDataset(src_path=r'./data/TEP', transfer_task=[[1], [2]]).get_subset('test')
    analyze = Analyze(dataset=tep_dataset, model=pretrained_model,
                      layer_name='conv5_x', algorithm='tent')
    analyze.embedding_analyze()
    # analyze.domain_divergence(data_path=r'./data/TEP', domains=[1, 2, 3, 4, 5, 6], dim=50)
