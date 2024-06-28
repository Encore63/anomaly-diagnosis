import torch
import hydra
import matplotlib.pyplot as plt

from pprint import pprint
from algorithms import tent
from algorithms import divtent
from algorithms.norm import Norm
from algorithms.delta import DELTA
from utils.hook_manager import HookManager
from utils.visualize import plot_embedding
from utils.visualize import plot_confusion_matrix
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
            plt.show()


if __name__ == '__main__':
    pretrained_model = torch.load(r'checkpoints/best_model_resnet_1.pth')
    tep_dataset = TEPDataset(src_path=r'./data/TEP',
                             split_ratio={'train': 0.7, 'eval': 0.1},
                             data_domains={'source': 1, 'target': 3},
                             dataset_mode='test',
                             data_dim=4,
                             transform=None,
                             overlap=True)
    analyze = Analyze(dataset=tep_dataset, model=pretrained_model,
                      layer_name='conv5_x', algorithm='division')
    # analyze.embedding_analyze()
    analyze.confusion_matrix()
