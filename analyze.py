import torch
import hydra
import matplotlib.pyplot as plt

from algorithms.norm import Norm
from algorithms.tent import Tent
from algorithms.delta import DELTA
from algorithms.divtent import DivTent
from datasets.tep_dataset import TEPDataset
from utils.hook_manager import HookManager
from torch.utils.data.dataloader import DataLoader
from utils.visualize import plot_embedding


class Analyze(object):
    def __init__(self, dataset, model, layer_name, algorithm):
        self.model = model
        self.algorithm = algorithm
        self.layer_name = layer_name
        self.dataset = dataset
        self._init_model(self._get_configs())

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
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.optim.learning_rate)
            self.model = Tent(self.model, optimizer, steps=args.algorithm.steps)
            self.layer_name = 'model.{}'.format(self.layer_name)
        elif self.algorithm == 'delta':
            self.model = DELTA(args.algorithm, self.model)
            self.layer_name = 'model.{}'.format(self.layer_name)
        elif self.algorithm == 'division':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.optim.learning_rate)
            self.model = DivTent(self.model, optimizer, steps=args.algorithm.steps,
                                 use_entropy=args.algorithm.use_entropy,
                                 weighting=args.algorithm.weighting)

    def embedding_analyze(self, **kwargs) -> None:
        data_iter = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        hook_manager = HookManager(self.model)
        hook_manager.register_hook(self.layer_name)
        title = f'{self.algorithm} [{self.layer_name}]'
        for x, y in data_iter:
            if torch.cuda.is_available():
                x = x.to(torch.device('cuda'))
            embeddings = hook_manager.get_activations(x)
            data = embeddings[self.layer_name].cpu()
            plot_embedding(data, y, title=title, **kwargs)
            plt.show()
        hook_manager.remove_hooks()


if __name__ == '__main__':
    pretrained_model = torch.load(r'./checkpoints/best_model_resnet_1.pth')
    tep_dataset = TEPDataset(src_path=r'./data/TEP',
                             split_ratio={'train': 0.7, 'eval': 0.1},
                             data_domains={'source': 1, 'target': 5},
                             dataset_mode='test',
                             data_dim=4,
                             transform=None,
                             overlap=True)
    analyze = Analyze(dataset=tep_dataset, model=pretrained_model, layer_name='conv5_x', algorithm=None)
    analyze.embedding_analyze()
