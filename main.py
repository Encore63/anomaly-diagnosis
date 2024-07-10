import hydra
import pathlib

from omegaconf import DictConfig
from utils.logger import get_time
from models.resnet import resnet
from models.tenet import ReTENet
from models.cnn import CNN
from models.dagcn import DAGCN
from models.convformer import LiConvFormer
from utils.training_pipeline import *
from utils.testing_pipeline import *


@hydra.main(version_base=None,
            config_path="./configs",
            config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.cuda.set_device(cfg.cuda_id)

    if not pathlib.Path(cfg.dataset.path).exists():
        pathlib.Path(cfg.dataset.path).mkdir()
    if not pathlib.Path(cfg.log_path).exists():
        pathlib.Path(cfg.log_path).mkdir()
    if not pathlib.Path(cfg.model.save_path).exists():
        pathlib.Path(cfg.model.save_path).mkdir()

    if cfg.model.name == 'TENet':
        model = ReTENet(num_classes=cfg.model.num_classes).to(cfg.device)
    elif cfg.model.name == 'CNN':
        model = CNN(in_channels=cfg.dataset.time_window).to(cfg.device)
    elif cfg.model.name == 'ConvFormer':
        model = LiConvFormer(use_residual=cfg.model.use_residual,
                             in_channel=cfg.model.in_channels,
                             out_channel=cfg.model.num_classes).to(cfg.device)
    elif cfg.model.name == 'DAGCN':
        model = DAGCN(in_channels=cfg.model.in_channels,
                      num_classes=cfg.model.num_classes,
                      pretrained=cfg.model.pretrained).to(cfg.device)
    else:
        model = resnet(in_channels=cfg.model.in_channels,
                       num_classes=cfg.model.num_classes,
                       num_block=cfg.model.num_block).to(cfg.device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.util.train.label_smoothing)

    cfg.log_path = str(pathlib.Path(cfg.log_path).joinpath(f'{get_time()}'))

    datasets, dataloaders = {}, {}
    data_domains = [[int(cfg.domain.source)], [int(cfg.domain.target)]]
    dataset_tool = TEPDataset(cfg.dataset.path,
                              transfer_task=data_domains,
                              transfer=cfg.dataset.transfer,
                              seed=cfg.random_seed,
                              data_dim=cfg.dataset.dim,
                              time_win=cfg.dataset.time_window,
                              overlap=cfg.dataset.overlap)
    datasets = dataset_tool.get_subset()

    dataloaders.setdefault('train', DataLoader(datasets['train'], batch_size=cfg.util.train.batch_size, shuffle=True))
    dataloaders.setdefault('val', DataLoader(datasets['val'], batch_size=cfg.util.train.batch_size, shuffle=True))
    dataloaders.setdefault('test', DataLoader(datasets['test'], batch_size=cfg.util.test.batch_size, shuffle=False))

    if cfg.util.train.pipeline == 'default':
        train_default(train_iter=dataloaders['train'],
                      eval_iter=dataloaders['val'],
                      model=model,
                      criterion=criterion,
                      args=cfg)

    if cfg.util.train.pipeline == 'arm':
        domains = [int(domain) for domain in cfg.domain.sources]
        train_with_arm(domains=domains,
                       model=model,
                       criterion=criterion,
                       args=cfg)

    model_name = f'best_model_{cfg.model.suffix}_{cfg.domain.source}.pth' \
        if cfg.model.suffix != '' \
        else 'best_model.pth'

    for testing_pipeline in cfg.util.test.pipeline:
        if testing_pipeline == 'default':
            test_default(test_iter=dataloaders['test'],
                         model_path=pathlib.Path(cfg.model.save_path).joinpath(model_name),
                         args=cfg)

        if testing_pipeline == 'norm':
            test_with_adaptive_norm(test_iter=dataloaders['test'],
                                    model_path=pathlib.Path(cfg.model.save_path).joinpath(model_name),
                                    args=cfg)

        if testing_pipeline == 'tent':
            test_with_tent(test_iter=dataloaders['test'],
                           model_path=pathlib.Path(cfg.model.save_path).joinpath(model_name),
                           args=cfg)

        if testing_pipeline == 'arm':
            test_with_arm(test_iter=dataloaders['test'],
                          model_path=pathlib.Path(cfg.model.save_path).joinpath(model_name),
                          args=cfg)

        if testing_pipeline == 'delta':
            test_with_delta(test_iter=dataloaders['test'],
                            model_path=pathlib.Path(cfg.model.save_path).joinpath(model_name),
                            args=cfg)

        if testing_pipeline == 'division':
            test_with_data_division(test_iter=dataloaders['test'],
                                    model_path=pathlib.Path(cfg.model.save_path).joinpath(model_name),
                                    args=cfg)


if __name__ == '__main__':
    main()
