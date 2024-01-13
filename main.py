import torch
import pathlib
import argparse

from default import cfg, merge_from_file
from utils.logger import get_time
from utils.logger import save_config
from models.mlp import MLP
from models.tenet import TENet, ReTENet
from datasets.tep_dataset import TEPDataset
from torch.utils.data.dataloader import DataLoader
from training_pipeline import train_default, train_with_learned_loss
from testing_pipeline import test_default, test_with_learned_loss, test_with_adaptive_norm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    if cfg.BASIC.LOG_FLAG:
        save_config(cfg, args.cfg_file)

    torch.manual_seed(cfg.BASIC.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.BASIC.RANDOM_SEED)
    torch.cuda.set_device(cfg.BASIC.CUDA_ID)

    if not pathlib.Path(cfg.PATH.DATA_PATH).exists():
        pathlib.Path(cfg.PATH.DATA_PATH).mkdir()
    if not pathlib.Path(cfg.PATH.LOG_PATH).exists():
        pathlib.Path(cfg.PATH.LOG_PATH).mkdir()
    if not pathlib.Path(cfg.PATH.CKPT_PATH).exists():
        pathlib.Path(cfg.PATH.CKPT_PATH).mkdir()

    if cfg.MODEL.NAME == 'ReTENet':
        model = ReTENet(f1=16, f2=32, depth=8, num_classes=cfg.MODEL.NUM_CLASSES).to(cfg.BASIC.DEVICE)
    else:
        model = TENet(f1=16, f2=32, depth=8, num_classes=cfg.MODEL.NUM_CLASSES).to(cfg.BASIC.DEVICE)

    if cfg.TRAINING.PIPELINE == 'arm_ll':
        ll_model = MLP(in_features=cfg.MODEL.NUM_CLASSES, hidden_dim=32,
                       out_features=1, norm_reduce=True).to(cfg.BASIC.DEVICE)
    else:
        ll_model = None

    criterion = torch.nn.CrossEntropyLoss()

    cfg.PATH.LOG_PATH = str(pathlib.Path(cfg.PATH.LOG_PATH).joinpath(f'{get_time()}'))

    split_ratio = {'train': cfg.BASIC.SPLIT_RATIO[0],
                   'eval': cfg.BASIC.SPLIT_RATIO[1]}

    datasets, dataloaders = {}, {}
    data_domains = {'source': int(cfg.BASIC.SOURCE), 'target': int(cfg.BASIC.TARGET)}
    datasets.setdefault('train', TEPDataset(cfg.PATH.DATA_PATH, split_ratio, data_domains,
                                            'train', seed=cfg.BASIC.RANDOM_SEED))
    datasets.setdefault('val', TEPDataset(cfg.PATH.DATA_PATH, split_ratio, data_domains,
                                          'eval', seed=cfg.BASIC.RANDOM_SEED))
    datasets.setdefault('test', TEPDataset(cfg.PATH.DATA_PATH, split_ratio, data_domains,
                                           'test', seed=cfg.BASIC.RANDOM_SEED))

    dataloaders.setdefault('train', DataLoader(datasets['train'], batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True))
    dataloaders.setdefault('val', DataLoader(datasets['val'], batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True))
    dataloaders.setdefault('test', DataLoader(datasets['test'], batch_size=cfg.TESTING.BATCH_SIZE, shuffle=False))

    if cfg.TRAINING.PIPELINE == 'default':
        train_default(train_iter=dataloaders['train'],
                      eval_iter=dataloaders['val'],
                      model=model,
                      criterion=criterion,
                      args=cfg)

    if cfg.TRAINING.PIPELINE == 'arm_ll':
        domains = [int(domain) for domain in cfg.BASIC.SOURCE]
        train_with_learned_loss(domains=domains,
                                model=model,
                                ll_model=ll_model,
                                criterion=criterion,
                                args=cfg)

    if cfg.TESTING.PIPELINE == 'default':
        test_default(test_iter=dataloaders['test'],
                     model_path=pathlib.Path(cfg.PATH.CKPT_PATH).joinpath(f'best_model_{cfg.MODEL.CKPT_SUFFIX}.pth'),
                     args=cfg)

    if cfg.TESTING.PIPELINE == 'norm':
        test_with_adaptive_norm(test_dataset=datasets['test'],
                                model_path=pathlib.Path(cfg.PATH.CKPT_PATH).
                                joinpath(f'best_model_{cfg.MODEL.CKPT_SUFFIX}.pth'),
                                criterion=criterion,
                                args=cfg)

    if cfg.TESTING.PIPELINE == 'arm_ll':
        test_with_learned_loss(test_iter=dataloaders['test'],
                               model_path=pathlib.Path(cfg.PATH.CKPT_PATH).joinpath(
                                   f'best_model_{cfg.MODEL.CKPT_SUFFIX}.pth'),
                               ll_model_path=pathlib.Path(cfg.PATH.CKPT_PATH).joinpath(f'learned_loss.pth'),
                               args=cfg)
