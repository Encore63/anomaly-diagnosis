import pathlib
import argparse

from default import cfg, merge_from_file
from utils.logger import get_time
from utils.logger import save_log
from models.bilstm import BiLSTM
from models.resnet import resnet18
from models.tenet import TENet, ReTENet
from models.dagcn import DAGCN
from models.cnn import CNN
from models.convformer import ConvFormer
from training_pipeline import *
from testing_pipeline import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                        help="See default.py for all options")
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    if cfg.BASIC.LOG_FLAG:
        save_log(cfg, args.cfg_file)

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
        model = ReTENet(num_classes=cfg.MODEL.NUM_CLASSES).to(cfg.BASIC.DEVICE)
    elif cfg.MODEL.NAME == 'BiLSTM':
        model = BiLSTM(out_channel=cfg.MODEL.NUM_CLASSES).to(cfg.BASIC.DEVICE)
    elif cfg.MODEL.NAME == 'DAGCN':
        model = DAGCN(num_classes=10).to(cfg.BASIC.DEVICE)
    elif cfg.MODEL.NAME == 'CNN':
        model = CNN(in_channels=50).to(cfg.BASIC.DEVICE)
    elif cfg.MODEL.NAME == 'ConvFormer':
        model = ConvFormer().to(cfg.BASIC.DEVICE)
    else:
        model = resnet18(in_channels=1, num_classes=10).to(cfg.BASIC.DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    cfg.PATH.LOG_PATH = str(pathlib.Path(cfg.PATH.LOG_PATH).joinpath(f'{get_time()}'))

    split_ratio = {'train': cfg.DATA.SPLIT_RATIO[0],
                   'eval': cfg.DATA.SPLIT_RATIO[1]}

    datasets, dataloaders = {}, {}
    data_domains = {'source': int(cfg.DATA.SOURCE), 'target': int(cfg.DATA.TARGET)}
    datasets.setdefault('train', TEPDataset(cfg.PATH.DATA_PATH, split_ratio, data_domains,
                                            'train', seed=cfg.BASIC.RANDOM_SEED,
                                            data_dim=cfg.DATA.DIM,
                                            time_win=cfg.DATA.TIME_WINDOW,
                                            overlap=cfg.DATA.OVERLAP))
    datasets.setdefault('val', TEPDataset(cfg.PATH.DATA_PATH, split_ratio, data_domains,
                                          'eval', seed=cfg.BASIC.RANDOM_SEED,
                                          data_dim=cfg.DATA.DIM,
                                          time_win=cfg.DATA.TIME_WINDOW,
                                          overlap=cfg.DATA.OVERLAP))
    datasets.setdefault('test', TEPDataset(cfg.PATH.DATA_PATH, split_ratio, data_domains,
                                           'test', seed=cfg.BASIC.RANDOM_SEED,
                                           data_dim=cfg.DATA.DIM,
                                           time_win=cfg.DATA.TIME_WINDOW,
                                           overlap=cfg.DATA.OVERLAP))

    dataloaders.setdefault('train', DataLoader(datasets['train'], batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True))
    dataloaders.setdefault('val', DataLoader(datasets['val'], batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True))
    dataloaders.setdefault('test', DataLoader(datasets['test'], batch_size=cfg.TESTING.BATCH_SIZE, shuffle=False))

    if cfg.TRAINING.PIPELINE == 'default':
        train_default(train_iter=dataloaders['train'],
                      eval_iter=dataloaders['val'],
                      model=model,
                      criterion=criterion,
                      args=cfg)

    if cfg.TRAINING.PIPELINE == 'arm':
        domains = [int(domain) for domain in cfg.DATA.SOURCES]
        train_with_arm(domains=domains,
                       model=model,
                       criterion=criterion,
                       args=cfg)

    model_name = f'best_model_{cfg.MODEL.CKPT_SUFFIX}.pth' if cfg.MODEL.CKPT_SUFFIX != '' else 'best_model.pth'

    for testing_pipeline in cfg.TESTING.PIPELINE:
        if testing_pipeline == 'default':
            test_default(test_iter=dataloaders['test'],
                         model_path=pathlib.Path(cfg.PATH.CKPT_PATH).joinpath(model_name),
                         args=cfg)

        if testing_pipeline == 'norm':
            test_with_adaptive_norm(test_iter=dataloaders['test'],
                                    model_path=pathlib.Path(cfg.PATH.CKPT_PATH).joinpath(model_name),
                                    args=cfg)

        if testing_pipeline == 'tent':
            test_with_tent(test_iter=dataloaders['test'],
                           model_path=pathlib.Path(cfg.PATH.CKPT_PATH).joinpath(model_name),
                           args=cfg)

        if testing_pipeline == 'arm':
            test_with_arm(test_iter=dataloaders['test'],
                          model_path=pathlib.Path(cfg.PATH.CKPT_PATH).joinpath(model_name),
                          args=cfg)
