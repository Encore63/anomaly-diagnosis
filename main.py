import torch
import pathlib
import argparse

from utils.logger import get_time
from models.mlp import MLP
from models.tenet import TENet, ReTENet
from datasets.tep_dataset import TEPDataset
from torch.utils.data.dataloader import DataLoader
from training_pipeline import train, train_with_learned_loss
from testing_pipeline import test, test_with_learned_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--data-dir', type=str, default=r'./data/TEP')
    parser.add_argument('--split-ratio', type=tuple, default=(0.7, 0.3))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda-id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--inner-epochs', type=int, default=1, help='num of inner loop iterations')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--train', action='store_true', help='train or not')
    parser.add_argument('--train-ll', action='store_true', help='choice of train with learned loss')
    parser.add_argument('--test-ll', action='store_true', help='choice of test with learned loss')
    parser.add_argument('--ttba', action='store_true', help='choice of TTBA')
    parser.add_argument('--step-size', type=int, default=10, help='step size of learning rate scheduler')
    parser.add_argument('--s', type=str, default='1', help='source domain')
    parser.add_argument('--t', type=str, default='3', help='target domain')
    parser.add_argument('--ckpt-suffix', type=str, default=None, help='suffix of saved checkpoint')
    parser.add_argument('--log-dir', type=str, default=r'./logs', help='save path of logs')
    parser.add_argument('--output-dir', type=str, default=r'./checkpoints', help='save path of model weights')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.cuda_id)

    if not pathlib.Path(args.data_dir).exists():
        pathlib.Path(args.data_dir).mkdir()
    if not pathlib.Path(args.log_dir).exists():
        pathlib.Path(args.log_dir).mkdir()
    if not pathlib.Path(args.output_dir).exists():
        pathlib.Path(args.output_dir).mkdir()

    model = ReTENet(f1=16, f2=32, depth=8, num_classes=args.num_classes).to(args.device)

    ll_model = MLP(in_features=args.num_classes, hidden_dim=32, out_features=1, norm_reduce=True).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    args.log_dir = str(pathlib.Path(args.log_dir).joinpath(f'{get_time()}'))

    args.split_ratio = {'train': args.split_ratio[0],
                        'eval': args.split_ratio[1]}

    check = False
    datasets, dataloaders = {}, {}
    if 1 <= int(args.s) <= 6 and 1 <= int(args.t) <= 6:
        check = True
        data_domains = {'source': int(args.s), 'target': int(args.t)}
        datasets.setdefault('train', TEPDataset(args.data_dir, args.split_ratio, data_domains,
                                                'train', seed=args.seed))
        datasets.setdefault('val', TEPDataset(args.data_dir, args.split_ratio, data_domains,
                                              'eval', seed=args.seed))
        datasets.setdefault('test', TEPDataset(args.data_dir, args.split_ratio, data_domains,
                                               'test', seed=args.seed))

        dataloaders.setdefault('train', DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True))
        dataloaders.setdefault('val', DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=True))
        dataloaders.setdefault('test', DataLoader(datasets['test'], shuffle=False))

    if check and args.train:
        train(train_iter=dataloaders['train'],
              eval_iter=dataloaders['val'],
              model=model,
              criterion=criterion,
              args=args)

    if args.train_ll:
        domains = [int(domain) for domain in args.s]
        train_with_learned_loss(domains=domains,
                                model=model,
                                ll_model=ll_model,
                                criterion=criterion,
                                args=args)

    if args.test_ll:
        test_with_learned_loss(test_iter=dataloaders['test'],
                               model_path=pathlib.Path(args.output_dir).joinpath(f'best_model_{args.ckpt_suffix}.pth'),
                               ll_model_path=pathlib.Path(args.output_dir).joinpath(f'learned_loss.pth'),
                               args=args)

    test(test_iter=dataloaders['test'],
         model_path=pathlib.Path(args.output_dir).joinpath(f'best_model_{args.ckpt_suffix}.pth'),
         args=args)
