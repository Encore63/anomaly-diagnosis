import torch
import pathlib
import argparse

from models.tenet import TENet
from utils.logger import get_time
from datasets.tep_dataset import TEPDataset
from torch.utils.data.dataloader import DataLoader
from training_pipeline import train
from testing_pipeline import test, adaptive_test


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
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--train', action='store_true', help='Train or not.')
    parser.add_argument('--step-size', type=int, default=10, help='step size of learning rate scheduler')
    parser.add_argument('--s', type=int, default=1, help='source domain', choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--t', type=int, default=3, help='target domain', choices=[1, 2, 3, 4, 5, 6])
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

    model = TENet(f1=16, f2=32, depth=8, num_classes=args.num_classes).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    args.log_dir = str(pathlib.Path(args.log_dir).joinpath(f'{get_time()}_{args.s}_{args.t}'))

    args.split_ratio = {'train': args.split_ratio[0],
                        'eval': args.split_ratio[1]}
    data_domains = {'source': args.s, 'target': args.t}
    train_dataset = TEPDataset(args.data_dir, args.split_ratio, data_domains, 'train', seed=args.seed)
    eval_dataset = TEPDataset(args.data_dir, args.split_ratio, data_domains, 'eval', seed=args.seed)
    test_dataset = TEPDataset(args.data_dir, args.split_ratio, data_domains, 'test', seed=args.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    if args.train:
        train(train_iter=train_dataloader,
              eval_iter=eval_dataloader,
              model=model,
              criterion=criterion,
              args=args)

    test(test_iter=test_dataloader,
         model_path=pathlib.Path(args.output_dir).joinpath(f'best_model_{args.s}_{args.t}.pth'),
         args=args)

    adaptive_test(test_dataset=test_dataset,
                  model_path=pathlib.Path(args.output_dir).joinpath(f'best_model_{args.s}_{args.t}.pth'),
                  criterion=criterion,
                  args=args)
