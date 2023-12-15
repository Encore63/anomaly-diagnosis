import torch
import pathlib
import argparse

from models.tenet import TENet
from utils.logger import get_time
from datasets.tep_dataset import TEPDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from training_pipeline import train
from testing_pipeline import test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--data-dir', type=str, default=r'./data/TEP')
    parser.add_argument('--split-ratio', type=tuple, default=(0.7, 0.3))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cuda-id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--step-size', type=int, default=5, help='step size of learning rate scheduler')
    parser.add_argument('--s', type=int, default=1, help='source domain', choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--t', type=int, default=3, help='target domain', choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--log-dir', type=str, default=r'./logs', help='save path of logs')
    parser.add_argument('--output-dir', type=str, default=r'./checkpoints', help='save path of model')
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

    model = TENet(f1=32, f2=64, depth=8, num_classes=args.num_classes).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    args.log_dir = str(pathlib.Path(args.log_dir).joinpath(get_time()))
    log_writer = SummaryWriter(log_dir=args.log_dir)

    args.split_ratio = {'train': args.split_ratio[0],
                        'eval': args.split_ratio[1]}
    data_domains = {'source': args.s, 'target': args.t}
    train_dataset = TEPDataset(args.data_dir, args.split_ratio, data_domains, 'train')
    eval_dataset = TEPDataset(args.data_dir, args.split_ratio, data_domains, 'eval')
    test_dataset = TEPDataset(args.data_dir, args.split_ratio, data_domains, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train(train_iter=train_dataloader,
          eval_iter=eval_dataloader,
          model=model,
          criterion=criterion,
          writer=log_writer,
          save_path=pathlib.Path(args.output_dir),
          args=args)

    test(test_iter=test_dataloader,
         model_path=pathlib.Path(args.output_dir).joinpath('best_model.pth'),
         criterion=criterion)

    log_writer.close()
