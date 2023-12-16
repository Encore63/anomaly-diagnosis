import torch
import pathlib
import numpy as np


class EarlyStopping(object):
    """
    Stop the training if validation loss does not improve after a given patience
    """
    def __init__(self, save_path: str, patience: int = 7, verbose: bool = False, delta: int = 0, **kwargs):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.source = kwargs['s']
        self.target = kwargs['t']

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation Loss Decreased ({self.val_loss_min: .6f} -> {val_loss: .6f})')
        path = pathlib.Path(self.save_path).joinpath(f'best_model_{self.source}_{self.target}.pth')
        torch.save(model, path)
        self.val_loss_min = val_loss
