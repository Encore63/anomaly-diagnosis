from datasets.tep_dataset import TEPDataset
from datasets.cwru_dataset import CWRUDataset


def get_dataset(dataset_name, transfer_task, dataset_mode, **kwargs):
    import os

    if dataset_name == 'tep':
        # default params setting
        time_win = 10 if 'time_win' not in kwargs.keys() else kwargs['time_win']
        data_dim = 3 if 'data_dim' not in kwargs.keys() else kwargs['data_dim']

        dataset_tool = TEPDataset(src_path=rf'{os.getcwd()}\data\{dataset_name.upper()}',
                                  transfer_task=transfer_task,
                                  time_win=time_win,
                                  data_dim=data_dim)
        _datasets = dataset_tool.get_subset()
        if dataset_mode == 'train':
            return _datasets['train']
        elif dataset_mode == 'val':
            return _datasets['val']
        elif dataset_mode == 'test':
            return _datasets['test']
    elif dataset_name == 'cwru':
        # default params setting
        data_dim = 3 if 'data_dim' not in kwargs.keys() else kwargs['data_dim']
        transfer_learning = True if 'transfer_learning' not in kwargs.keys() else kwargs['transfer_learning']

        dataset_tool = CWRUDataset(data_dir=rf'{os.getcwd()}\data\{dataset_name.upper()}',
                                   transfer_task=transfer_task,
                                   data_dim=data_dim)
        _datasets = dataset_tool.data_split(transfer_learning=transfer_learning)
        source_train, source_val, target_train, target_val = _datasets
        if dataset_mode == 'train':
            return source_train
        elif dataset_mode == 'val':
            return source_val
        elif dataset_mode == 'test':
            return target_train, target_val
    else:
        raise ValueError("Unknown dataset")
