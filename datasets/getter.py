from datasets.tep_dataset import TEPDataset
from datasets.cwru_dataset import CWRUDataset


def get_dataset(dataset_name, **kwargs):
    if dataset_name == 'tep':
        task = kwargs['transfer_task']
        domains = {'source': task[0], 'target': task[1]}
        return TEPDataset(src_path=r'../../data/TEP',
                          split_ratio={'train': 0.7, 'eval': 0.2},
                          data_domains=domains,
                          dataset_mode=kwargs['dataset_mode'],
                          time_win=kwargs['time_win'],
                          data_dim=kwargs['data_dim'])
    elif dataset_name == 'cwru':
        dataset_tool = CWRUDataset(data_dir=r'../../data/CWRU',
                                   transfer_task=kwargs['transfer_task'],
                                   data_dim=kwargs['data_dim'])
        _datasets = dataset_tool.data_split(transfer_learning=kwargs['transfer_learning'])
        source_train, source_val, target_train, target_val = _datasets
        if kwargs['dataset_mode'] == 'train':
            return source_train
        elif kwargs['dataset_mode'] == 'eval':
            return source_val
        elif kwargs['dataset_mode'] == 'test':
            return target_train, target_val
    else:
        raise ValueError("Unknown dataset")
