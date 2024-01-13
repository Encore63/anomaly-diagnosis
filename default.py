"""Configuration file (powered by YACS)."""
from yacs.config import CfgNode

_C = CfgNode()
cfg = _C

"""
    Basic Hyper-parameters
"""
_C.BASIC = CfgNode()
_C.BASIC.RANDOM_SEED = 2024
_C.BASIC.DEVICE = 'cuda'  # DEVICE CHOICE ('cpu', 'cuda')
_C.BASIC.CUDA_ID = 0  # CUDA-ID CHOICE (0, 1, 2, 3)
_C.BASIC.SOURCE = 1  # SOURCE CHOICE (1, 2, 3, 4, 5, 6)
_C.BASIC.TARGET = 2  # SOURCE CHOICE (1, 2, 3, 4, 5, 6)
_C.BASIC.SPLIT_RATIO = [0.7, 0.3]
_C.BASIC.LOG_TIME = ''
_C.BASIC.LOG_FLAG = True


"""
    MODEL OPTION
"""
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'ReTENet'   # MODEL CHOICE ('TENet', 'ReTENet')
_C.MODEL.CKPT_SUFFIX = ''
_C.MODEL.NUM_CLASSES = 10

"""
    TRAINING OPTION
"""
_C.TRAINING = CfgNode()
_C.TRAINING.EPOCHS = 50
_C.TRAINING.LEARNING_RATE = 1e-3  # PARAM OF OPTIMIZER
_C.TRAINING.STEP_SIZE = 10  # PARAM OF LEARNING RATE SCHEDULER
_C.TRAINING.BATCH_SIZE = 128
_C.TRAINING.PIPELINE = 'default'  # TRAINING PIPELINE CHOICE ('default', 'arm_ll')

"""
    TESTING OPTION
"""
_C.TESTING = CfgNode()
_C.TESTING.BATCH_SIZE = 1
_C.TESTING.PIPELINE = 'default'  # TESTING PIPELINE CHOICE ('default', 'arm_ll', 'tent')

"""
    PATH OPTION
"""
_C.PATH = CfgNode()
_C.PATH.DATA_PATH = r'./data/TEP'
_C.PATH.CKPT_PATH = r'./checkpoints'
_C.PATH.LOG_PATH = r'./logs'
_C.PATH.CONFIG_PATH = r'./configs'

"""
    SET DEFAULT CONFIG
"""
_DEFAULT_CFG = _C.clone()
_DEFAULT_CFG.freeze()


def merge_from_file(cfg_file_path):
    with open(cfg_file_path, 'r') as f:
        new_cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(new_cfg)


if __name__ == '__main__':
    merge_from_file(r'./configs/norm.yaml')
    print(cfg)
