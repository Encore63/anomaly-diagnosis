"""Configuration file (powered by YACS)."""
from yacs.config import CfgNode

_C = CfgNode()
cfg = _C

"""
    BASIC OPTION
"""
_C.BASIC = CfgNode()
_C.BASIC.RANDOM_SEED = 2024
_C.BASIC.DEVICE = 'cuda'  # DEVICE CHOICE ('cpu', 'cuda')
_C.BASIC.CUDA_ID = 0  # CUDA-ID CHOICE (0, 1, 2, 3)
_C.BASIC.LOG_TIME = ''
_C.BASIC.LOG_FLAG = True

"""
    MODEL OPTION
"""
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'ResNet'  # MODEL CHOICE ('TENet', 'ReTENet', 'ResNet')

_C.MODEL.ARM = 'ARM_LL'  # ARM ALGORITHM CHOICE ('ERM', 'ARM_BN', 'ARM_LL', 'ARM_CML')
_C.MODEL.CONTEXT_CHANNELS = 1
_C.MODEL.ADAPT_BN = False

_C.MODEL.CKPT_SUFFIX = ''
_C.MODEL.NUM_CLASSES = 10

"""
    DATA OPTION
"""
_C.DATA = CfgNode()
_C.DATA.SOURCE = 1  # SOURCE CHOICE (1, 2, 3, 4, 5, 6)
_C.DATA.TARGET = 1  # TARGET CHOICE (1, 2, 3, 4, 5, 6)
_C.DATA.SOURCES = []
_C.DATA.TARGETS = []
_C.DATA.SPLIT_RATIO = [0.7, 0.3]
_C.DATA.TIME_WINDOW = 10

"""
    OPTIMIZER OPTION
"""
_C.OPTIM = CfgNode()
_C.OPTIM.LEARNING_RATE = 1e-3
_C.OPTIM.STEP_SIZE = 30
_C.OPTIM.WEIGHT_DECAY = 0

"""
    TRAINING OPTION
"""
_C.TRAINING = CfgNode()
_C.TRAINING.EPOCHS = 50
_C.TRAINING.BATCH_SIZE = 128
_C.TRAINING.PIPELINE = 'default'  # TRAINING PIPELINE CHOICE ('default', 'arm')

#  PARAM OF EARLY STOPPING TOOL
_C.TRAINING.PATIENCE = 7
_C.TRAINING.DELTA = 0

"""
    TESTING OPTION
"""
_C.TESTING = CfgNode()
_C.TESTING.BATCH_SIZE = 1
_C.TESTING.PIPELINE = ['default']  # TESTING PIPELINE CHOICE ('default', 'arm_ll', 'tent')

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
