import torch
import logging

from pathlib import Path
from datetime import datetime


def get_time(fmt='%Y%m%d_%H%M%S') -> str:
    time_stamp = datetime.now().strftime(fmt)
    return time_stamp


def save_log(configs, cfg_file) -> None:
    log_path = Path(configs.PATH.LOG_PATH)
    configs.BASIC.LOG_TIME = get_time()
    file_name = Path(cfg_file).name.replace('.yaml', 'core{}.txt'.format(configs.BASIC.LOG_TIME))
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(filename)s: %(lineno)4d] %(message)s',
                        datefmt="%y/%m/%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(log_path.joinpath(file_name)),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda]
    logger.info('\nversion: torch={}, cuda={}'.format(*version))
    logger.info('\n' + configs.__str__())
