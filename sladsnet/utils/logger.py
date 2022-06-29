import logging
from datetime import datetime
from pathlib import Path


def setup_logging(out_dir: str = 'LOGS',
                 out_prefix:str ='smart_scan',
                 level='INFO'):
    formatter = logging.Formatter("%(asctime)s;%(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    dt = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    fh = logging.FileHandler(Path(f'{out_dir}/{out_prefix}_{dt}'))
    fh.setFormatter(formatter)
    logging.basicConfig(handlers=[ch, fh], format=formatter, level=level)