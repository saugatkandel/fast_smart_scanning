import logging
from datetime import datetime
from pathlib import Path


def setup_logging(out_dir: str = 'LOGS',
                  out_prefix: str = 'smart_scan',
                  level='INFO'):
    formatter = logging.Formatter("%(asctime)s; %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    out_dir = Path(f'{out_dir}')
    out_dir.mkdir(exist_ok=True)
    dt = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    fh = logging.FileHandler(out_dir / f'{out_prefix}_{dt}.log')
    fh.setFormatter(formatter)
    logging.basicConfig(handlers=[ch, fh], level=level)
