"""Metadata reader utilities"""
import errno
from pathlib import Path

CONFIG_NAMES = ['meta', 'conf', 'config', 'params', 'param', 'input']


def read_json(fd):
    from json import load
    return load(fd)


def read_yaml(fd):
    from oyaml import safe_load
    return safe_load(fd)


READERS = {'yml': read_yaml, 'yaml': read_yaml, 'json': read_json}


def read_meta(fpath):
    """Auto-detect `fpath`'s type and return a contents `dict`.

    Args:
      fpath (Path or str): filename
    Returns:
      dict: contents
    """
    fpath = Path(fpath)
    return READERS[fpath.suffix[1:].lower()](fpath.open())


def find_meta(src_path, config_name_hints=None):
    """Find and return metadata file contents

    Args:
      src_path (Path or str): where to look
      config_name_hints (list): stems to search for in `src_path`
    Returns:
      dict: contents
    """
    src_path = Path(src_path)
    if config_name_hints is None:
        config_names = CONFIG_NAMES
    else:
        config_names = list(config_name_hints) + CONFIG_NAMES
    for ext in sorted([''] + ['.' + i for i in READERS]):
        for meta in config_names:
            if (src_path / f'{meta}{ext}').is_file():
                return read_meta(src_path / f'{meta}{ext}')
    raise IOError(errno.ENOENT, "Input metadata not found")
