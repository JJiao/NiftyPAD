from pathlib import Path


def read_json(fd):
    from json import load
    return load(fd)


def read_yaml(fd):
    from oyaml import safe_load
    return safe_load(fd)


READERS = {'yml': read_yaml, 'yaml': read_yaml, 'json': read_json}


def read_meta(fpath):
    fpath = Path(fpath)
    return READERS[fpath.suffix[1:].lower()](fpath.open())
