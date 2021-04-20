from os import getenv
from pathlib import Path

from pytest import fixture, skip

HOME = Path(getenv("DATA_ROOT", "~")).expanduser()


@fixture(scope="session")
def folder_in():
    AMYPAD = HOME / "AMYPAD"
    if not AMYPAD.is_dir():
        skip(f"Cannot find AMYPAD in ${{DATA_ROOT:-~}} ({HOME})")
    return AMYPAD
