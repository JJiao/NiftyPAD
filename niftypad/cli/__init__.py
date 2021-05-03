"""Usage:
  niftypad [<cmd>] [options]

Arguments:
  <cmd>  : Command to run

Options:
  -i PATH, --input PATH  : Input file/folder
  -o PATH, --output PATH  : Output file/folder
  --log LEVEL  : verbosity: ERROR|WARN(ING)|[default: INFO]|DEBUG
"""
import logging
import sys

from argopt import argopt

log = logging.getLogger(__name__)


def run(args):
    raise NotImplementedError


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = argopt(__doc__).parse_args(args=argv)
    logging.basicConfig(level=getattr(logging, args.log, logging.INFO))
    log.debug(args)
    return run(args) or 0


if __name__ == "__main__":
    main()
