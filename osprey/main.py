from __future__ import print_function

import sys
import argparse

from . import __version__
from . import main_worker
from . import main_dump
from . import main_createrc

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        '-V', '--version',
        action='version',
        version='osprey %s' % __version__,
    )
    sub_parsers = p.add_subparsers(
        metavar='command',
        dest='cmd',
    )

    main_worker.configure_parser(sub_parsers)
    main_dump.configure_parser(sub_parsers)
    main_createrc.configure_parser(sub_parsers)

    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = p.parse_args()
    args_func(args, p)


def args_func(args, p):
    try:
        args.func(args, p)
    except RuntimeError as e:
        sys.exit("Error: %s" % e)
    except Exception as e:
        if e.__class__.__name__ not in ('ScannerError', 'ParserError'):
            message = """\
An unexpected error has occurred, please consider sending the
following traceback to the mixtape GitHub issue tracker at:

        https://github.com/rmcgibbo/osprey/issues

"""
            print(message, file=sys.stderr)
        raise  # as if we did not catch it
