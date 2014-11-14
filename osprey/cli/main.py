from __future__ import print_function, absolute_import, division

import sys
import argparse

from .. import __version__
from . import parser_dump
from . import parser_skeleton
from . import parser_worker
from . import parser_plot


def main():
    help = 'osprey is a tool for machine learning hyperparameter optimization.'
    p = argparse.ArgumentParser(description=help)
    p.add_argument(
        '-V', '--version',
        action='version',
        version='osprey %s' % __version__,
    )
    sub_parsers = p.add_subparsers(
        metavar='command',
        dest='cmd',
    )

    parser_dump.configure_parser(sub_parsers)
    parser_skeleton.configure_parser(sub_parsers)
    parser_worker.configure_parser(sub_parsers)
    parser_plot.configure_parser(sub_parsers)

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
An unexpected error has occurred with osprey (version %s), please
consider sending the following traceback to the osprey GitHub issue tracker at:

        https://github.com/pandegroup/osprey/issues
"""
            print(message % __version__, file=sys.stderr)
        raise  # as if we did not catch it
