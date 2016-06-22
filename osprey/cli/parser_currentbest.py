from __future__ import print_function, absolute_import, division
from argparse import ArgumentDefaultsHelpFormatter

__author__ = 'muneeb'


def func(args, parser):
    # delay import of the rest of the module to improve `osprey -h` performance
    from ..execute_currentbest import execute
    execute(args, parser)


def configure_parser(sub_parsers):
    help = 'Get parameters for the current best model'
    p = sub_parsers.add_parser('current_best', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('config', help='Path to worker config file (yaml)')

    p.set_defaults(func=func)
