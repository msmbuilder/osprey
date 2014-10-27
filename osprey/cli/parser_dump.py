from __future__ import print_function, absolute_import, division
from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    # delay import of the rest of the module to improve `osprey -h` performance
    from ..execute_dump import execute
    execute(args, parser)


def configure_parser(sub_parsers):
    help = 'Dump history SQL database to CSV or JSON'
    p = sub_parsers.add_parser('dump', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('config', help='Path to worker config file (yaml)')
    p.add_argument('-o', '--output', choices=['csv', 'json'], default='json',
                   help='output format')
    p.set_defaults(func=func)
