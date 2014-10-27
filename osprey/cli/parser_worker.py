from __future__ import print_function, absolute_import, division
from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    # delay import of the rest of the module to improve `osprey -h` performance
    from ..execute_worker import execute
    execute(args, parser)


def configure_parser(sub_parsers):
    help = 'Run a worker process (hyperparameter optimization)'
    p = sub_parsers.add_parser('worker', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('config', help='Path to worker config file (yaml)')
    p.add_argument('-n', '--n-iters', default=1, type=int, help='Number of '
                   'trials to run sequentially.')
    p.set_defaults(func=func)
