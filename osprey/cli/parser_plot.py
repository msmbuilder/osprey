from __future__ import print_function, absolute_import, division
from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    # delay import of the rest of the module to improve `osprey -h` performance
    from ..execute_plot import execute
    execute(args, parser)


def configure_parser(sub_parsers):
    help = 'Visualize the collection of jobs in the trials database'
    p = sub_parsers.add_parser('plot', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('config', help='Path to worker config file (yaml)')
    p.add_argument('--filename', default='plot.html')
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument('--browser', action='store_true', default=True,
                   help='launch browser')
    g.add_argument('--no-browser', action='store_false', dest='browser')

    p.set_defaults(func=func)
