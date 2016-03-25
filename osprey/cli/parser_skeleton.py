from __future__ import print_function, absolute_import, division
from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    # delay import of the rest of the module to improve `osprey -h` performance
    from ..execute_skeleton import execute
    execute(args, parser)


def configure_parser(sub_parsers):
    help = 'create skeleton config.yaml file'
    p = sub_parsers.add_parser('skeleton', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-t', '--template', help=(
        "which skeleton to create. 'msmbuilder' is a skeleton config file for"
        "MSMBuilder molecular dynamics / Markov state model based "
        "projects."), choices=['msmbuilder', 'sklearn'], default='msmbuilder',)
    p.add_argument('-f', '--filename', help='config filename to create',
                   default='config.yaml')
    p.set_defaults(func=func)
