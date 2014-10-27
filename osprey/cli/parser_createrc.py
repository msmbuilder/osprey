from __future__ import print_function, absolute_import, division
from argparse import ArgumentDefaultsHelpFormatter


def func(args, parser):
    # delay import of the rest of the module to improve `osprey -h` performance
    from ..execute_createrc import execute
    execute(args, parser)


def configure_parser(sub_parsers):
    help = 'create .ospreyrc file'
    description = '''The .ospreyrc file is read and used together with the
config.yaml file to initialize the osprey worker. Options from the config.yaml
file will always override the .ospreyrc file.
'''
    p = sub_parsers.add_parser('createrc', description=description, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-l', '--loc', help='Location for .ospreyrc file',
                   choices=['user', 'curdir'], default='user')
    p.add_argument('-t', '--template', help="Template to use",
                   choices=['mixtape'], default='mixtape')
    p.set_defaults(func=func)
