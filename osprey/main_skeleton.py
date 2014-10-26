from __future__ import print_function, absolute_import, division

import shutil
from os.path import join, exists
from pkg_resources import resource_filename
from argparse import ArgumentDefaultsHelpFormatter


def configure_parser(sub_parsers):
    help = 'create skeleton config.yaml file'
    p = sub_parsers.add_parser('skeleton', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-t', '--template', help=(
        "which skeleton to create. 'mixtape' is a skeleton config file for"
        "mixtape-based molecular dynamics / Markov state model based "
        "projects."), choices=['mixtape'], default='mixtape',)
    p.add_argument('-f', '--filename', help='config filename to create',
                   default='config.yaml')
    p.set_defaults(func=execute)


def execute(args, parser):
    if args.template == 'mixtape':
        fn = resource_filename('osprey', join('data',
                               'mixape_skeleton_config.yaml'))
    else:
        raise RuntimeError('unknown template: %s' % args.template)

    if exists(args.filename):
        raise RuntimeError('file already exists: %s' % args.filename)

    print("\033[92mcreate\033[0m  {:s}".format(args.filename))
    shutil.copy(fn, args.filename)
