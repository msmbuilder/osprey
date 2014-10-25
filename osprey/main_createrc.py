from __future__ import print_function, absolute_import, division

import os
import yaml
from argparse import ArgumentDefaultsHelpFormatter

from .rcfile import USER_RC_PATH, CURDIR_RC_PATH


def configure_parser(sub_parsers):
    help = 'Initialize .ospreyrc file'
    p = sub_parsers.add_parser('createrc', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-l', '--loc', help='Location for .ospreyrc file',
                   choices=['user', 'curdir'], default='user')
    p.add_argument('-t', '--template', help="Template to use",
                   choices=['mixtape'], default='mixtape')
    p.set_defaults(func=execute)


def execute(args, parser):
    if args.template == 'mixtape':
        options = {
            'estimator': {'__eval_globals__':
                          'osprey.plugins.mixtape.eval_globals'},
            'dataset': {'__loader__':
                        'osprey.plugins.mixtape.trajectory_dataset'},
        }
    else:
        raise RuntimeError('unknown template: %s' % args.template)

    path_map = {'user': USER_RC_PATH, 'curdir': CURDIR_RC_PATH}
    path = path_map[args.loc]

    if os.path.exists(path):
        raise RuntimeError('%s already exists' % path)

    print('saving file %s' % path)
    with open(path, 'w') as f:
        yaml.dump(options, f, default_flow_style=False)
