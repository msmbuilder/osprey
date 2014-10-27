from __future__ import print_function, absolute_import, division

import os
import yaml
from .rcfile import USER_RC_PATH, CURDIR_RC_PATH


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

    print("\033[92mcreate\033[0m  {:s}".format(path))
    with open(path, 'w') as f:
        yaml.dump(options, f, default_flow_style=False)
