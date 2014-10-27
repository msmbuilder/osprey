from __future__ import print_function, absolute_import, division

import shutil
from os.path import join, exists
from pkg_resources import resource_filename


def execute(args, parser):
    if args.template == 'mixtape':
        fn = resource_filename('osprey', join('data',
                               'mixape_skeleton_config.yaml'))
    else:
        raise RuntimeError('unknown template: %s' % args.template)

    if exists(args.filename):
        raise RuntimeError('file already exists: %s' % args.filename)

    print("\033[92mcreate\033[0m  {0:s}".format(args.filename))
    shutil.copy(fn, args.filename)
