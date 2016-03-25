from __future__ import print_function, absolute_import, division

from os.path import join, exists
from pkg_resources import resource_filename


def execute(args, parser):
    if args.template == 'msmbuilder':
        fn = resource_filename('osprey', join('data',
                               'msmbuilder_skeleton_config.yaml'))
    elif args.template == 'sklearn':
        fn = resource_filename('osprey', join('data',
                               'sklearn_skeleton_config.yaml'))
    else:
        raise RuntimeError('unknown template: %s' % args.template)

    if exists(args.filename):
        raise RuntimeError('file already exists: %s' % args.filename)

    print("\033[92mcreate\033[0m  {0:s}".format(args.filename))

    with open(args.filename, 'wb') as out:
        with open(fn, 'rb') as inp:
            out.write(inp.read())
