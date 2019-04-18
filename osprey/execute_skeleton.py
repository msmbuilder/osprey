from __future__ import print_function, absolute_import, division

from os.path import join, exists
from pkg_resources import resource_filename

TEMPLATES = {'msmbuilder': 'msmbuilder_skeleton_config.yaml',
             'sklearn': 'sklearn_skeleton_config.yaml',
             'random_example': 'random_example.yaml',
             'bayes_example': 'sklearn_skeleton_config.yaml',
             'grid_example': 'grid_example.yaml',
             'msmb_feat_select': 'msmb_feat_select_skeleton_config.yaml'}


def execute(args, parser):
    template_file = TEMPLATES.get(args.template, None)
    if template_file:
        fn = resource_filename('osprey', join('data', template_file))
    else:
        raise RuntimeError('unknown template: %s' % args.template)

    if exists(args.filename):
        raise RuntimeError('file already exists: %s' % args.filename)

    print("\033[92mcreate\033[0m  {0:s}".format(args.filename))

    with open(args.filename, 'wb') as out:
        with open(fn, 'rb') as inp:
            out.write(inp.read())
