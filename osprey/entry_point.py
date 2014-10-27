from __future__ import print_function, absolute_import, division

import sys
import traceback
import importlib


def load_entry_point(entry_point_str, section=''):
    try:
        package_str, obj_str = entry_point_str.rsplit('.', 1)
    except ValueError:
        raise RuntimeError('%s syntax error' % section)

    try:
        package = importlib.import_module(package_str)
    except ImportError:
        print('-'*78)
        print('Error parsing %s' % section, file=sys.stderr)
        print('-'*78)
        traceback.print_exc(sys.stderr)
        sys.exit(1)

    try:
        value = getattr(package, obj_str)
    except (AttributeError, KeyError):
        raise RuntimeError('%s: %r does not contain object %r' % (
            section, package_str, obj_str))

    return value
