from __future__ import print_function
import os
import yaml

__all__ = ['load_rcfile', 'CURDIR_RC_PATH', 'USER_RC_PATH']


CURDIR_RC_PATH = os.path.abspath('.ospreyrc')
USER_RC_PATH = os.path.abspath(os.path.expanduser('~/.ospreyrc'))


def load_rcfile(verbose=True):

    def get_rc_path():
        path = os.getenv('OSPREYRC')
        if path == ' ':
            return None
        if path:
            return path
        for path in (CURDIR_RC_PATH, USER_RC_PATH):
            if os.path.isfile(path):
                return path
        return None

    path = get_rc_path()
    if not path:
        return dict()
    if verbose:
        print('Loading .ospreyrc from %s...' % path)

    with open(path) as f:
        return yaml.load(f) or dict()
