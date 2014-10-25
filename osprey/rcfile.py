import os
import sys
import yaml

__all__ = ['RC']


CURDIR_RC_PATH = os.path.abspath('.ospreyrc')
USER_RC_PATH = os.path.abspath(os.path.expanduser('~/.ospreyrc'))
SYS_RC_PATH = os.path.join(sys.prefix, '.ospreyrc')


def get_rc_path():
    path = os.getenv('OSPREYRC')
    if path == ' ':
        return None
    if path:
        return path
    for path in (CURDIR_RC_PATH, USER_RC_PATH, SYS_RC_PATH):
        if os.path.isfile(path):
            return path
    return None


def load_rcfile(path):
    if not path:
        return {}
    return yaml.load(open(path)) or {}


RC = load_rcfile(get_rc_path())
