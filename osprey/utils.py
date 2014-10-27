from __future__ import print_function, absolute_import, division
import os.path
import contextlib

__all__ = ['dict_merge']


def dict_merge(base, top):
    """Recursively merge two dictionaries, with the elements from `top`
    taking precedence over elements from `top`.

    Returns
    -------
    out : dict
        A new dict, containing the merged records.
    """
    out = dict(top)
    for key in base:
        if key in top:
            if isinstance(base[key], dict) and isinstance(top[key], dict):
                out[key] = dict_merge(base[key], top[key])
        else:
            out[key] = base[key]
    return out


@contextlib.contextmanager
def in_directory(path):
    """Context manager (with statement) that changes the current directory
    during the context.
    """
    curdir = os.path.abspath(os.curdir)
    os.chdir(path)
    yield
    os.chdir(curdir)


class Unbuffered(object):
    # used to turn off output buffering
    # http://stackoverflow.com/questions/107705/python-output-buffering

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def mock_module(name):

    class MockModule(object):
        def __cal__(self, *args, **kwargs):
            raise ImportError('no module named %s' % name)

        def __getattr__(self, *args, **kwargs):
            raise ImportError('no module named %s' % name)

    return MockModule()
