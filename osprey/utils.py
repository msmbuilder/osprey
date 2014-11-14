from __future__ import print_function, absolute_import, division
import os.path
import sys
import contextlib
from datetime import datetime

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


@contextlib.contextmanager
def prepend_syspath(path):
    """Contect manager (with statement) that prepends path to sys.path"""
    sys.path.insert(0, path)
    yield
    sys.path.pop(0)


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


def format_timedelta(td_object):
    """Format a timedelta object for display to users

    Returns
    -------
    str
    """
    def get_total_seconds(td):
        # timedelta.total_seconds not in py2.6
        return (td.microseconds +
                (td.seconds + td.days * 24 * 3600) * 1e6) / 1e6

    seconds = int(get_total_seconds(td_object))
    periods = [('year',    60*60*24*365),
               ('month',   60*60*24*30),
               ('day',     60*60*24),
               ('hour',    60*60),
               ('minute',  60),
               ('second',  1)]

    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            if period_value == 1:
                strings.append("%s %s" % (period_value, period_name))
            else:
                strings.append("%s %ss" % (period_value, period_name))

    return ", ".join(strings)


def current_pretty_time():
    return datetime.now().strftime("%B %d, %Y %l:%M %p")


def _squeeze_time(t):
    """Remove .1s to the time under Windows: this is the time it take to
    stat files. This is needed to make results similar to timings under
    Unix, for tests
    """
    if sys.platform.startswith('win'):
        return max(0, t - .1)
    else:
        return t


def short_format_time(t):
    t = _squeeze_time(t)
    if t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.1fs" % (t)


def mock_module(name):

    class MockModule(object):
        def __cal__(self, *args, **kwargs):
            raise ImportError('no module named %s' % name)

        def __getattr__(self, *args, **kwargs):
            raise ImportError('no module named %s' % name)

    return MockModule()


def join_quoted(values, quote="'"):
    return ', '.join("%s%s%s" % (quote, e, quote) for e in values)


def expand_path(path, base='.'):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return path
