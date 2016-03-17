from __future__ import print_function, absolute_import, division
import warnings
import numpy as np
import scipy.sparse as sp
import os.path
import sys
import contextlib
from datetime import datetime
from sklearn.pipeline import Pipeline

from .eval_scopes import import_all_estimators


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


def is_msmbuilder_estimator(estimator):
    try:
        import msmbuilder
    except ImportError:
        return False
    msmbuilder_estimators = import_all_estimators(msmbuilder).values()

    out = estimator.__class__ in msmbuilder_estimators
    if isinstance(estimator, Pipeline):
        out = any(step.__class__ in msmbuilder_estimators
                  for name, step in estimator.steps)
    return out


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method
    if (X.dtype.char in np.typecodes['AllFloat'] and
            not np.isfinite(X.sum()) and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def _warn_if_not_finite(X):
    """UserWarning if array contains non-finite elements"""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method
    if (X.dtype.char in np.typecodes['AllFloat'] and
            not np.isfinite(X.sum()) and not np.isfinite(X).all()):
        warnings.warn("Result contains NaN, infinity"
                      " or a value too large for %r." % X.dtype,
                      category=UserWarning)


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit'):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_arrays(*arrays, **options):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.
    By default lists and tuples are converted to numpy arrays.

    It is possible to enforce certain properties, such as dtype, continguity
    and sparse matrix format (if a sparse matrix is passed).

    Converting lists to arrays can be disabled by setting ``allow_lists=True``.
    Lists can then contain arbitrary objects and are not checked for dtype,
    finiteness or anything else but length. Arrays are still checked
    and possibly converted.

    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays, unless allow_lists is specified.
    sparse_format : 'csr', 'csc' or 'dense', None by default
        If not None, any scipy.sparse matrix is converted to
        Compressed Sparse Rows or Compressed Sparse Columns representations.
        If 'dense', an error is raised when a sparse array is
        passed.
    copy : boolean, False by default
        If copy is True, ensure that returned arrays are copies of the original
        (if not already converted to another format earlier in the process).
    check_ccontiguous : boolean, False by default
        Check that the arrays are C contiguous
    dtype : a numpy dtype instance, None by default
        Enforce a specific dtype.
    warn_nans : boolean, False by default
        Prints warning if nans in the arrays
        Disables allow_nans
    replace_nans : boolean, False by default
        Replace nans in the arrays with zeros
    allow_lists : bool
        Allow lists of arbitrary objects as input, just check their length.
        Disables
    allow_nans : boolean, False by default
        Allows nans in the arrays
    allow_nd : boolean, False by default
        Allows arrays of more than 2 dimensions.
    """
    sparse_format = options.pop('sparse_format', None)
    if sparse_format not in (None, 'csr', 'csc', 'dense'):
        raise ValueError('Unexpected sparse format: %r' % sparse_format)
    copy = options.pop('copy', False)
    check_ccontiguous = options.pop('check_ccontiguous', False)
    dtype = options.pop('dtype', None)
    warn_nans = options.pop('warn_nans', False)
    replace_nans = options.pop('replace_nans', False)
    allow_lists = options.pop('allow_lists', False)
    allow_nans = options.pop('allow_nans', False)
    allow_nd = options.pop('allow_nd', False)

    if options:
        raise TypeError("Unexpected keyword arguments: %r" % options.keys())

    if len(arrays) == 0:
        return None

    n_samples = _num_samples(arrays[0])

    checked_arrays = []
    for array in arrays:
        array_orig = array
        if array is None:
            # special case: ignore optional y=None kwarg pattern
            checked_arrays.append(array)
            continue
        size = _num_samples(array)

        if size != n_samples:
            raise ValueError("Found array with dim %d. Expected %d"
                             % (size, n_samples))

        if not allow_lists or hasattr(array, "shape"):
            if sp.issparse(array):
                if sparse_format == 'csr':
                    array = array.tocsr()
                elif sparse_format == 'csc':
                    array = array.tocsc()
                elif sparse_format == 'dense':
                    raise TypeError('A sparse matrix was passed, but dense '
                                    'data is required. Use X.toarray() to '
                                    'convert to a dense numpy array.')
                if check_ccontiguous:
                    array.data = np.ascontiguousarray(array.data, dtype=dtype)
                elif hasattr(array, 'data'):
                    array.data = np.asarray(array.data, dtype=dtype)
                elif array.dtype != dtype:
                    array = array.astype(dtype)
                if not allow_nans:
                    if hasattr(array, 'data'):
                        _assert_all_finite(array.data)
                    else:
                        _assert_all_finite(array.values())
            else:
                if check_ccontiguous:
                    array = np.ascontiguousarray(array, dtype=dtype)
                else:
                    array = np.asarray(array, dtype=dtype)
                if warn_nans:
                    allow_nans = True
                    _warn_if_not_finite(array)
                if replace_nans:
                    array = np.nan_to_num(array)
                if not allow_nans:
                    _assert_all_finite(array)

            if not allow_nd and array.ndim >= 3:
                raise ValueError("Found array with dim %d. Expected <= 2" %
                                 array.ndim)

        if copy and array is array_orig:
            array = array.copy()
        checked_arrays.append(array)

    return checked_arrays
