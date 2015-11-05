from __future__ import print_function, absolute_import, division

import inspect
import itertools
import collections
from six import string_types
from .utils import join_quoted


def init_subclass_by_name(baseclass, short_name, params):
    """
    Find the subclass, `kls` of baseclass with class attribute `short_name`
    that matches the supplied `short_name`, and then instantiate and return
    that class with:

        return kls(**params)

    This function also tries its best to catch any possible TypeErrors due
    to binding of the arguments, and rethrows them as nicely formatted
    RuntimeErrors that are suitable for showing to users.
    """
    sc = baseclass.__subclasses__()
    for kls in sc:
        if kls.short_name == short_name or \
           (_is_collection(kls.short_name) and short_name in kls.short_name):
            try:
                return kls(**params)
            except TypeError as e:
                spec = inspect.getargspec(kls.__init__)

                # try to give nice errors to the user if the params failed
                # to bind
                if 'unexpected' in str(e):
                    avail = join_quoted(spec.args[1:])
                    raise RuntimeError(
                        "%s's %s. Available params for this subclass are: %s."
                        % (short_name, str(e), avail))
                elif 'takes exactly' in str(e):
                    required = join_quoted(spec.args[1:-len(spec.defaults)])
                    raise RuntimeError(
                        "%s's %s. Required params for this subclass are %s."
                        % (short_name, str(e), required))
                elif 'takes at least' in str(e):
                    required = join_quoted(spec.args[1:-len(spec.defaults)])
                    optional = join_quoted(spec.args[-len(spec.defaults):])
                    raise RuntimeError(
                        "%s's %s. Required params for this subclass are: %s. "
                        "Optional params are: %s" % (
                            short_name, str(e), required, optional))
                # :(
                raise

    chain = itertools.chain.from_iterable(
        e.short_name if _is_collection(e.short_name) else [e.short_name]
        for e in sc)
    avail_names = ', '.join(str(n) for n in chain)
    raise ValueError('"%s" is not a recognized subclass. available names '
                     'are: %s' % (short_name, avail_names))


def _is_collection(obj):
    if isinstance(obj, string_types):
        return False
    return isinstance(obj, collections.Iterable)
