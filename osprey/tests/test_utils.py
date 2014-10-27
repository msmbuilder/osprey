from __future__ import print_function, absolute_import, division
import os.path
from os.path import samefile, abspath
import tempfile
import datetime
from osprey.utils import dict_merge, in_directory, dict_is_subset
from osprey.utils import format_timedelta, current_pretty_time


def test_dict_merge_1():
    base = {'a': 1}
    top = {'a': 2}
    assert dict_merge(base, top) == top


def test_dict_merge_2():
    base = {'a': 1}
    top = {'a': 2, 'b': 3}
    assert dict_merge(base, top) == top


def test_dict_merge_3():
    base = {'a': 1, 'b': 2}
    top = {'a': 2}
    assert dict_merge(base, top) == {'a': 2, 'b': 2}


def test_dict_merge_4():
    base = {'a': {'a': 1}, 'b': {'b': 2}}
    top = {'a': {'a': 2, 'b': 3}}
    assert dict_merge(base, top) == {'a': {'a': 2, 'b': 3}, 'b': {'b': 2}}


def test_in_directory_1():
    tempdir = tempfile.mkdtemp()
    try:
        initialdir = abspath(os.curdir)
        with in_directory(tempdir):
            assert samefile(abspath(os.curdir), tempdir)
        assert samefile(abspath(os.curdir), initialdir)
    finally:
        os.rmdir(tempdir)


def test_format_timedelta():
    print(format_timedelta(datetime.timedelta(seconds=413302.33)))


def test_current_pretty_time():
    print(current_pretty_time())


def test_dict_is_subset():
    assert dict_is_subset({'one': 1}, {'one': 1, 'two': 2})
    assert dict_is_subset({'one': 1, 'two': 2}, {'one': 1, 'two': 2})
    assert not dict_is_subset({'three': 3}, {'one': 1, 'two': 2})
    assert not dict_is_subset({'one': 1, 'two': 3}, {'one': 1, 'two': 2})

    assert dict_is_subset({'one': {'two': 2}}, {'one': {'two': 2}})
    assert dict_is_subset({'one': {'two': 2}}, {'one': {'two': 2, 'three': 3}})
    assert not dict_is_subset({'one': {'three': 3}}, {'one': {'two': 2}})
