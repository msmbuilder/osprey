from __future__ import print_function, absolute_import, division
from osprey.utils import dict_merge


def test_merge_1():
    base = {'a': {'1': 'a', '2': 'd'}}
    top = {'a': {'1': 'b'}}

    result = dict_merge(base, top)
    assert result == {'a': {'1': 'b', '2': 'd'}}


def test_merge_2():
    base = {}
    top = {'a': {'1': 'b'}}

    result = dict_merge(base, top)
    assert result == {'a': {'1': 'b'}}


def test_merge_3():
    base = {'a': {'1': 'b'}}
    top = {}

    result = dict_merge(base, top)
    assert result == {'a': {'1': 'b'}}
