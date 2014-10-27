from __future__ import print_function, absolute_import, division
from osprey.entry_point import load_entry_point
from nose.tools import assert_raises


def test_1():
    from sklearn.cluster import KMeans
    assert load_entry_point('sklearn.cluster.KMeans') is KMeans


def test_2():
    from numpy.random import randint
    assert load_entry_point('numpy.random.randint') is randint


def test_3():
    assert load_entry_point('osprey.entry_point.load_entry_point') \
        is load_entry_point


def test_4():
    with assert_raises(RuntimeError):
        load_entry_point('sklearn')

    with assert_raises(RuntimeError):
        load_entry_point('sklearn.sdsdfjhgdsf')
