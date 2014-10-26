import os.path
from os.path import samefile, abspath
import tempfile
from osprey.utils import dict_merge
from osprey.utils import in_directory


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
