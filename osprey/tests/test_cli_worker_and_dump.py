from __future__ import print_function, absolute_import, division
import os
import sys
import json
import shutil
import subprocess
import tempfile
from distutils.spawn import find_executable
from numpy.testing.decorators import skipif

try:
    __import__('msmbuilder')
    HAVE_MSMBUILDER = True
except:
    HAVE_MSMBUILDER = False

OSPREY_BIN = find_executable('osprey')


@skipif(not HAVE_MSMBUILDER, 'this test requires MSMBuilder')
def test_1():
    from msmbuilder.example_datasets import FsPeptide
    assert OSPREY_BIN is not None
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    FsPeptide(dirname).get()

    try:
        os.chdir(dirname)
        subprocess.check_call([OSPREY_BIN, 'skeleton', '-t', 'msmbuilder',
                              '-f', 'config.yaml'])
        subprocess.check_call([OSPREY_BIN, 'worker', 'config.yaml', '-n', '1'])
        assert os.path.exists('osprey-trials.db')

        yield _test_dump_1

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def test_2():
    assert OSPREY_BIN is not None
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()

    try:
        os.chdir(dirname)
        subprocess.check_call([OSPREY_BIN, 'skeleton', '-t', 'sklearn',
                              '-f', 'config.yaml'])
        subprocess.check_call([OSPREY_BIN, 'worker', 'config.yaml', '-n', '1'])
        assert os.path.exists('osprey-trials.db')

        subprocess.check_call([OSPREY_BIN, 'current_best', 'config.yaml'])

        yield _test_dump_1

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def _test_dump_1():
    out = subprocess.check_output(
        [OSPREY_BIN, 'dump', 'config.yaml', '-o', 'json'])
    if sys.version_info >= (3, 0):
        out = out.decode()
    json.loads(out)
