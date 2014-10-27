import os
import shutil
import subprocess
import tempfile
import yaml
from distutils.spawn import find_executable

OSPREY_BIN = find_executable('osprey')


def test_1():
    assert OSPREY_BIN is not None
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()

    try:
        os.chdir(dirname)
        subprocess.check_call([OSPREY_BIN, 'createrc', '-l', 'curdir',
                              '-t', 'mixtape'])
        assert os.path.exists('.ospreyrc')
        with open('.ospreyrc', 'rb') as f:
            yaml.load(f)

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)
