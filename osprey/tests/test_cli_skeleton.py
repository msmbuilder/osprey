from __future__ import print_function, absolute_import, division
import os
import shutil
import subprocess
import tempfile
import yaml
from distutils.spawn import find_executable

from osprey.config import Config

OSPREY_BIN = find_executable('osprey')


def test_1():
    assert OSPREY_BIN is not None
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()

    try:
        os.chdir(dirname)
        subprocess.check_call([OSPREY_BIN, 'skeleton', '-t', 'mixtape',
                              '-f', 'config.yaml'])
        assert os.path.exists('config.yaml')
        with open('config.yaml', 'rb') as f:
            yaml.load(f)
        Config('config.yaml')

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)
