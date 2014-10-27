import os
import json
import shutil
import subprocess
import tempfile
from distutils.spawn import find_executable
from numpy.testing.decorators import skipif

try:
    __import__('mixtape')
    HAVE_MIXTAPE = True
except:
    HAVE_MIXTAPE = False

OSPREY_BIN = find_executable('osprey')


@skipif(not HAVE_MIXTAPE, 'this test requires mixtape')
def test_1():
    assert OSPREY_BIN is not None
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()

    try:
        os.chdir(dirname)
        subprocess.check_call([OSPREY_BIN, 'skeleton', '-t', 'mixtape',
                              '-f', 'config.yaml'])
        subprocess.check_call([OSPREY_BIN, 'worker', 'config.yaml', '-n', '1'])
        assert os.path.exists('osprey-trials.db')
        out = subprocess.check_output([OSPREY_BIN, 'dump', 'config.yaml',
                                       '-o', 'json'])
        json.loads(out)

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)
