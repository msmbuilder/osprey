from __future__ import print_function, absolute_import, division

import os
import sqlite3
import shutil
import tempfile

from osprey.trials import make_session


def test_1():
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    try:
        os.chdir(dirname)
        make_session('sqlite:///osprey-trials.db')

        con = sqlite3.connect('osprey-trials.db')
        table_names = con.execute("SELECT name FROM sqlite_master "
                                  "WHERE type='table'").fetchone()
        assert table_names == (u'trials', )

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)
