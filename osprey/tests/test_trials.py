from __future__ import print_function, absolute_import, division

import os
import sqlite3
import shutil
import tempfile

from osprey.trials import make_session, Trial


def test_1():
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    try:
        os.chdir(dirname)
        session = make_session('sqlite:///db', project_name='abc123')
        session.add(Trial())
        session.commit()

        con = sqlite3.connect('db')
        table_names = con.execute("SELECT name FROM sqlite_master "
                                  "WHERE type='table'").fetchone()
        assert table_names == (u'trials_v3',)

        table_names = con.execute(
            "SELECT project_name FROM trials_v3").fetchone()
        assert table_names == (u'abc123',)

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)
