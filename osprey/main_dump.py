from __future__ import print_function, absolute_import, division

import csv
import json
from argparse import ArgumentDefaultsHelpFormatter

from six.moves import cStringIO

from .config import Config
from .trials import Trial


def configure_parser(sub_parsers):
    help = 'Dump history SQL database to CSV or JSON'
    p = sub_parsers.add_parser('dump', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('config', help='Path to worker config file (yaml)')
    p.add_argument('-f', '--format', choices=['csv', 'json'], default='json',
                   help='output format')

    p.set_defaults(func=execute)


def execute(args, parser):
    config = Config(args.config)

    session = config.trials()
    columns = Trial.__mapper__.columns

    if args.format == 'json':
        items = [curr.to_dict() for curr in session.query(Trial).all()]
        value = json.dumps(items)

    elif args.format == 'csv':
        buf = cStringIO()
        outcsv = csv.writer(buf)
        outcsv.writerow([column.name for column in columns])
        for curr in session.query(Trial).all():
            row = [getattr(curr, column.name) for column in columns]
            outcsv.writerow(row)
        value = buf.getvalue()

    print(value)
