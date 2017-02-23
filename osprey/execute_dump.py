from __future__ import print_function, absolute_import, division

import csv
import json
from six.moves import cStringIO
from .config import Config
from .trials import Trial


def execute(args, parser):
    config = Config(args.config, verbose=False)

    session = config.trials()
    columns = Trial.__mapper__.columns

    if args.output == 'json':
        items = [curr.to_dict() for curr in session.query(Trial).all()]
        new_items = []
        # Instead of saving the parameters on their own nested dict,
        # save them along the rest of elements
        for item in items:
            parameters = item.pop('parameters')  # remove dict
            item.update(parameters)  # update original dict with the parameters
            new_items.append(item)
        value = json.dumps(new_items)

    elif args.output == 'csv':
        buf = cStringIO()
        outcsv = csv.writer(buf)
        outcsv.writerow([column.name for column in columns])
        for curr in session.query(Trial).all():
            row = [getattr(curr, column.name) for column in columns]
            outcsv.writerow(row)
        value = buf.getvalue()

    print(value)
    return value
