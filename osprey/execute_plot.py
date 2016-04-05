from __future__ import print_function, absolute_import, division

from .trials import Trial
from .config import Config
from .plot import plot_1, plot_2, plot_3, plot_4

try:
    import bokeh.plotting as bk
    from bokeh.io import vplot
except ImportError:
    raise RuntimeError(
        'This command requires the Bokeh library (http://bokeh.pydata.org/) '
        'version >=0.10.0.\n\n    $ conda install bokeh  # (recommended)\n'
        'or\n    $ pip install bokeh')

PLOTS = [plot_4, plot_1, plot_2, plot_3]


def execute(args, parser):
    config = Config(args.config, verbose=False)
    with config.trialscontext() as session:
        q = (session.query(Trial)
             .filter(Trial.status == 'SUCCEEDED')
             .order_by(Trial.started))
        data = [curr.to_dict() for curr in q.all()]

    bk.output_file(args.filename, title='osprey')

    plots = []
    ss = config.search_space()
    for plot in PLOTS:
        plt = plot(data, ss)
        if plt is not None:
            plt = plt if isinstance(plt, list) else [plt]
            plots.extend(plt)

    p = vplot(*plots)
    if args.browser:
        bk.show(p)
    else:
        bk.save(p)
