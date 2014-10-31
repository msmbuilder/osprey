from __future__ import print_function, absolute_import, division

import numpy as np
from matplotlib import cm
from matplotlib.colors import rgb2hex
from sklearn.manifold import TSNE
from collections import OrderedDict

from .trials import Trial
from .config import Config

try:
    import pandas as pd
    import bokeh.plotting as bk
    from bokeh.objects import HoverTool, ColumnDataSource
except ImportError:
    raise RuntimeError(
        'This command requires the Bokeh library (http://bokeh.pydata.org/). '
        '\n\n    $ conda install bokeh  # (recommended)\n'
        'or\n    $ pip install bokeh')

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover"


def execute(args, parser):
    config = Config(args.config, verbose=False)
    with config.trialscontext() as session:
        q = (session.query(Trial)
             .filter(Trial.status == 'SUCCEEDED')
             .order_by(Trial.started))
        data = [curr.to_dict() for curr in q.all()]

    bk.output_file(args.filename, title='osprey')

    plot_4(data)
    plot_1(data)
    plot_2(data)
    plot_3(data, config.search_space())

    if args.browser:
        bk.show()
    else:
        bk.save()


def plot_1(data):
    """Plot 1. All iterations (scatter plot)"""
    df_all = pd.DataFrame(data)
    df_params = nonconstant_parameters(data)
    build_scatter_tooltip(
        x=df_all['id'], y=df_all['mean_cv_score'], tt=df_params,
        title='All Iterations')


def plot_2(data):
    """Plot 2. Running best score (scatter plot)"""
    df_all = pd.DataFrame(data)
    df_params = nonconstant_parameters(data)
    x = [df_all['id'][0]]
    y = [df_all['mean_cv_score'][0]]
    params = [df_params.loc[0]]
    for i in range(len(df_all)):
        if df_all['mean_cv_score'][i] > y[-1]:
            x.append(df_all['id'][i])
            y.append(df_all['mean_cv_score'][i])
            params.append(df_params.loc[i])
    build_scatter_tooltip(
        x=x, y=y, tt=pd.DataFrame(params), title='Running best')


def plot_3(data, ss):
    """t-SNE embedding of the parameters, colored by score
    """
    scores = np.array([d['mean_cv_score'] for d in data])
    # maps each parameters to a vector of floats
    warped = np.array([ss.point_to_moe(d['parameters']) for d in data])

    # Embed into 2 dimensions with t-SNE
    X = TSNE(n_components=2).fit_transform(warped)

    e_scores = np.exp(scores)
    mine, maxe = np.min(e_scores), np.max(e_scores)
    color = (e_scores - mine) / (maxe - mine)
    mapped_colors = map(rgb2hex, cm.get_cmap('RdBu_r')(color))

    bk.figure(title='t-SNE (unsupervised)')
    bk.hold()
    df_params = nonconstant_parameters(data)
    df_params['score'] = scores
    bk.circle(
        X[:, 0], X[:, 1], color=mapped_colors, radius=1,
        source=ColumnDataSource(df_params), fill_alpha=0.6,
        line_color=None, tools=TOOLS)
    cp = bk.curplot()
    hover = cp.select(dict(type=HoverTool))
    format_tt = [(s, '@%s' % s) for s in df_params.columns]
    hover.tooltips = OrderedDict([("index", "$index")] + format_tt)

    xax, yax = bk.axis()
    xax.axis_label = 't-SNE coord 1'
    yax.axis_label = 't-SNE coord 2'


def plot_4(data):
    """Scatter plot of score vs each param
    """
    params = nonconstant_parameters(data)
    scores = np.array([d['mean_cv_score'] for d in data])
    order = np.argsort(scores)

    for key in params.keys():
        if params[key].dtype == np.dtype('bool'):
            params[key] = params[key].astype(np.int)

    for key in params.keys():
        x = params[key][order]
        y = scores[order]
        params = params.loc[order]
        try:
            radius = (np.max(x) - np.min(x)) / 100.0
        except:
            print("error making plot4 for '%s'" % key)
            continue

        build_scatter_tooltip(
            x=x, y=y, radius=radius, add_line=False, tt=params,
            xlabel=key, title='Score vs %s' % key)
        # bk.line(x, y, line_width=2)


def nonconstant_parameters(data):
    assert len(data) > 0
    df = pd.DataFrame([d['parameters'] for d in data])
    # http://stackoverflow.com/a/20210048/1079728
    filtered = df.loc[:, (df != df.ix[0]).any()]
    return filtered


def build_scatter_tooltip(x, y, tt, add_line=True, radius=3, title='My Plot',
                          xlabel='Iteration number', ylabel='Score'):
    bk.figure(title=title)
    bk.hold()
    bk.circle(
        x, y, radius=radius, source=ColumnDataSource(tt),
        fill_alpha=0.6, line_color=None, tools=TOOLS)

    if add_line:
        bk.line(x, y, line_width=2)

    xax, yax = bk.axis()
    xax.axis_label = xlabel
    yax.axis_label = ylabel

    cp = bk.curplot()
    hover = cp.select(dict(type=HoverTool))
    format_tt = [(s, '@%s' % s) for s in tt.columns]
    hover.tooltips = OrderedDict([("index", "$index")] + format_tt)
