Getting Started
===============

Introduction
------------

Getting started with Osprey is as simple as setting up a single ``YAML``
configuration file. This configuration file will contain your model
estimators, hyperparameter search strategy, hyperparameter search space,
dataset information, cross-validation strategy, and a path to a
``SQL``-like database. You can use the command ``osprey skeleton`` to
generate an example configuration file.

First, we will describe how to prepare your dataset for Osprey. Then, we will
show how to use Osprey for a simple scikit-learn classification task. We'll
also demonstrate how one might use Osprey to model a
`molecular dynamics (MD) <https://en.wikipedia.org/wiki/Molecular_dynamics>`_
dataset. And finally, we'll show how to query and use your final Osprey results.


Formatting Your Dataset
-----------------------

Osprey supports a wide variety of file formats (see `here
<./config_file.html#dataset-loader>`_ for a full list); however, some of these offer
more flexibility than others. In general, your data should be formatted as a
two-dimensional array, where columns represent different features or variables
and rows are individual observations. This is a fairly natural format for
delimiter-separated value files (e.g ``.csv``, ``.tsv``), which Osprey handles
natively using ``DSVDatasetLoader``. If you choose to save your dataset as a
``.pkl``, ``.npz``, or ``.npy`` file, it's as simple as saving your datasets as
2d NumPy arrays. Note that each file should only contain a single NumPy array.
If you'd like to store multiple arrays to a single file for Osprey to read, we
recommend storing your data in an HDF5 file.

When working with datasets with labels or a response variable, there are slight
differences in how your data should be stored. With delimiter-separated value,
NumPY files, and HDF5 files, you can simply append these as an additional
column and then select its index as the ``y_col`` parameter in the corresponding
dataset loader. With Pickle and JobLib files, you should instead save each as a
separate value in a ``dict`` object and declare the corresponding keys
(``x_name`` and ``y_name``) in the ``JoblibDatasetLoader``. Please note that if
you wish to use multiple response variables the ``JoblibDatasetLoader`` is the
only dataset loader currently equipped to do so.


SVM Classification with ``scikit-learn``
----------------------------------------

Let's train a basic C-Support Vector Classification example using
``scikit-learn`` and introduce the basic ``YAML`` fields for Osprey. To
tell Osprey that we want to use ``sklearn``'s ``SVC`` as our estimator, we
can type:

.. code:: yaml

    estimator:
      entry_point: sklearn.svm.SVC

If we want to use random search to decide where to search next in
hyperparameter space, we can add:

.. code:: yaml

    strategy:
      name: random

The search space can be defined for any hyperparameter available in the
``estimator`` class. Here we can adjust the value range of the ``C`` and
``gamma`` hyperparamters. We'll search over a range of 0.1 to 10 for
``C`` and over 1E-5 to 1 in log-space (note: ``warp: log``) for
``gamma``.

.. code:: yaml

    search_space:
      C:
        min: 0.1
        max: 10
        type: float

      gamma:
        min: 1e-5
        max: 1
        warp: log
        type: float

To perform 5-fold cross validation, we add:

.. code:: yaml

    cv: 5

To load the digits classification example dataset from ``scikit-learn``,
we write:

.. code:: yaml

    dataset_loader:
      name: sklearn_dataset
      params:
        method: load_digits

And finally we need to list the SQLite database where our cross-validation
results will be saved:

.. code:: yaml

    trials:
        uri: sqlite:///osprey-trials.db

Once this all has been written to a ``YAML`` file (in this example
``config.yaml``), we can start an osprey job in the command-line by invoking:

.. code:: bash

    $ osprey worker config.yaml


Molecular Dynamics with ``msmbuilder``
--------------------------------------

Now that we understand the basics, we can move on to a more practical example.
This section will go over how to set up a Osprey configuration for
cross-validating Markov state models from protein simulations. Our model will
be constructed by first calculating torsion angles, performing dimensionality
reduction using tICA, clustering using mini-batch k-means, and, finally, an
maximum-likelihood estimated Markov state model.

We begin by defining a ``Pipeline`` which will construct our desired model:

.. code:: yaml

    estimator:
        eval: |
            Pipeline([
                    ('featurizer', DihedralFeaturizer()),
                    ('tica', tICA()),
                    ('cluster', MiniBatchKMeans()),
                    ('msm', MarkovStateModel(n_timescales=5, verbose=False)),
            ])
        eval_scope: msmbuilder

Notice that we can easily set default parameters (e.g. ``msm.n_timescales``)
in our ``Pipeline`` even if we don't plan on optimizing them.

If we wish to use `gaussian process
prediction <https://en.wikipedia.org/wiki/Gaussian_process#Gaussian_process_prediction.2C_or_kriging>`__
to decide where to search in hyperparameter space, we can add:

.. code:: yaml

    strategy:
        name: gp
        params:
          seeds: 50

In this example, we'll be optimizing the type of featurization,
the number of cluster centers and the number of independent components:

.. code:: yaml

    search_space:

    featurizer__types:
      choices:
        - ['phi', 'psi']
        - ['phi', 'psi', 'chi1']
      type: enum

    tica__n_components:
      min: 2
      max: 5
      type: int

    cluster__n_clusters:
      min: 10
      max: 100
      type: int

As seen in the previous example, we'll set ``tica__n_components`` and
``cluster__n_clusters`` as integers with a set range. Notice that we can
change which torsion angles to use in our featurization by creating an ``enum``
which contains a list of different dihedral angle types.


In this example, we'll be using 50-50 ``shufflesplit`` cross-validation.
This method is optimal for Markov state model cross-validation, as it maximizes
the amount of unique data available in your training and test sets:

.. code:: yaml

    cv:
      name: shufflesplit
    params:
      n_iter: 5
      test_size: 0.5

We'll be using MDTraj to load our trajectories. Osprey already includes an
``mdtraj`` dataset loader to make it easy to list your trajectory and topology
files as a glob-string:

.. code:: yaml

    dataset_loader:
      name: mdtraj
      params:
        trajectories: ~/local/msmbuilder/Tutorial/XTC/*/*.xtc
        topology: ~/local/msmbuilder/Tutorial/native.pdb
        stride: 1

And finally we need to list the SQLite database where our cross-validation
results will be saved:

.. code:: yaml

    trials:
      uri: sqlite:///osprey-trials.db


Just as before, once this all has been written to a ``YAML`` file
we can start an osprey job in the command-line by invoking:

.. code:: bash

    $ osprey worker config.yaml


Working with Osprey Results
---------------------------

As mentioned before, all Osprey results are stored in an SQL-like database, as
define by the ``trials`` field in the configuration file. This makes querying
and reproducing Osprey results fairly simple.

Osprey provides two command-line tools to quickly digest your results:
``current_best`` and ``plot``. ``current_best``, as the name suggests, prints
out the best scoring model currently in your trials database, as well as the
parameters used to create it. Here's some example output from our SVM
classification tutorial above:

.. code:: bash

    $ osprey current_best config.yaml

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Best Current Model = 0.975515 +- 0.013327
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    		 C 	 7.957695018309156
    		 gamma 	 0.0004726222555749291


This is useful if you just want to get a sense of how well your trials are
doing or just want to quickly get the best current result from Osprey.
The ``plot`` functionality provides interactive HTML charts using ``bokeh``
(note that ``bokeh`` must be installed to use ``osprey plot``).

.. code:: bash

    $ osprey plot config.yaml

The command above opens a browser window with a variety of plots. An example of
one such plot, showing the running best SVM model over many iterations, can be
seen below:

.. raw:: html

  <link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.12.1.min.css" type="text/css">
  <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.12.1.min.js"></script>
    <div class="bk-root">
        <div class="bk-grid-column bk-layout-fixed" id="modelid_64b617a4-4731-46a4-bd93-c53ecf9ab729" style="width: 600px; height: 600px;"><div class="bk-plot-layout bk-layout-fixed" id="modelid_ebd75eaf-eeb3-404d-b2ed-5691d769ebc8" style="width: 600px; height: 600px;"><div class="bk-toolbar-wrapper bk-layout-null" id="modelid_3377e86d-8aff-4853-9bba-412ba8ec675e" style="left: 570px; top: 28.964px; width: 30px; height: 571.036px;"><div class="bk-toolbar-right bk-plot-right bk-toolbar-sticky bk-toolbar-active">

        <a href="http://bokeh.pydata.org/" target="_blank" class="bk-logo bk-logo-small"></a>

        <div class="bk-button-bar">
        <ul class="bk-button-bar-list" type="pan"><li><button type="button" class="bk-toolbar-button hover active">
        <div class="bk-btn-icon bk-tool-icon-pan"></div>
        <span class="tip">Pan</span>
        </button>
        </li><li><button type="button" class="bk-toolbar-button hover">
        <div class="bk-btn-icon bk-tool-icon-box-zoom"></div>
        <span class="tip">Box Zoom</span>
        </button>
        </li></ul>
        <ul class="bk-button-bar-list" type="scroll"><li><button type="button" class="bk-toolbar-button hover">
        <div class="bk-btn-icon bk-tool-icon-wheel-zoom"></div>
        <span class="tip">Wheel Zoom</span>
        </button>
        </li></ul>
        <ul class="bk-button-bar-list" type="pinch"></ul>
        <ul class="bk-button-bar-list" type="tap"></ul>
        <ul class="bk-button-bar-list" type="press"></ul>
        <ul class="bk-button-bar-list" type="rotate"></ul>
        <ul class="bk-button-bar-list" type="actions"><li><button type="button" class="bk-toolbar-button hover">
        <div class="bk-btn-icon bk-tool-icon-reset"></div>
        <span class="tip">Reset</span>
        </button>
        </li></ul>
        <div class="bk-button-bar-list bk-bs-dropdown" type="inspectors"><a href="#" data-bk-bs-toggle="dropdown" class="bk-bs-dropdown-toggle">inspect <span class="bk-bs-caret"></span></a><ul class="bk-bs-dropdown-menu"><li><div class="bk-toolbar-inspector"><input type="checkbox" checked="">Hover Tool
        </div></li></ul></div>
        <ul class="bk-button-bar-list" type="help"></ul>
        </div>
        </div>
        </div><div class="bk-plot-wrapper" style="position: absolute; left: 0px; top: 0px; width: 600px; height: 600px;"><div class="bk-canvas-wrapper" style="touch-action: none; -webkit-user-select: none; -webkit-user-drag: none; -webkit-tap-highlight-color: rgba(0, 0, 0, 0); cursor: crosshair; width: 600px; height: 600px;">
        <div class="bk-canvas-events"></div>
        <div class="bk-canvas-overlays"><div class="bk-shading" style="display: none;"></div><div class="bk-tooltip" style="z-index: 1010; display: none;"></div><div class="bk-tooltip" style="z-index: 1010; display: none;"></div></div>
        <canvas class="bk-canvas" width="1200" height="1200" style="width: 600px; height: 600px;"></canvas></div></div></div></div>
        </div>

        <script type="text/javascript">
            Bokeh.$(function() {
            var docs_json = {"b0c1333e-e97d-4ca6-b96c-a6516ece11bd":{"roots":{"references":[{"attributes":{},"id":"a10e16f7-bce9-41d8-8df3-4435826317a9","type":"ToolEvents"},{"attributes":{"axis_label":"Score","formatter":{"id":"67f9b82e-45b7-4c3c-a88f-ae31ef2f2745","type":"BasicTickFormatter"},"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"},"ticker":{"id":"174627a6-11a3-4b18-bf9e-c4001eb0b51f","type":"BasicTicker"}},"id":"56302ce1-b4d8-46aa-85a8-82445a4588af","type":"LinearAxis"},{"attributes":{"children":[{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"}]},"id":"64b617a4-4731-46a4-bd93-c53ecf9ab729","type":"Column"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"radius":{"units":"data","value":3},"x":{"field":"x"},"y":{"field":"y"}},"id":"f69fe557-7e4d-4a76-a8f6-95d5872f82ac","type":"Circle"},{"attributes":{},"id":"174627a6-11a3-4b18-bf9e-c4001eb0b51f","type":"BasicTicker"},{"attributes":{"overlay":{"id":"1f471975-36a2-45d0-bec2-a94d49ed2062","type":"BoxAnnotation"},"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"}},"id":"b6a42f6a-09ab-41ac-aa74-9e52c3f37c9a","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"58b36642-69ce-4ca3-896d-5db0352c728c","type":"ColumnDataSource"},"glyph":{"id":"dd4c188e-ac1c-42b7-94de-2bd30f73551e","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"f69fe557-7e4d-4a76-a8f6-95d5872f82ac","type":"Circle"},"selection_glyph":null},"id":"27358ecc-4228-4b23-a33d-8a2dfced401e","type":"GlyphRenderer"},{"attributes":{"axis_label":"Iteration number","formatter":{"id":"cd297ec9-a2c1-406e-87d8-328657bece2b","type":"BasicTickFormatter"},"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"},"ticker":{"id":"d7534da2-068b-4637-ad4d-6fb08ca788fb","type":"BasicTicker"}},"id":"6e71ad64-6069-4a34-8368-c9d8495b540e","type":"LinearAxis"},{"attributes":{"line_color":{"value":"#1f77b4"},"line_width":{"value":2},"x":{"field":"x"},"y":{"field":"y"}},"id":"d03378f8-a044-465e-b7e8-b8c506f31a71","type":"Line"},{"attributes":{"callback":null},"id":"d23a5d58-29f8-46d2-a167-c5bb254c489d","type":"DataRange1d"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"1f471975-36a2-45d0-bec2-a94d49ed2062","type":"BoxAnnotation"},{"attributes":{"dimension":1,"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"},"ticker":{"id":"174627a6-11a3-4b18-bf9e-c4001eb0b51f","type":"BasicTicker"}},"id":"12989957-980f-4081-b5cf-9310dbb4561b","type":"Grid"},{"attributes":{"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"},"ticker":{"id":"d7534da2-068b-4637-ad4d-6fb08ca788fb","type":"BasicTicker"}},"id":"7282b708-fff4-4106-9a85-7959116fe900","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.6},"fill_color":{"value":"#1f77b4"},"line_color":{"value":null},"radius":{"units":"data","value":3},"x":{"field":"x"},"y":{"field":"y"}},"id":"dd4c188e-ac1c-42b7-94de-2bd30f73551e","type":"Circle"},{"attributes":{},"id":"d7534da2-068b-4637-ad4d-6fb08ca788fb","type":"BasicTicker"},{"attributes":{},"id":"67f9b82e-45b7-4c3c-a88f-ae31ef2f2745","type":"BasicTickFormatter"},{"attributes":{"plot":null,"text":"Running best"},"id":"d554689a-fb96-4108-8063-50bbe0609315","type":"Title"},{"attributes":{"callback":null,"column_names":["x","y"],"data":{"x":[1,2,6,12,25],"y":[0.22815804117974403,0.7156371730662214,0.9727323316638843,0.9749582637729549,0.9755147468002225]}},"id":"e80dea5f-e80f-4e3d-87f6-fe223cd02a09","type":"ColumnDataSource"},{"attributes":{"callback":null,"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"},"tooltips":[["index","$index"],["C","@C"],["gamma","@gamma"]]},"id":"47f6c882-d0a5-4af4-bf11-71f3d65561ab","type":"HoverTool"},{"attributes":{},"id":"cd297ec9-a2c1-406e-87d8-328657bece2b","type":"BasicTickFormatter"},{"attributes":{"callback":null},"id":"9db93b43-31ab-4e50-852f-5e2109580755","type":"DataRange1d"},{"attributes":{"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"}},"id":"0ddc9d2e-4ef8-4135-9689-832a703472c5","type":"PanTool"},{"attributes":{"callback":null,"column_names":["index","C","gamma","x","y"],"data":{"C":[2.3870007525421038,7.818940902700415,8.60033998880039,7.815321015495774,7.957695018309156],"gamma":[0.023591356016345048,0.009643857615941434,0.0009923851706765986,0.000617986534858686,0.0004726222555749291],"index":[0,1,5,11,24],"x":[1,2,6,12,25],"y":[0.22815804117974403,0.7156371730662214,0.9727323316638843,0.9749582637729549,0.9755147468002225]}},"id":"58b36642-69ce-4ca3-896d-5db0352c728c","type":"ColumnDataSource"},{"attributes":{"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"}},"id":"a279466f-f0f6-499c-ad80-e62e9eda3c80","type":"WheelZoomTool"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_width":{"value":2},"x":{"field":"x"},"y":{"field":"y"}},"id":"9facb964-c898-4703-8d68-95f535eaf0a0","type":"Line"},{"attributes":{"plot":{"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"}},"id":"ded9a7f6-078b-4f12-b1a1-271656fc62b0","type":"ResetTool"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"0ddc9d2e-4ef8-4135-9689-832a703472c5","type":"PanTool"},{"id":"a279466f-f0f6-499c-ad80-e62e9eda3c80","type":"WheelZoomTool"},{"id":"b6a42f6a-09ab-41ac-aa74-9e52c3f37c9a","type":"BoxZoomTool"},{"id":"ded9a7f6-078b-4f12-b1a1-271656fc62b0","type":"ResetTool"},{"id":"47f6c882-d0a5-4af4-bf11-71f3d65561ab","type":"HoverTool"}]},"id":"3377e86d-8aff-4853-9bba-412ba8ec675e","type":"Toolbar"},{"attributes":{"data_source":{"id":"e80dea5f-e80f-4e3d-87f6-fe223cd02a09","type":"ColumnDataSource"},"glyph":{"id":"d03378f8-a044-465e-b7e8-b8c506f31a71","type":"Line"},"hover_glyph":null,"nonselection_glyph":{"id":"9facb964-c898-4703-8d68-95f535eaf0a0","type":"Line"},"selection_glyph":null},"id":"b52001b2-507c-4c95-a7fa-5ce4d1cfdbe8","type":"GlyphRenderer"},{"attributes":{"below":[{"id":"6e71ad64-6069-4a34-8368-c9d8495b540e","type":"LinearAxis"}],"left":[{"id":"56302ce1-b4d8-46aa-85a8-82445a4588af","type":"LinearAxis"}],"renderers":[{"id":"6e71ad64-6069-4a34-8368-c9d8495b540e","type":"LinearAxis"},{"id":"7282b708-fff4-4106-9a85-7959116fe900","type":"Grid"},{"id":"56302ce1-b4d8-46aa-85a8-82445a4588af","type":"LinearAxis"},{"id":"12989957-980f-4081-b5cf-9310dbb4561b","type":"Grid"},{"id":"1f471975-36a2-45d0-bec2-a94d49ed2062","type":"BoxAnnotation"},{"id":"27358ecc-4228-4b23-a33d-8a2dfced401e","type":"GlyphRenderer"},{"id":"b52001b2-507c-4c95-a7fa-5ce4d1cfdbe8","type":"GlyphRenderer"}],"title":{"id":"d554689a-fb96-4108-8063-50bbe0609315","type":"Title"},"tool_events":{"id":"a10e16f7-bce9-41d8-8df3-4435826317a9","type":"ToolEvents"},"toolbar":{"id":"3377e86d-8aff-4853-9bba-412ba8ec675e","type":"Toolbar"},"x_range":{"id":"d23a5d58-29f8-46d2-a167-c5bb254c489d","type":"DataRange1d"},"y_range":{"id":"9db93b43-31ab-4e50-852f-5e2109580755","type":"DataRange1d"}},"id":"ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","subtype":"Figure","type":"Plot"}],"root_ids":["64b617a4-4731-46a4-bd93-c53ecf9ab729"]},"title":"Bokeh Application","version":"0.12.1"}};
            var render_items = [{"docid":"b0c1333e-e97d-4ca6-b96c-a6516ece11bd","elementid":"modelid_ebd75eaf-eeb3-404d-b2ed-5691d769ebc8","modelid":"64b617a4-4731-46a4-bd93-c53ecf9ab729"}];

            Bokeh.embed.embed_items(docs_json, render_items);
        });
        </script>


An alternative way to access trial data is to use the Python API to directly
access the SQL-like database. Here's an example of loading your Osprey results
as a ``pandas.DataFrame``:

.. code:: python

    # Imports
    from osprey.config import Config

    # Load Configuation File
    my_config = 'path/to/config.xml'
    config = Config(my_config)

    # Retrieve Trial Results
    df = config.trial_results()
