Osprey
======

Osprey is a tool for practical hyperparameter optimization of machine learning
algorithms. It's designed to provide a practical, **easy to use** way for
application scientists to find parameters that maximize the cross-validation
score of a model on their dataset.

Osprey is actively being developed by researchers around the world, with primary
application areas in computational protein dynamics and drug design, and distributed
under the `Apache License (v2.0) <https://www.apache.org/licenses/LICENSE-2.0>`_.
All development takes place on `GitHub <https://github.com/msmbuilder/osprey>`_.

Overview
--------
``osprey`` is a command line tool. It runs using a simple :ref:`config file
<config_file>` which sets up the problem by describing the :ref:`estimator
<estimator>`, :ref:`search space <search_space>`, :ref:`strategy <strategy>`,
:ref:`dataset <dataset_loader>`, :ref:`cross validation <cross_validation>`,
and storage for the :ref:`results <trials>`.


Related tools include and `spearmint <https://github.com/JasperSnoek/spearmint>`_,
`hyperopt <https://hyperopt.github.io/hyperopt/>`_, and
`GPy <https://sheffieldml.github.io/GPy/>`_. Both hyperopt and GPy can serve as backend
:ref:`search strategies <strategy>` for osprey.


To get started, run ``osprey skeleton`` to create an example config file, and
then boot up one or more parallel instances of ``osprey worker``.

If you happen to run into any issues while using Osprey or would like suggest a
new feature, please take a moment to read our :ref:`Contributing <contributing>`
section.


.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 2

   background
   installation
   getting_started
   contributing
   config_file
   batch_submission

.. raw:: html

   </div>

.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
