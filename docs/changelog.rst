.. _changelog:

Changelog
=========

v1.2.0dev
---------

API Changes
~~~~~~~~~~~
+ ``n_folds`` and ``n_iter`` parameters have been renamed to ``n_splits`` to
  conform to the ``sklearn`` API.

New Features
~~~~~~~~~~~~
+ Added support for ``TimeSeriesSplit`` and ``LeavePOut`` cross-validators.
+ Improved ``osprey dump`` JSON output. The hyperparameters for each run are now stored along all
  the other settings in the same dictionary, allowing for subsequent easier loading and plotting.
+ Added ability to specifiy arbitrary kernels for gaussian process strategy.
+ Added ``n_jobs`` flag for ``osprey worker`` to control how many threads are
+ Added ``max_param_suggestion_retries`` entry to the config file. This limits the number of times that
+ Added the ability to specify three different acquisition functions for the gaussian processes strategy: expected


Bug Fixes
~~~~~~~~~
+ Fixed issue that was causing crashes when there was an attempt to write estimator parameters (e.g. numpy arrays) which couldn't be serialized by JSON.
+ Fixed crashes when using ``jump`` variables of type ``int``.
+ Fixed error in the way integer variables were selected from results of Gaussian processes search strategy.


v1.1.0
------

API Changes
~~~~~~~~~~~
+ Implemented ``Config.trial_results``, allowing convenient retrieval of trials as a ``pandas.DataFrame`` (`#190 <https://github.com/msmbuilder/osprey/pull/190>`_)

New Features
~~~~~~~~~~~~
+ Added random seed via CLI (`#196 <https://github.com/msmbuilder/osprey/pull/196>`_)
+ Added ``DSVDatasetLoader`` (`#175 <https://github.com/msmbuilder/osprey/pull/175>`_)
+ Added ``random_seed`` as a configurable parameter (`#164 <https://github.com/msmbuilder/osprey/pull/164>`_)

Bug Fixes
~~~~~~~~~
+ Fixed issue where ``random_seed`` was not passed to estimator (`#198 <https://github.com/msmbuilder/osprey/pull/198>`_)
+ Fixed ``bokeh.io.vplot`` deprecation warning (`#192 <https://github.com/msmbuilder/osprey/pull/192>`_)
+ Fixed ungraceful failures when using GP with a single choice in
  search space (`#191 <https://github.com/msmbuilder/osprey/pull/191>`_)
+ Fixed parsing of ``jumpvar`` (`#164 <https://github.com/msmbuilder/osprey/pull/164>`_)


v1.0.0
------

This is the first stable version of Osprey
