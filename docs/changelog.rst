.. _changelog:

Changelog
=========

v1.1.0dev
------------------

API Changes
~~~~~~~~~~~
+ Implemented ``Config.trial_results``, allowing convenient retrieval of
  trials as a ``pandas.DataFrame`` (`#190 <https://github.com/msmbuilder/osprey/pull/190>`_)

New Features
~~~~~~~~~~~~
+ Added random seed via CLI (`#196 <https://github.com/msmbuilder/osprey/pull/196>`_)
+ Added ``DSVDatasetLoader`` (`#175 <https://github.com/msmbuilder/osprey/pull/175>`_)
+ Added ``random_seed`` as a configurable parameter (`#164 <https://github.com/msmbuilder/osprey/pull/164>`_)
+

Bug Fixes
~~~~~~~~~~~~
+ Fixed ``bokeh.io.vplot`` deprecation warning (`#192 <https://github.com/msmbuilder/osprey/pull/192>`_)
+ Fixed ungraceful failures when using GP with a single choice in
  search space (`#191 <https://github.com/msmbuilder/osprey/pull/191>`_)
+ Fixed parsing of ``jumpvar`` (`#164 <https://github.com/msmbuilder/osprey/pull/164>`_)


v1.0.0
------------------

This is the first stable version of Osprey
