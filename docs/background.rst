.. _background:

Background
==========

Theory
------

Osprey is designed to optimize the hyperparameters of machine learning models
by maximizing a cross-validation score. As an optimization problem, the key
factors here are

  - very expensive objective function evaluations (minutes to hours, or more)
  - no gradient information is available
  - tension between exploration of parameter space and local optimization (explore / exploit dilemma)

A good, if somewhat dated overview of this problem setting can be found in
Jones, Schonlau, Welch (1998) [#f1]_. The key idea is that we can procede by
fitting a **surrogate function** or **response surface**. This surrogate function needs to provide both our best guess of the function as well as our
degree of belief -- our uncertainty in the parts of parameter space that we
haven't yet explored. Does the maxima lie over there? Then at each iteration,
a new point can be selected by maximizing the **expected improvement** over
our current best solution, by maximize the expected **entropy reduction in the distribution of maxima**, [#f3]_ or a similar so-called acquisition function.


``osprey`` supports multiple :ref:`search strategies <strategy>` for choosing
the next set of hyperparameters to evaluate your model at. The most
theoretically elegant of the supported methods, Gaussian process expected improvement using the MOE backend, attacks this problem directly by modeling
the objective function as a draw from a `Gaussian process <http://en.wikipedia.org/wiki/Gaussian_process>`_.

References
----------

.. raw:: html

   <style>
   .wy-table-responsive {
       margin-bottom:0px;
   }
   </style>

.. [#f1] Jones, D. R., M. Schonlau, and W. J. Welch. `"Efficient global optimization of expensive black-box functions." <http://link.springer.com/article/10.1023/A:1008306431147>`_ *J. Global Optim.* 13.4 (1998): 455-492.
.. [#f2] Bergstra, James S., et al. `"Algorithms for hyper-parameter optimization." <http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization>`_ NIPS. 2011.
.. [#f3] Hennig, P., and C. J. Schuler. `"Entropy search for information-efficient global optimization." <http://jmlr.org/papers/volume13/hennig12a/hennig12a.pdf>`_ *JMLR* 98888.1 (2012): 1809-1837.
.. [#f4] Snoek, J., H. Larochelle, and R. P. Adams. `"Practical Bayesian optimization of machine learning algorithms." <http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms>`_ *NIPS* 2012.