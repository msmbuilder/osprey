---
title: 'Osprey: Hyperparameter Optimization for Machine Learning'
tags:
  - Python
  - optimization
  - cross-validation
  - machine learning
authors:
  - name: Robert T. McGibbon
    affiliation: D.E. Shaw Research
  - name: Carlos X. Hernández
    orcid: 0000-0002-8146-5904
    affiliation: Stanford University
  - name: Matthew P. Harrigan
    affiliation: Stanford University
  - name: Steven Kearnes
    affiliation: Stanford University
  - name: Mohammad M. Sultan
    affiliation: Stanford University
  - name: Stanislaw Jastrzebski
    affiliation: Jagiellonian University
  - name: Brooke E. Husic
    affiliation: Stanford University
  - name: Vijay S. Pande
    affiliation: Stanford University
date: 1 July 2016
bibliography: paper.bib
---


# Summary

*Osprey* is a tool for hyperparameter optimization of machine learning
algorithms in Python. Hyperparameter optimization can often be an onerous
process for researchers, due to time-consuming experimental replicates,
non-convex functionals, and constant tension between exploration of global
parameter space and local optimization [@Jones1998]. We've designed *Osprey* to
provide scientists with a practical, easy-to-use way of finding optimal model
parameters. The software works seamlessly with `scikit-learn` estimators
[@scikit-learn] and supports many different search strategies for choosing the
next set of parameters with which to evaluate a given model, including gaussian
processes [@gpy2014], tree-structured Parzen estimators [@hyperopt], as well as
random and grid search. As hyperparameter optimization is an embarrassingly
parallel problem, *Osprey* can easily scale to hundreds of concurrent processes
by executing a simple command-line program multiple times. This makes it easy
to exploit large resources available in high-performance computing environments.

*Osprey* is actively maintained by researchers at Stanford University and other
institutions around the world. While originally developed to analyze
computational protein dynamics [@msmbuilder], it is applicable to any
`scikit-learn`-compatible pipeline. The source code for *Osprey* is hosted on
GitHub and has been archived to Zenodo [@osprey_archive]. Full documentation can
be found at [http://msmbuilder.org/osprey](http://msmbuilder.org/osprey).


# References
