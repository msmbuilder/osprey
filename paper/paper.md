---
title: 'Osprey: Hyperparameter Optimization for Machine Learning'
tags:
  - Python
  - optimization
  - cross-validation
  - machine learning
authors:
  - name: Robert T. McGibbon
    affiliation: 1
  - name: Carlos X. Hernández
    orcid: 0000-0002-8146-5904
    affiliation: 1
  - name: Matthew P. Harrigan
    affiliation: 1
  - name: Steven Kearnes
    affiliation: 1
  - name: Mohammad M. Sultan
    affiliation: 1
  - name: Stanislaw Jastrzebski
    affiliation: 2
  - name: Brooke E. Husic
    affiliation: 1
  - name: Vijay S. Pande
    affiliation: 1
affiliations:
  - name: Stanford University
    index: 1
  - name: Jagiellonian University
    index: 2
date: 5 August 2016
bibliography: paper.bib
repository: https://github.com/msmbuilder/osprey
archive_doi: http://dx.doi.org/10.5281/zenodo.56251
---


# Summary

*Osprey* is a tool for hyperparameter optimization of machine learning
algorithms in Python. Hyperparameter optimization can often be an onerous
process for researchers, due to time-consuming experimental replicates,
non-convex objective functions, and constant tension between exploration of
global parameter space and local optimization [@Jones1998]. We've designed
*Osprey* to provide scientists with a practical, easy-to-use way of finding
optimal model parameters. The software works seamlessly with `scikit-learn`
estimators [@scikit-learn] and supports many different search strategies for
choosing the next set of parameters with which to evaluate a given model,
including gaussian processes [@gpy2014], tree-structured Parzen estimators
[@hyperopt], as well as random and grid search. As hyperparameter optimization
is an embarrassingly parallel problem, *Osprey* can easily scale to hundreds of
concurrent processes by executing a simple command-line program multiple times.
This makes it easy to exploit large resources available in high-performance
computing environments.

*Osprey* is actively maintained by researchers at Stanford University and other
institutions around the world. While originally developed to analyze
computational protein dynamics [@msmbuilder], it is applicable to any
`scikit-learn`-compatible pipeline. The source code for *Osprey* is hosted on
GitHub and has been archived to Zenodo [@osprey_archive]. Full documentation can
be found at [http://msmbuilder.org/osprey](http://msmbuilder.org/osprey).


# References
