from __future__ import print_function, absolute_import, division
import numpy as np
import scipy.stats
from six.moves import xrange

from osprey.searchspace import SearchSpace

def test_1():
    s = SearchSpace()
    s.add_int('a', 1, 2)
    s.add_float('b', 2, 3)
    s.add_enum('c', ['a', 'b', 'c'])
    
    
    assert s['a'].min == 1
    assert s['a'].max == 2
    assert s['a'].name == 'a'

    assert s['b'].min == 2
    assert s['b'].max == 3
    assert s['b'].name == 'b'

    assert s['c'].choices == ['a', 'b', 'c']
    assert s['c'].name == 'c'


def test_2():
    s = SearchSpace()
    s.add_int('a', 0, 3)

    values = [s.rvs()['a'] for _ in xrange(100)]
    counts = np.bincount(values)
    expected = [100/4 for _ in range(4)]
    
    # print(counts),
    # pri

    p = scipy.stats.chisquare(counts, f_exp=expected)[1]    
    if p < 0.001:
        raise ValueError('distribution not being sampled correctly, p=%f' % p)

    
    


def test_3():
    s = SearchSpace()
    s.add_float('b', -2, 2)

    assert all(-2 < s.rvs()['b'] < 2 for _ in xrange(100))


def test_4():
    s = SearchSpace()
    s.add_enum('c', [True, False])

    assert all(s.rvs()['c'] in [True, False] for _ in xrange(100))


def test_5():
    s = SearchSpace()
    s.add_float('a', 1e-5, 1, warp='log')
    
    n_bins = 10
    n_samples = 1000
    
    bin_edges = np.logspace(np.log10(s['a'].min), np.log10(s['a'].max),
                            num=n_bins+1)

    values = [s.rvs()['a'] for _ in xrange(n_samples)]
    counts, bin_edges = np.histogram(values, bin_edges)
   
    p = scipy.stats.chisquare(counts, f_exp=[n_samples/n_bins] * n_bins)[1]
    
    if p < 0.001:
        raise ValueError('distribution not being sampled correctly, p=%f' % p)

