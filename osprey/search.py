from six import iteritems
from sklearn.utils import check_random_state

__all__ = ['random', 'hyperopt_tpe']


def random(history, searchspace, random_state=None):
    """

    """
    return searchspace.rvs()


def hyperopt_tpe(history, searchspace, random_state=None):
    """


    """
    from hyperopt import Trials, tpe, Domain
    random = check_random_state(random_state)
    hp_searchspace = searchspace.to_hyperopt()

    trials = Trials()
    for i, h in enumerate(history):
        trials.insert_trial_doc({
            'book_time': None,
            'exp_key': None,
            'misc': {
                'cmd': ('domain_attachment', 'FMinIter_Domain'),
                'idxs': dict((k, [i]) for k in hp_searchspace.keys()),
                'tid': i,
                'vals': h[0],
                'workdir': None},
            'owner': None,
            'refresh_time': None,
            'result': {'loss': h[1], 'status': 'ok'},
            'spec': None,
            'state': 2,
            'tid': i,
            'version': 0})

    new_id = len(history)
    domain = Domain(lambda x: x**2, hp_searchspace, False)

    new_trials = tpe.suggest([new_id], domain, trials, random.randint(2**32-1))
    val = dict((k, v[0]) for k, v in iteritems(new_trials[0]['misc']['vals']))
    return val
