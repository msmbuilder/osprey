from six import iteritems
from sklearn.utils import check_random_state

__all__ = ['random', 'hyperopt_tpe']


def random(history, bounds, random_state=None):
    """

    """

    rval = {}
    random = check_random_state(random_state)

    for param_name, info in iteritems(bounds):
        low, high = info['min'], info['max']
        if info['type'] == 'int':
            rval[param_name] = random.randint(low, high)
        elif info['type'] == 'float':
            rval[param_name] = random.uniform(low, high)
        else:
            raise ValueError('type should be int/float')

    return rval


def hyperopt_tpe(history, bounds, random_state=None):
    """


    """
    from hyperopt import hp, pyll, Trials, tpe, Domain
    random = check_random_state(random_state)

    hp_searchspace = {}
    for param_name, info in iteritems(bounds):
        if info['type'] == 'int':
            hp_searchspace['param_name'] = pyll.scope.int(hp.quniform(
                param_name, info['min'], info['max'], q=1))
        elif info['type'] == 'float':
            hp_searchspace['param_name'] = hp.uniform(
                param_name, info['min'], info['max'])
        else:
            raise ValueError('type should be int/float')

    trials = Trials()
    for i, h in enumerate(history):
        trials.insert_trial_doc({
            'book_time': None,
            'exp_key': None,
            'misc': {
                'cmd': ('domain_attachment', 'FMinIter_Domain'),
                'idxs': dict((k, [i]) for k in bounds.keys()),
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
