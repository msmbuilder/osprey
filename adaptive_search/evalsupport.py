from sklearn.base import BaseEstimator
from six import iteritems


def mixtape_globals():
    import mixtape.featurizer
    import mixtape.tica
    import mixtape.cluster
    import mixtape.ghmm
    import mixtape.markovstatemodel
    from sklearn.pipeline import Pipeline

    modules = [mixtape.featurizer, mixtape.tica, mixtape.cluster,
               mixtape.markovstatemodel, mixtape.ghmm]

    scope = {'Pipeline': Pipeline}
    for m in modules:
        for key, item in iteritems(m.__dict__):
            try:
                if issubclass(item, BaseEstimator) and not key.startswith('_'):
                    scope[key] = item
            except TypeError:
                pass
    return scope
