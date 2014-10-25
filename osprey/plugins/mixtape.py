import os
import glob

from six import iteritems

__all__ = ['trajectory_dataset', 'eval_globals']


def trajectory_dataset(trajectories, topology=None, stride=1, **kwargs):
    import mdtraj

    traj_glob = _expand_path(trajectories)
    if topology is not None:
        topology = _expand_path(topology)
    filenames = glob.glob(traj_glob)

    X = [mdtraj.load(f, top=topology, stride=stride) for f in filenames]
    y = None

    return X, y


def eval_globals():
    import mixtape.featurizer
    import mixtape.tica
    import mixtape.cluster
    import mixtape.ghmm
    import mixtape.markovstatemodel
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator

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


def _expand_path(path, base='.'):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return path
