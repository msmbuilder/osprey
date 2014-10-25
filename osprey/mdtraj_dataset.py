import os
import glob

__all__ = ['MDTrajDataset']


def MDTrajDataset(trajectories, topology=None, stride=1, **kwargs):
    import mdtraj

    traj_glob = expand_path(trajectories)
    if topology is not None:
        topology = expand_path(topology)
    filenames = glob.glob(traj_glob)

    def loader():
        X = [mdtraj.load(f, top=topology, stride=stride) for f in filenames]
        y = None

        return X, y

    return loader


def expand_path(path, base='.'):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return path
