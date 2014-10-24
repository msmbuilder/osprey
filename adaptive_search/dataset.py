

class MDTrajDataset(object):
    def __init__(self, filenames, topology=None, stride=1):
        self.filenames = filenames
        self.topology = topology
        self.stride = stride
        
    def __getitem__(self):
        import mdtraj as md
        return md.load(self.filenames[i], top=self.topology, stride=self.stride)
        
    def __len__(self):
        return len(self.filenames)