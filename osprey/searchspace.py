from collections import namedtuple, Iterable

import numpy as np
from sklearn.utils import check_random_state

  
class SearchSpace(object):
    def __init__(self):
        self.variables = {}

    def add_int(self, name, min, max):
        """An integer-valued variable between `min` <= x <= `max`.
        Note that the right endpoint of the interval includes `max`.
        """
        min, max = map(int, (min, max))
        if max < min:
            raise ValueError('variable %s: max < min error' % name)
        self.variables[name] = IntVariable(name, min, max)

    def add_float(self, name, min, max, warp=None):
        """A floating point-valued variable `min` <= x < `max`
        
        """
        min, max = map(float, (min, max))
        if not min < max:
            raise ValueError('variable %s: min >= max error' % name)
        if warp not in (None, 'log'):
            raise ValueError('variable %s: warp=%s is not supported. use '
                             'None or "log",' % (name, warp))
        self.variables[name] = FloatVariable(name, min, max, warp)

    def add_enum(self, name, choices):
        if not isinstance(choices, Iterable):
            raise ValueError('variable %s: choices must be iterable' % name)
        self.variables[name] = EnumVariable(name, choices)

    def __getitem__(self, name):
        return self.variables[name]

    def __iter__(self):
        return iter(self.variables.values())

    def rvs(self, seed=None):
        random = check_random_state(seed)
        return dict((param.name, param.rvs(random)) for param in self)

    def __repr__(self):
        lines = ['Search Space:'] + ['  ' + repr(var) for var in self]
        return '\n'.join(lines)


class IntVariable(namedtuple('IntVariable', ('name', 'min', 'max'))):
    __slots__ = ()
    def __repr__(self):
        return '{:<15s}\t(int)   {:8d} <= x <= {:d}'.format(self.name, self.min, self.max)

    def rvs(self, random):
        # extra +1 here because of the _inclusive_ endpoint
        return random.randint(self.min, self.max + 1)


class FloatVariable(namedtuple('FloatVariable', ('name', 'min', 'max', 'warp'))):
    __slots__ = ()

    def __repr__(self):
        return '{:<15s}\t(float) {:8f} <= x <  {:f}'.format(self.name, self.min, self.max)

    def rvs(self, random):
        if self.warp is None:
            return random.uniform(self.min, self.max)
        elif self.warp == 'log':
            return np.exp(random.uniform(np.log(self.min), np.log(self.max)))
        raise ValueError('unknown warp: %s' % self.warp)
        
class EnumVariable(namedtuple('EnumVariable', ('name', 'choices'))):
    __slots__ = ()

    def __repr__(self):
        c = [str(e) for e in self.choices]
        return '{:<15s}\t(enum)    choices = ({:s})'.format(self.name, ', '.join(c))
    
    def rvs(self, random):
        return self.choices[random.randint(len(self.choices))]
