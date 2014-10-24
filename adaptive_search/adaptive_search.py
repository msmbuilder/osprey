class AdaptiveSearchCV(object):
    def __init__(self, estimator, param_grid, cv=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        
    def next_parameters(self, history):
        # use history to compute a next set of parameters to try
        # the model at
        return {}
    
    def fit_one(self, X, parameters):
        pass