from skopt import gp_minimize
from skopt import dummy_minimize

class OptimizerWrapper:
    """
    Each algorithm that want to use the optimizers need to implement a function called:
    get_optimize_params()

    TO DO
    """

    def __init__(self, algorithm_class, mode, cluster):
        self.space, self.objective = algorithm_class.get_optimize_params()
    def optimize_bayesian(self):
        """
        bayesian optimizer
        """
        best_param = gp_minimize(self.objective, self.space, n_random_starts=10, n_calls=100)
        print(best_param)
    def optimize_random(self):
        """
        random optimizer
        """
        best_param = dummy_minimize(self.objective, self.space, n_calls=1000)
