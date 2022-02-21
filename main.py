import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2, -2]),
                         xu=np.array([2, 2]))

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = X[:, 0] ** 2 + X[:, 1] ** 2
        f2 = (X[:, 0] - 1) ** 2 + X[:, 1] ** 2

        g1 = 2 * (X[:, 0] - 0.1) * (X[:, 0] - 0.9) / 0.18
        g2 = - 20 * (X[:, 0] - 0.4) * (X[:, 0] - 0.6) / 4.8

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


# Objectives.
def get_obj():
    objectives = [
        lambda x: x[0] ** 2 + x[1] ** 2,
        lambda x: (x[0] - 1) ** 2 + x[1] ** 2
    ]
    return objectives


# Constrains.
def get_cons():
    constrains = [
        lambda x: 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18,
        lambda x: - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8
    ]
    return constrains


functional_problem = FunctionalProblem(2,
                                       get_obj(),
                                       constr_ieq=get_cons(),
                                       xl=np.array([-2, -2]),
                                       xu=np.array([2, 2]))

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True)

# Run the algorithm for 40 iteration.
problem = MyProblem()
res = minimize(problem,
               algorithm,
               get_termination("n_gen", 40),
               seed=1,
               save_history=True,
               verbose=True)

# Get the Pareto-Front for plotting.
pareto_front = problem.pareto_front(use_cache=False, flatten=False)
approx_ideal_value = res.F.min(axis=0)
approx_nadir_value = res.F.max(axis=0)

# Nadir & ideal Points.
ideal0, ideal1 = approx_ideal_value[0], approx_ideal_value[1]
nadir0, nadir1 = approx_nadir_value[0], approx_nadir_value[1]

# Show plot.
plt.title("Pareto Optimal Solutions")
plt.scatter(res.F[:, 0], res.F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(ideal0, ideal1, facecolors='none', edgecolors='red', marker="*", s=100, label="Ideal Point")
plt.scatter(nadir0, nadir1, facecolors='none', edgecolors='green', marker="p", s=100, label="Nadir Point")
plt.legend()
plt.show()
