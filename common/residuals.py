import time
import numpy as np


def evaluate_residuals(variables, states, components, start=None, counter=None):
    """Update a process model and return its scaled residual vector."""
    i = 0

    for state in states.values():
        i = state.unpack_vars(variables, i)

    if i != len(variables):
        raise ValueError(f'Solver vector has {len(variables)} values, but the model used {i}')

    eqs = []
    for component in components.values():
        eqs.extend(component.residuals())

    eqs = np.array(eqs)

    if counter is not None and counter[0] % 200 == 0:
        print('-'*20)
        run = time.perf_counter() - start
        print('Run Time:', round(run, 2), '(s)')
        print('Residuals Norm:', np.linalg.norm(eqs))
        print('Residual Evaluations:', counter[0] + 1)

    if counter is not None:
        counter[0] += 1

    return eqs
