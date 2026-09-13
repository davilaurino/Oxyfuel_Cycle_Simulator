import time

import numpy as np


def asu_residuals(variables, states, components, start=None, counter=None):
    """Update the ASU model and return its scaled residual vector."""
    i = 0

    for state in states.values():
        i = state.unpack_vars(variables, i)

    for component in components.values():
        i = component.unpack_vars(variables, i)

    if i != len(variables):
        raise ValueError(f'Solver vector has {len(variables)} values, but the model used {i}')

    eqs = []
    for component in components.values():
        eqs.extend(component.residuals())

    eqs = np.array(eqs)

    if counter is not None and counter[0] % 1000 == 0:
        run = time.perf_counter() - start
        print('Run Time:', round(run, 2), '(s)')
        print('Residuals Norm:', np.linalg.norm(eqs))

        top = np.argsort(np.abs(eqs))[-5:]
        for i in top:
            print(f'  Residual[{i}] = {eqs[i]:.8f}')

    if counter is not None:
        counter[0] += 1

    return eqs
