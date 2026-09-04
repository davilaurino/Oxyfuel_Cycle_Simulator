import utils
import time
import numpy as np

def allam_residuals(vars, S, Component, start, counter):
    idx = 0
    residuals = []

    # Unpacking variables
    for state in S:
        idx = state.unpack_vars(vars, idx)
        state.h = utils.mixture_enthalpy(state.T, state.P, state.y, state.spc)
    for comp_name in Component:
        idx = Component[comp_name].unpack_vars(vars, idx)

    for comp_name in Component:
        residuals.extend(Component[comp_name].residuals())

    if counter[0] % 120 == 0:
        run = time.time() - start
        print('Run Time:', round(run, 2), '(s)')
        print("Residuals Norm:", np.linalg.norm(residuals))

        top = np.argsort(np.abs(residuals))[-3:]
        for i in top:
            print(f"  Residual[{i}] = {residuals[i]:.8f}")

    counter[0] += 1

    return residuals