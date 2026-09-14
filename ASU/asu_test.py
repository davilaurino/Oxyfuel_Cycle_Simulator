"""Test scalar recovery optimization around the ASU fsolve solver."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASU_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.optimize import minimize_scalar

from ASU.asu_classes import O2Specification
from ASU.asu_cycle_residuals import asu_residuals
from ASU.asu_setup import build_asu
from ASU.asu_solver import build_bounds, solve_asu

SOLUTION_PATH = ASU_DIR/'0_solution.npy'
MAX_RECOVERY = 0.99
RESIDUAL_TOLERANCE = 1.0e-6


def _oxygen_specification(components):
    specifications = [
        component
        for component in components.values()
        if isinstance(component, O2Specification)
    ]
    if len(specifications) != 1:
        raise ValueError(
            'The recovery optimization requires exactly one O2Specification'
        )
    return specifications[0]


def _oxygen_recovery(specification):
    oxygen = specification.o2_index
    feed_oxygen = (
        specification.air_feed.m_dot
        * specification.air_feed.w[oxygen]
    )
    product_oxygen = (
        specification.o2_product.m_dot
        * specification.o2_product.w[oxygen]
    )
    return product_oxygen/feed_oxygen


def _solve_recovery_target(recovery_target, initial_guess):
    states, components, setup_guess = build_asu()
    specification = _oxygen_specification(components)
    specification.recovery_target = recovery_target

    x0 = initial_guess if initial_guess is not None else setup_guess

    try:
        # Keep the outer optimization output readable. Each trial is summarized
        # below after its residuals and physical bounds have been checked.
        with redirect_stdout(StringIO()):
            solution, solver_success = solve_asu(
                states,
                components,
                x0,
                solver='fsolve',
            )

        residuals = asu_residuals(solution, states, components)
    except ValueError:
        return None

    lower, upper = build_bounds(states, components)
    inside_bounds = np.all(solution >= lower) and np.all(solution <= upper)
    maximum_residual = np.max(np.abs(residuals))
    recovery = _oxygen_recovery(specification)
    purity = specification.o2_product.z[specification.o2_index]

    feasible = (
        solver_success
        and np.isfinite(maximum_residual)
        and maximum_residual <= RESIDUAL_TOLERANCE
        and inside_bounds
    )

    return solution, feasible, recovery, purity, maximum_residual


def maximize_recovery():
    states, components, setup_guess = build_asu()
    minimum_recovery = _oxygen_specification(components).recovery_target

    if SOLUTION_PATH.exists():
        initial_guess = np.load(SOLUTION_PATH)
    else:
        initial_guess = setup_guess

    if len(initial_guess) != len(setup_guess):
        raise ValueError(
            'The saved solution length does not match the current ASU model'
        )

    evaluations = {}
    best = {
        'recovery': -np.inf,
        'solution': None,
        'purity': None,
        'maximum_residual': None,
    }

    def objective(recovery_target):
        key = float(recovery_target)
        if key in evaluations:
            return evaluations[key]

        result = _solve_recovery_target(key, initial_guess)
        if result is None:
            objective_value = 1.0 + key
            print(f'Recovery target {key:.6f}: invalid thermodynamic state')
        else:
            solution, feasible, recovery, purity, maximum_residual = result
            status = 'feasible' if feasible else 'rejected'
            print(
                f'Recovery target {key:.6f}: {status}, '
                f'recovery={recovery:.6f}, purity={purity:.6f}, '
                f'max residual={maximum_residual:.3e}'
            )

            if feasible:
                objective_value = recovery
                if recovery > best['recovery']:
                    best.update({
                        'recovery': recovery,
                        'solution': solution.copy(),
                        'purity': purity,
                        'maximum_residual': maximum_residual,
                    })
            else:
                # Feasible points have negative objectives, while rejected
                # points have positive penalties that favor lower targets.
                objective_value = 1.0 + key

        evaluations[key] = objective_value
        return objective_value

    # scipy's bounded method does not necessarily evaluate either endpoint.
    objective(minimum_recovery)
    result = minimize_scalar(
        objective,
        bounds=(minimum_recovery, MAX_RECOVERY),
        method='bounded',
        options={
            'xatol': 1.0e-4,
            'maxiter': 20,
        },
    )

    print('\nOuter optimization:', result.message)
    print('Outer evaluations:', result.nfev + 1)

    if best['solution'] is None:
        print('No physically feasible fsolve solution was found.')
        return None

    print('Best feasible recovery:', best['recovery'])
    print('O2 product purity:', best['purity'])
    print('Maximum residual:', best['maximum_residual'])
    return best['solution']


if __name__ == '__main__':
    maximize_recovery()
