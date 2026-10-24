"""Solve, report, and save the coupled LOX-ASU/ALLAM configuration."""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTEGRATED_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from INTEGRATED import integrated_config
from INTEGRATED.integrated_reporting import integrated_report, results_table
from INTEGRATED.integrated_setup import (
    build_integrated_cycle,
    load_solution,
    standalone_initial_guess,
)
from common.residuals import ResidualScaling, evaluate_residuals
from common.solver import solve

SOLVER = 'fsolve'  # 'fsolve' or 'least_squares'


def main():
    states, components, x0 = build_integrated_cycle()
    previous = input(
        'Start from previous integrated solution (p), '
        'standalone solutions (b), or setup (s)? '
    ).strip().lower()
    solution_path = INTEGRATED_DIR / '0_solution.npy'
    if previous not in ('p', 'b', 's'):
        raise ValueError(f'Unknown starting point: {previous}')
    if previous == 'p' and solution_path.exists():
        x0 = load_solution(solution_path, len(x0))
    elif previous in ('p', 'b'):
        if previous == 'p':
            print('No integrated solution found; trying standalone solutions.')
        try:
            x0 = standalone_initial_guess(states)
        except FileNotFoundError:
            print('Standalone solutions are missing; using setup values.')

    x_final, success = solve(
        states, components, x0, solver=SOLVER, config=integrated_config,
    )
    residuals = evaluate_residuals(
        x_final, states, components,
        ResidualScaling.from_config(integrated_config),
    )
    closed = (
        np.all(np.isfinite(residuals))
        and np.max(np.abs(residuals)) <= integrated_config.MAX_SCALED_RESIDUAL
    )
    if not (success and closed):
        print('Integrated solve did not converge to closed balances; results were not saved.')
        return x_final

    integrated_report(states, components)
    results = results_table(states)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results.to_csv(INTEGRATED_DIR / f'0_results_{timestamp}.csv')
    save = input(
        'Do you want to save this as the new main solution? (y/n) '
    ).strip().lower()
    if save == 'y':
        np.save(solution_path, x_final)
        results.to_csv(INTEGRATED_DIR / '0_results.csv')
    return x_final


if __name__ == '__main__':
    main()
