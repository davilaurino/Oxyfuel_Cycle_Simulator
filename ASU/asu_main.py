"""Select, solve, report, and save an ASU configuration."""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASU_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ASU.asu_reporting import (
    gox_report,
    lox_report,
    print_states,
    results_table,
)
from ASU.asu_setup_gox import build_asu as build_gox
from ASU.asu_setup_lox import build_asu as build_lox
from common.solver import solve

SOLVER = 'fsolve'  # 'fsolve' or 'least_squares'

CASES = {
    'lox': (build_lox, lox_report),
    'gox': (build_gox, gox_report),
}


def main():
    case_name = input('Select ASU setup (lox/gox): ').strip().lower()
    if case_name not in CASES:
        raise ValueError(f'Unknown ASU setup: {case_name}')

    build_asu, report = CASES[case_name]
    states, components, x0 = build_asu()

    previous = input(
        'Do you want to use the previous solution (p) or setup (s)? '
    ).strip().lower()
    solution_path = ASU_DIR / f'0_solution_{case_name}.npy'

    if previous == 'p':
        if solution_path.exists():
            x0 = np.load(solution_path)
        else:
            print('No saved solution found; using setup values.')

    x_final, success = solve(states, components, x0, solver=SOLVER)

    if success:
        print_states(states)
        report(states, components)

        results = results_table(states)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_results_path = (
            ASU_DIR / f'0_results_{case_name}_{timestamp}.csv'
        )
        results.to_csv(timestamped_results_path)

        save = input(
            'Do you want to save this as the new main solution? (y/n) '
        ).strip().lower()
        if save == 'y':
            np.save(solution_path, x_final)
            results.to_csv(ASU_DIR / f'0_results_{case_name}.csv')

    return x_final


if __name__ == '__main__':
    main()
