"""Select, solve, report, and save an ALLAM configuration."""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLAM_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ALLAM_v2.allam_reporting import (
    allam_report,
    results_table,
)
from ALLAM_v2 import allam_config
from ALLAM_v2.allam_setup import build_cycle
from common.solver import solve

SOLVER = 'fsolve'  # 'fsolve' or 'least_squares'


def main():
    states, components, x0 = build_cycle()

    previous = input('Do you want to use the previous solution (p) or setup (s)? ').strip().lower()
    solution_path = ALLAM_DIR / '0_solution.npy'

    if previous == 'p':
        if solution_path.exists():
            x0 = np.load(solution_path)
        else:
            print('No saved solution found; using setup values.')

    x_final, success = solve(
        states,
        components,
        x0,
        solver=SOLVER,
        config=allam_config,
    )

    if success:
        allam_report(states, components)

        results = results_table(states)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamped_results_path = (
            ALLAM_DIR / f'0_results_{timestamp}.csv'
        )
        results.to_csv(timestamped_results_path)

        save = input('Do you want to save this as the new main solution? (y/n) ').strip().lower()
        if save == 'y':
            np.save(solution_path, x_final)
            results.to_csv(ALLAM_DIR / '0_results.csv')

    return x_final


if __name__ == '__main__':
    main()
