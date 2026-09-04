import time
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import fsolve
from cycle_residuals import allam_residuals
from setup import build_cycle
from main import report

EXC_O2_RANGE       = [1.01, 1.02]
R_CO2_O2_RANGE     = [9, 10, 11]
TIT_RANGE          = [1350, 1400]
O2_PURITY_RANGE    = [0.97, 0.98, 0.99, 0.995]
M_DOT_FUEL_RANGE   = [6, 6.5, 7, 7.5, 8]

def run_sweep():
    grid = list(itertools.product(EXC_O2_RANGE, R_CO2_O2_RANGE, TIT_RANGE, O2_PURITY_RANGE, M_DOT_FUEL_RANGE))
    print(f"Sweep: {len(grid)} combinations")

    records = []
    best_eta = -np.inf
    best_run = None
    X_prev = None

    for i, (exc_O2, r_CO2_O2, TIT, O2_purity, m_dot_fuel) in enumerate(grid):
        print(f"\n[{i+1}/{len(grid)}] exc_O2={exc_O2}, r_CO2_O2={r_CO2_O2}, TIT={TIT}, O2_purity={O2_purity}, m_dot_fuel={m_dot_fuel}")
        S, Component, X0, LHV_ng = build_cycle(
            exc_O2=exc_O2, r_CO2_O2=r_CO2_O2, TIT=TIT, O2_purity=O2_purity, m_dot_fuel=m_dot_fuel
        )

        X_init = X_prev if X_prev is not None else X0

        start = time.time()
        counter = [0]
        X_final, _, ier, _ = fsolve(
            allam_residuals, X_init,
            args=(S, Component, start, counter),
            xtol=1e-8, full_output=True
        )
        elapsed = time.time() - start

        if ier != 1:
            print("  Did not converge — skipping")
            continue

        X_prev = X_final

        metrics = report(S, Component, LHV_ng, elapsed, verbose=False)
        row = {
            'exc_O2': exc_O2,
            'r_CO2_O2': r_CO2_O2,
            'TIT': TIT,
            'O2_purity': O2_purity,
            'm_dot_fuel': m_dot_fuel,
            **metrics,
            'elapsed_s': elapsed,
        }
        records.append(row)
        print(f"  eta={metrics['eta_pct']:.2f}%  W_net={metrics['W_net_MW']:.2f} MW")

        if metrics['eta_pct'] > best_eta:
            best_eta = metrics['eta_pct']
            best_run = (S, Component, LHV_ng, elapsed, X_final, row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if records:
        df = pd.DataFrame(records)
        df.round(4).to_csv(f"1_sweep_summary_{timestamp}.csv", index=False)
        print(f"\nSweep summary saved to 1_sweep_summary_{timestamp}.csv")

    if best_run is not None:
        S_b, Comp_b, LHV_b, el_b, X_b, params_b = best_run
        print(f"\n=== Best run: {params_b} ===")
        report(S_b, Comp_b, LHV_b, el_b, verbose=True)
        results = {state.Stream: state.to_dict() for state in S_b}
        pd.DataFrame(results).T.to_csv(f"1_sweep_best_{timestamp}.csv")
        print(f"Best run states saved to 1_sweep_best_{timestamp}.csv")


if __name__ == '__main__':
    run_sweep()
