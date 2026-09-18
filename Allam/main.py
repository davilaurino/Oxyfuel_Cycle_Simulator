import time
import sys
from pathlib import Path

# Allow this file to be launched directly while shared modules remain at the
# project root (for example: ``python Allam/main.py``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLAM_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import fsolve
from Allam.cycle_residuals import allam_residuals
from classes import Compressor, Turbine, Pump, Intercooler, Condensator
from Allam.setup import build_cycle
from common import utils


def report_heos_diagnostics(S):
    h_errors = np.asarray([state.h - state.h_heos for state in S])
    s_errors = np.asarray([state.s - state.s_heos for state in S])
    h_max_index = np.argmax(np.abs(h_errors))
    s_max_index = np.argmax(np.abs(s_errors))

    print('HEOS validation (model - phase-imposed HEOS):')
    print(
        f'  Enthalpy: bias = {np.mean(h_errors):.2f} J/kg, '
        f'RMSE = {np.sqrt(np.mean(h_errors**2)):.2f} J/kg, '
        f'max = {h_errors[h_max_index]:.2f} J/kg '
        f'[{S[h_max_index].Stream}]'
    )
    print(
        f'  Entropy:  bias = {np.mean(s_errors):.4f} J/(kg K), '
        f'RMSE = {np.sqrt(np.mean(s_errors**2)):.4f} J/(kg K), '
        f'max = {s_errors[s_max_index]:.4f} J/(kg K) '
        f'[{S[s_max_index].Stream}]'
    )


def report(S, Component, LHV_ng, elapsed, verbose=True):
    W_net = 0
    W_gen = 0
    W_con = 0
    Ic_Q = 0

    for cmp in Component.values():
        if isinstance(cmp, Turbine):
            W_net += cmp.W
            W_gen += cmp.W
            if verbose:
                print(f'[{cmp.Name:<26}] - Work: {cmp.W/1e3:>10.2f} (kW) - Iso Temp: {cmp.T_iso:>10.2f} (K)')

        elif isinstance(cmp, (Compressor, Pump)):
            W_net += cmp.W
            W_con += cmp.W
            if verbose:
                print(f'[{cmp.Name:<26}] - Work: {cmp.W/1e3:>10.2f} (kW) - Iso Temp: {cmp.T_iso:>10.2f} (K)')

        elif isinstance(cmp, (Intercooler, Condensator)):
            Ic_Q += cmp.Q
            if verbose:
                print(f'[{cmp.Name:<26}] - Heat: {cmp.Q/1e3:>10.2f} (kW)')

    O2_input = next(state for state in S if state.Stream == 'O2 Input')
    _, x_O2_input = utils.mass_fraction(O2_input.y, O2_input.spc)
    i_O2 = O2_input.spc.index(utils.SPS['O2'])
    m_dot_O2_pure = O2_input.m_dot*x_O2_input[i_O2]
    asu_sp_work = utils.oxygen_separation_work(O2_input.y[i_O2])
    W_asu = -m_dot_O2_pure*asu_sp_work

    W_con_total = W_con + W_asu
    W_net_total = W_net + W_asu

    m_bal_global = 0
    m_bal_CO2, m_bal_H2O, inv_m_bal_O2 = utils.fuel_requirements(S[0])
    m_bal_O2 = -inv_m_bal_O2
    m_bal_N2 = 0
    H_in = 0
    H_out = 0
    Energy_LHV = 0

    for state in S:
        if state.Stream == 'Fuel Compressor 1 - Inlet':
            m_bal_global += state.m_dot
            H = state.m_dot * state.h
            H_in += H
            Energy_LHV = state.m_dot*LHV_ng
            if verbose:
                print(f'[{state.Stream:<26}] - Absolute enthalpy: {H/1e3:>10.2f} (kW)')

        elif state.Stream == 'O2 Input':
            MW, x = utils.mass_fraction(state.y, state.spc)
            m_bal_global += state.m_dot
            m_bal_O2 += state.m_dot * x[0]
            m_bal_N2 += state.m_dot * x[1]
            H = state.m_dot * state.h
            H_in += H
            if verbose:
                print(f'[{state.Stream:<26}] - Absolute enthalpy: {H/1e3:>10.2f} (kW)')

        elif state.Stream == 'H2O Out':
            m_bal_global -= state.m_dot
            m_bal_H2O -= state.m_dot
            H = state.m_dot * state.h
            H_out += H
            if verbose:
                print(f'[{state.Stream:<26}] - Absolute enthalpy: {H/1e3:>10.2f} (kW)')

        elif state.Stream == 'CO2 Out':
            MW, x = utils.mass_fraction(state.y, state.spc)
            m_bal_global -= state.m_dot
            m_bal_CO2 -= state.m_dot * x[0]
            m_bal_H2O -= state.m_dot * x[1]
            m_bal_O2 -= state.m_dot * x[2]
            m_bal_N2 -= state.m_dot * x[3]
            H = state.m_dot * state.h
            H_out += H
            if verbose:
                print(f'[{state.Stream:<26}] - Absolute enthalpy: {H/1e3:>10.2f} (kW)')

    Energy_Bal = H_in - H_out + Ic_Q - W_net
    Energy_RE = Energy_Bal / Energy_LHV
    eta = W_net_total / Energy_LHV

    if verbose:
        print('Global Mass Balance:', f"{m_bal_global:.2e}", '(kg/s)')
        print('CO2 Mass Balance:', f"{m_bal_CO2:.2e}", '(kg/s)')
        print('H2O Mass Balance:', f"{m_bal_H2O:.2e}", '(kg/s)')
        print('O2 Mass Balance:', f"{m_bal_O2:.2e}", '(kg/s)')
        print('N2 Mass Balance:', f"{m_bal_N2:.2e}", '(kg/s)')
        print('Energy Balance:', f"{(Energy_Bal/1e3):.2e}", '(kW)')
        print('Energy Balance relative error (LHV):', round(Energy_RE * 100, 16), '(%)')
        print('ASU Specific Work:', f"{asu_sp_work/1e3:.2f}", '(kJ/kg_O2)')
        print('ASU Work:', f"{W_asu/1e6:.2f}", '(MW)')
        print('Fuel Energy (LHV):', f"{Energy_LHV/1e6:.2f}", '(MW)')
        print('Total Work Generated (turbines):', round(W_gen/1e6, 2), '(MW)')
        print('Total Work Consumed (compressors/pumps/ASU):', round(W_con_total/1e6, 2), '(MW)')
        print('Total Heat Output (intercoolers/condensator):', round(Ic_Q/1e6, 2), '(MW)')
        print('Net Work:', round(W_net_total/1e6, 2), '(MW)')
        print('Efficiency (LHV):', round(eta*100, 2), '(%)')
        print(f"Execution time: {elapsed:.2f} seconds")

    return {
        'asu_sp_work_kJ_kg': asu_sp_work / 1e3,
        'W_asu_MW': W_asu / 1e6,
        'm_bal_global': m_bal_global,
        'energy_bal_rel_err_pct': Energy_RE * 100,
        'W_gen_MW': W_gen / 1e6,
        'W_con_MW': abs(W_con_total) / 1e6,
        'Q_out_MW': abs(Ic_Q) / 1e6,
        'W_net_MW': W_net_total / 1e6,
        'eta_pct': eta * 100,
    }


def main():
    prev = input('Do you want to use the previous solution (p) or setup (s)?')
    S, Component, X0_setup, LHV_ng_local = build_cycle()
    X0 = X0_setup
    if prev == 'p':
        X0_saved = np.load(ALLAM_DIR / "0_solution.npy")
        if len(X0_saved) == len(X0_setup):
            X0 = X0_saved
        else:
            print('Saved solution is incompatible with the current cycle; using setup values.')

    start = time.time()
    counter = [0]
    X_final, _, ier, mesg = fsolve(
        allam_residuals, X0,
        args=(S, Component, start, counter),
        xtol=1e-8, full_output=True
    )
    elapsed = time.time() - start

    if ier == 1:
        report(S, Component, LHV_ng_local, elapsed, verbose=True)

        print('Running post-convergence phase and HEOS diagnostics...')
        results = {state.Stream: state.to_dict() for state in S}
        report_heos_diagnostics(S)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(results).T.to_csv(
            ALLAM_DIR / f"0_results_v11_{timestamp}.csv"
        )
        if input('Do you want to save this as new main solution? (y/n)') == 'y':
            np.save(ALLAM_DIR / "0_solution.npy", X_final)
            pd.DataFrame(results).T.to_csv(ALLAM_DIR / "0_results_v11.csv")

    else:
        print("Fsolve did not converge:", mesg)


if __name__ == '__main__':
    main()
