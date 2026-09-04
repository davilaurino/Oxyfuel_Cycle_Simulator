import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import fsolve
from cycle_residuals import allam_residuals
from classes import ColdBox_ASU, Compressor, Turbine, Pump, Intercooler, Condensator
from setup import build_cycle
import utils


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

        elif isinstance(cmp, ColdBox_ASU):
            coldbox = cmp
            W_net += cmp.W
            W_con += cmp.W
            Ic_Q += cmp.Q
            if verbose:
                print(f'[{cmp.Name:<26}] - Work: {cmp.W/1e3:>10.2f} (kW) - Heat: {cmp.Q/1e3:>10.2f} (kW)')

    _, x_O2 = utils.mass_fraction(coldbox.O2_Outlet.y, coldbox.O2_Outlet.spc)
    m_dot_O2_pure = coldbox.O2_Outlet.m_dot*x_O2[0]
    W_MAC = sum(c.W for c in Component.values()
                if (isinstance(c, Compressor) and c.Name.startswith('Air Comp')) or
                   (isinstance(c, Pump) and c.Name == 'LOX Pump'))
    asu_sp_work = -(W_MAC + coldbox.W)/m_dot_O2_pure/1e3

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

        elif state.Stream == 'Air Comp 1 - Inlet':
            MW, x = utils.mass_fraction(state.y, state.spc)
            m_bal_global += state.m_dot
            m_bal_O2 += state.m_dot * x[0]
            m_bal_N2 += state.m_dot * x[1]
            H = state.m_dot * state.h
            H_in += H
            if verbose:
                print(f'[{state.Stream:<26}] - Absolute enthalpy: {H/1e3:>10.2f} (kW)')

        elif state.Stream == 'N2 Vent':
            MW, x = utils.mass_fraction(state.y, state.spc)
            m_bal_global -= state.m_dot
            m_bal_O2 -= state.m_dot * x[0]
            m_bal_N2 -= state.m_dot * x[1]
            H = state.m_dot * state.h
            H_out += H
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
    eta = W_net / Energy_LHV

    if verbose:
        print('Global Mass Balance:', f"{m_bal_global:.2e}", '(kg/s)')
        print('CO2 Mass Balance:', f"{m_bal_CO2:.2e}", '(kg/s)')
        print('H2O Mass Balance:', f"{m_bal_H2O:.2e}", '(kg/s)')
        print('O2 Mass Balance:', f"{m_bal_O2:.2e}", '(kg/s)')
        print('N2 Mass Balance:', f"{m_bal_N2:.2e}", '(kg/s)')
        print('Energy Balance:', f"{(Energy_Bal/1e3):.2e}", '(kW)')
        print('Energy Balance relative error (LHV):', round(Energy_RE * 100, 16), '(%)')
        print('ASU Specific Work:', f"{asu_sp_work:.2f}", '(kJ/kg_O2)')
        print('Fuel Energy (LHV):', f"{Energy_LHV/1e6:.2f}", '(MW)')
        print('Total Work Generated (turbines):', round(W_gen/1e6, 2), '(MW)')
        print('Total Work Consumed (compressors/pumps/ASU):', round(W_con/1e6, 2), '(MW)')
        print('Total Heat Output (intercoolers/condensator/ASU):', round(Ic_Q/1e6, 2), '(MW)')
        print('Net Work:', round(W_net/1e6, 2), '(MW)')
        print('Efficiency (LHV):', round(eta*100, 2), '(%)')
        print(f"Execution time: {elapsed:.2f} seconds")

    return {
        'asu_sp_work_kJ_kg': asu_sp_work,
        'm_bal_global': m_bal_global,
        'energy_bal_rel_err_pct': Energy_RE * 100,
        'W_gen_MW': W_gen / 1e6,
        'W_con_MW': abs(W_con) / 1e6,
        'Q_out_MW': abs(Ic_Q) / 1e6,
        'W_net_MW': W_net / 1e6,
        'eta_pct': eta * 100,
    }


def main():
    prev = input('Do you want to use the previous solution (p) or setup (s)?')
    S, Component, X0_setup, LHV_ng_local = build_cycle()
    X0 = np.load("0_solution.npy") if prev == 'p' else X0_setup

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {state.Stream: state.to_dict() for state in S}
        pd.DataFrame(results).T.to_csv(f"0_results_v11_{timestamp}.csv")
        if input('Do you want to save this as new main solution? (y/n)') == 'y':
            np.save("0_solution.npy", X_final)
            pd.DataFrame(results).T.to_csv("0_results_v11.csv")

    else:
        print("Fsolve did not converge:", mesg)


if __name__ == '__main__':
    main()
