"""Solve the ASU model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASU_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ASU.asu_classes import Intercooler
from ASU.asu_setup import build_asu
from ASU.asu_solver import solve_asu

SOLVER = 'fsolve'  # 'fsolve' or 'least_squares'

def report(states, components):
    """Report ASU duties, work, boundary balances, and performance."""
    print('\nComponent duties and work:')

    work_generated = 0.0
    work_consumed = 0.0
    external_heat = 0.0

    for component in components.values():
        if hasattr(component, 'Q_condenser'):
            print(
                f'[{component.name:<32}] '
                f'Q_condenser={component.Q_condenser/1e3:11.3f} kW  '
                f'Q_reboiler={component.Q_reboiler/1e3:11.3f} kW  '
                f'Q_net_external='
                f'{(component.Q_condenser + component.Q_reboiler)/1e3:9.3f} kW  '
                f'loss target='
                f'{100.0*component.condenser_heat_loss_fraction:5.2f}%'
            )
            # The uncoupled duties are independent external heat interactions.
            external_heat += component.Q_condenser + component.Q_reboiler
        elif hasattr(component, 'Q') and component.Q is not None:
            print(
                f'[{component.name:<32}] '
                f'Q={component.Q/1e3:11.3f} kW'
            )

        if hasattr(component, 'W') and component.W is not None:
            print(
                f'[{component.name:<32}] '
                f'W={component.W/1e3:11.3f} kW'
            )
            if component.W >= 0.0:
                work_generated += component.W
            else:
                work_consumed -= component.W

        # Intercoolers exchange heat with utilities outside the ASU. HX1,
        # HX2, and the condenser-reboiler provide internal heat integration.
        if isinstance(component, Intercooler) and component.Q is not None:
            external_heat += component.Q

    air_feed = states['MAC air intake']
    o2_product = states['Warm O2 product']
    n2_product = states['Warm N2 product']
    products = [o2_product, n2_product]

    mass_balance = air_feed.m_dot - sum(state.m_dot for state in products)
    species_balances = air_feed.m_dot*air_feed.w.copy()
    for state in products:
        species_balances -= state.m_dot*state.w

    enthalpy_in = air_feed.m_dot*air_feed.h
    enthalpy_out = sum(state.m_dot*state.h for state in products)
    net_work = work_generated - work_consumed
    net_power_input = -net_work
    energy_balance = enthalpy_in - enthalpy_out + external_heat - net_work

    o2_index = next(
        i for i, species in enumerate(air_feed.species)
        if species.name == 'O2'
    )
    pure_o2_feed = air_feed.m_dot*air_feed.w[o2_index]
    pure_o2_product = o2_product.m_dot*o2_product.w[o2_index]
    o2_recovery = pure_o2_product/pure_o2_feed
    specific_work = net_power_input/pure_o2_product

    mac_split = states['MAC air to HPC path'].m_dot/states['MAC outlet'].m_dot
    hpc_split = states['HPC reflux'].m_dot/states['Condensed HPC N2'].m_dot
    reflux_ratio = states['HPC reflux'].m_dot/states['HPC N2 to HX2'].m_dot

    print('\nASU balances and performance:')
    print(f'Global mass balance:          {mass_balance: .6e} kg/s')
    for species, balance in zip(air_feed.species, species_balances):
        print(f'{species.name} mass balance:              {balance: .6e} kg/s')
    print(f'Energy balance:               {energy_balance/1e3: .6e} kW')
    if work_consumed > 0.0:
        print(
            'Energy balance / gross work: '
            f'{100.0*energy_balance/work_consumed: .6e} %'
        )
    print(f'External heat transfer:       {external_heat/1e3: .3f} kW')
    print(f'Gross work consumed:          {work_consumed/1e3: .3f} kW')
    print(f'Work generated:               {work_generated/1e3: .3f} kW')
    print(f'Net ASU power input:          {net_power_input/1e3: .3f} kW')
    print(f'ASU specific work:            {specific_work/1e3: .3f} kJ/kg_O2')
    print(f'O2 product flow:              {o2_product.m_dot: .6f} kg/s')
    print(f'Pure O2 component flow:       {pure_o2_product: .6f} kg/s')
    print(f'O2 purity:                    {100.0*o2_product.z[o2_index]: .6f} mol%')
    print(f'O2 recovery:                  {100.0*o2_recovery: .6f} %')
    print(f'N2 product purity:            {100.0*(1.0-n2_product.z[o2_index]): .6f} mol%')
    print(f'MAC split to HPC:             {mac_split: .6f}')
    print(f'HPC condensate reflux split:  {hpc_split: .6f}')
    print(f'HPC reflux ratio (R/D):       {reflux_ratio: .6f}')

    return {
        'mass_balance_kg_s': mass_balance,
        'species_balances_kg_s': species_balances,
        'energy_balance_W': energy_balance,
        'net_power_input_W': net_power_input,
        'specific_work_kJ_kg_O2': specific_work/1e3,
        'o2_purity_mol': o2_product.z[o2_index],
        'o2_recovery': o2_recovery,
        'mac_split_to_hpc': mac_split,
        'hpc_reflux_split': hpc_split,
        'hpc_reflux_ratio': reflux_ratio,
    }

def main():
    previous = input('Do you want to use the previous solution (p) or setup (s)? ')
    states, components, x0 = build_asu()

    if previous == 'p':
        solution_path = ASU_DIR / '0_solution.npy'
        if solution_path.exists():
            x0 = np.load(solution_path)
        else:
            print('No saved solution found; using setup values.')

    x_final, success = solve_asu(states, components, x0, solver=SOLVER)

    for state in states.values():
        vapor_fraction = '-'
        if 0.0 <= state.vapor_fraction <= 1.0:
            vapor_fraction = f'{state.vapor_fraction:.3f}'
        print(
            f'{state.name:<16} '
            f'm_dot={state.m_dot:9.4f} kg/s  '
            f'T={state.T:8.3f} K  '
            f'P={state.P/1e6:7.5f} MPa  '
            f'z={state.z}  '
            f'phase={state.phase:<9}  '
            f'VF={vapor_fraction}'
        )

    report(states, components)

    if success:
        save = input('Do you want to save this as the new main solution? (y/n) ')
        if save == 'y':
            np.save(ASU_DIR / '0_solution.npy', x_final)

    return x_final

if __name__ == '__main__':
    main()
