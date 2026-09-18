"""Reporting and CSV output for the ALLAM cycle."""

import pandas as pd

from common.classes import Compressor, Intercooler, Pump, Turbine

COMPONENT_NAME_WIDTH = 40


def component_report(components):
    """Print component work and heat duties and return their totals."""
    work_generated = 0.0
    work_consumed = 0.0
    external_heat = 0.0

    print('\nComponent duties and work:')
    for component in components.values():
        if isinstance(component, (Compressor, Turbine, Pump)):
            print(
                f'[{component.name:<{COMPONENT_NAME_WIDTH}}] - Work: '
                f'{component.W/1e3:>10.2f} (kW)'
            )

            if component.W >= 0.0:
                work_generated += component.W
            else:
                work_consumed -= component.W

        elif isinstance(component, Intercooler):
            print(
                f'[{component.name:<{COMPONENT_NAME_WIDTH}}] - Heat: '
                f'{component.Q_ext/1e3:>10.2f} (kW)'
            )
            external_heat += component.Q_ext

        elif hasattr(component, 'Q_condenser'):
            print(
                f'[{component.name:<{COMPONENT_NAME_WIDTH}}] - Heat: '
                f'Condenser={component.Q_condenser/1e3:>10.2f} (kW) - '
                f'Reboiler={component.Q_reboiler/1e3:>10.2f} (kW)'
            )

        elif getattr(component, 'Q', None) is not None:
            print(
                f'[{component.name:<{COMPONENT_NAME_WIDTH}}] - Heat: '
                f'{component.Q/1e3:>10.2f} (kW)'
            )

    print(f'Gross work consumed: {work_consumed/1e6: .3f} MW')
    print(f'Work generated:      {work_generated/1e6: .3f} MW')
    print(f'External heat:       {external_heat/1e6: .3f} MW')

    return {
        'gross_work_consumed_W': work_consumed,
        'work_generated_W': work_generated,
        'external_heat_W': external_heat,
        'net_work_W': work_generated - work_consumed,
    }


def allam_report(states, components, feeds=None, products=None,
                 fuel_lhv=None):
    """Report current ALLAM duties and optional cycle-level balances.

    ``feeds`` and ``products`` remain optional while the cycle is assembled.
    Once the complete flowsheet exists, passing them enables global mass and
    energy balances without changing the reporting interface.
    """
    results = component_report(components)

    if feeds is not None and products is not None:
        reference_state = feeds[0]
        species_balances = {
            species: 0.0 for species in reference_state.species
        }
        mass_in = sum(state.m_dot for state in feeds)
        mass_out = sum(state.m_dot for state in products)
        enthalpy_in = sum(state.m_dot*state.h for state in feeds)
        enthalpy_out = sum(state.m_dot*state.h for state in products)

        for state in feeds:
            for species, mass_fraction in zip(state.species, state.w):
                species_balances[species] += state.m_dot*mass_fraction
        for state in products:
            for species, mass_fraction in zip(state.species, state.w):
                species_balances[species] -= state.m_dot*mass_fraction

        net_power_input = (
            results['gross_work_consumed_W']
            - results['work_generated_W']
        )
        energy_balance = (
            enthalpy_in
            - enthalpy_out
            + results['external_heat_W']
            + net_power_input
        )

        print('\nALLAM balances:')
        print(f'Global mass balance: {mass_in - mass_out: .6e} kg/s')
        for species, balance in species_balances.items():
            print(f'{species.name} mass balance: {balance: .6e} kg/s')
        print(f'Energy balance: {energy_balance/1e6: .6e} MW')

        results.update({
            'mass_balance_kg_s': mass_in - mass_out,
            'species_balances_kg_s': species_balances,
            'energy_balance_W': energy_balance,
        })

        if fuel_lhv is not None:
            fuel_energy = feeds[0].m_dot*fuel_lhv
            results['fuel_energy_W'] = fuel_energy
            results['net_efficiency'] = (
                -net_power_input/fuel_energy
            )

    return results


def results_table(states):
    """Return a tabular representation of all process states."""
    results = {}

    for state in states.values():
        row = {
            'm_dot (kg/s)': round(state.m_dot, 6),
            'T (K)': round(state.T, 6),
            'P (MPa)': round(state.P/1e6, 6),
            'phase': state.phase,
            'vapor fraction': round(state.vapor_fraction, 6),
            'h (J/kg)': round(state.h, 6),
            's (J/kg K)': round(state.s, 6),
            'molar mass (kg/mol)': round(state.molar_mass, 9),
        }

        for species, mole_fraction in zip(state.species, state.z):
            row[f'z_{species.name}'] = round(mole_fraction, 8)

        results[state.name] = row

    return (
        pd.DataFrame.from_dict(results, orient='index')
        .fillna(0.0)
        .rename_axis('State')
    )
