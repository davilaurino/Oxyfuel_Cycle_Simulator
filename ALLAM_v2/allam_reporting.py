"""Reporting and CSV output for the ALLAM cycle."""

import pandas as pd

from common import utils
from common.classes import Compressor, Condenser, Intercooler, Pump, Turbine

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

        elif isinstance(component, (Intercooler, Condenser)):
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
        reference_state = products[0]
        species_balances = {
            species: 0.0 for species in reference_state.species
        }
        mass_in = sum(state.m_dot for state in feeds)
        mass_out = sum(state.m_dot for state in products)
        enthalpy_in = sum(state.m_dot*state.h for state in feeds)
        enthalpy_out = sum(state.m_dot*state.h for state in products)

        for state in feeds:
            for species, mass_fraction in zip(state.species, state.w):
                if species in species_balances:
                    species_balances[species] += state.m_dot*mass_fraction
        for state in products:
            for species, mass_fraction in zip(state.species, state.w):
                species_balances[species] -= state.m_dot*mass_fraction

        _, _, carbon_flow, hydrogen_flow = utils.stoichiometry(feeds[0])
        species_balances[utils.CO2] += carbon_flow*utils.CO2.MW
        species_balances[utils.H2O] += hydrogen_flow/2.0*utils.H2O.MW
        species_balances[utils.O2] -= (
            carbon_flow + hydrogen_flow/4.0
        )*utils.O2.MW

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

        print('\nBoundary-stream enthalpies:')
        for state in (*feeds, *products):
            print(
                f'[{state.name:<{COMPONENT_NAME_WIDTH}}] - '
                f'Absolute enthalpy: {state.m_dot*state.h/1e3:>10.2f} (kW)'
            )

        print('\nALLAM balances and performance:')
        print(f'Global Mass Balance: {mass_in - mass_out:.2e} (kg/s)')
        for species in reference_state.species:
            print(
                f'{species.name} Mass Balance: '
                f'{species_balances[species]:.2e} (kg/s)'
            )
        print(f'Energy Balance: {energy_balance/1e3:.2e} (kW)')

        results.update({
            'mass_balance_kg_s': mass_in - mass_out,
            'species_balances_kg_s': species_balances,
            'energy_balance_W': energy_balance,
        })

        if fuel_lhv is not None:
            fuel_energy = feeds[0].m_dot*fuel_lhv
            energy_relative_error = energy_balance/fuel_energy
            oxygen_feed = next(
                state for state in feeds
                if utils.O2 in state.species
                and all(species.LHV == 0.0 for species in state.species)
            )
            oxygen_index = oxygen_feed.species.index(utils.O2)
            asu_specific_work = utils.oxygen_separation_work(
                oxygen_feed.z[oxygen_index]
            )
            pure_oxygen_flow = (
                oxygen_feed.m_dot*oxygen_feed.w[oxygen_index]
            )
            asu_work = pure_oxygen_flow*asu_specific_work
            total_work_consumed = (
                results['gross_work_consumed_W'] + asu_work
            )
            net_work = (
                results['work_generated_W'] - total_work_consumed
            )

            results['fuel_energy_W'] = fuel_energy
            results['asu_specific_work_J_kg'] = asu_specific_work
            results['asu_work_W'] = asu_work
            results['total_work_consumed_W'] = total_work_consumed
            results['net_work_W'] = net_work
            results['net_efficiency'] = net_work/fuel_energy

            print(
                'Energy Balance relative error (LHV): '
                f'{energy_relative_error*100:.6e} (%)'
            )
            print(
                f'ASU Specific Work: {asu_specific_work/1e3:.2f} '
                '(kJ/kg_O2)'
            )
            print(f'ASU Work: {-asu_work/1e6:.2f} (MW)')
            print(f'Fuel Energy (LHV): {fuel_energy/1e6:.2f} (MW)')
            print(
                'Total Work Generated (turbines): '
                f'{results["work_generated_W"]/1e6:.2f} (MW)'
            )
            print(
                'Total Work Consumed (compressors/pumps/ASU): '
                f'{total_work_consumed/1e6:.2f} (MW)'
            )
            print(
                'Total Heat Output (intercoolers/condenser): '
                f'{abs(results["external_heat_W"])/1e6:.2f} (MW)'
            )
            print(f'Net Work: {net_work/1e6:.2f} (MW)')
            print(
                f'Efficiency (LHV): '
                f'{results["net_efficiency"]*100:.2f} (%)'
            )

    return results


def results_table(states):
    """Return a tabular representation of all process states."""
    results = {}

    for state in states.values():
        row = {
            'm_dot (kg/s)': round(state.m_dot, 2),
            'T (K)': round(state.T, 2),
            'P (MPa)': round(state.P/1e6, 1),
            'phase': state.phase,
            'vapor fraction': round(state.vapor_fraction, 4),
            'h (J/kg)': round(state.h, 1),
            's (J/kg K)': round(state.s, 1),
            'molar mass (kg/mol)': round(state.molar_mass, 4),
        }

        for species, mole_fraction in zip(state.species, state.z):
            row[f'z_{species.name}'] = round(mole_fraction, 6)

        results[state.name] = row

    return (
        pd.DataFrame.from_dict(results, orient='index')
        .fillna(0.0)
        .rename_axis('State')
    )
