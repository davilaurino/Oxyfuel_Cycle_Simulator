"""ALLAM-style reporting for the complete internal-compression ASU plant."""

from scipy.constants import R

from ALLAM_v2.allam_reporting import component_report, results_table as allam_results_table
from common import utils
from common.classes import Compressor, Pump, Turbine


def results_table(states):
    """Return the ALLAM state table with pressure precision for ASU trays."""
    table = allam_results_table(states)
    table['P (MPa)'] = [round(state.P/1e6, 6) for state in states.values()]
    return table


def integrated_report(states, components):
    """Print actual plant duties, boundary balances, and ASU consumption.

    Component duties must have been refreshed by evaluating the residuals at
    the reported solution. Positive ASU work denotes net electrical input,
    after crediting the MP air turbine. Specific consumption uses the pure
    O2 component, excluding nitrogen and argon in the oxygen product.
    """
    results = component_report(components)
    asu_components = [
        component for component in components.values()
        if getattr(component, 'subsystem', None) == 'ASU'
    ]
    if not asu_components:
        raise ValueError('ASU ownership is missing; use build_integrated_cycle()')
    asu_work_consumed = 0.0
    asu_work_generated = 0.0
    for component in asu_components:
        if isinstance(component, (Compressor, Pump, Turbine)):
            asu_work_consumed += max(-component.W, 0.0)
            asu_work_generated += max(component.W, 0.0)
    asu_work = asu_work_consumed - asu_work_generated

    fuel = states['Fuel Compressor - Inlet']
    air = states['MAC air intake']
    oxygen = states['O2 product']
    feeds = (fuel, air)
    products = (
        states['N2 product'],
        states['Condenser Liquid Outlet'],
        states['Purged CO2'],
    )
    # Oxygen is an internal connection, not a plant boundary stream.
    mass_balance = (
        sum(state.m_dot for state in feeds)
        - sum(state.m_dot for state in products)
    )
    species_balances = dict.fromkeys(
        (utils.CO2, utils.H2O, utils.O2, utils.N2, utils.AR), 0.0
    )
    for streams, sign in ((feeds, 1.0), (products, -1.0)):
        for state in streams:
            for species, mass_fraction in zip(state.species, state.w):
                if species in species_balances:
                    species_balances[species] += sign*state.m_dot*mass_fraction

    _, _, carbon_flow, hydrogen_flow = utils.stoichiometry(fuel)
    species_balances[utils.CO2] += carbon_flow*utils.CO2.MW
    species_balances[utils.H2O] += hydrogen_flow/2.0*utils.H2O.MW
    species_balances[utils.O2] -= (carbon_flow + hydrogen_flow/4.0)*utils.O2.MW

    energy_balance = (
        sum(state.m_dot*state.h for state in feeds)
        - sum(state.m_dot*state.h for state in products)
        + results['external_heat_W'] - results['net_work_W']
    )
    fuel_lhv = sum(
        mass_fraction*species.LHV
        for mass_fraction, species in zip(fuel.w, fuel.species)
    )
    fuel_energy = fuel.m_dot*fuel_lhv
    o2_index = oxygen.species.index(utils.O2)
    air_o2_index = air.species.index(utils.O2)
    pure_o2_flow = oxygen.m_dot*oxygen.w[o2_index]
    specific_work = asu_work/pure_o2_flow
    # Normal cubic metres of pure oxygen: 273.15 K and 101325 Pa.
    normal_o2_volume_flow = pure_o2_flow/utils.O2.MW*R*273.15/101325.0
    specific_work_normal_volume = asu_work/normal_o2_volume_flow/3.6e6
    o2_recovery = pure_o2_flow/(air.m_dot*air.w[air_o2_index])

    results.update({
        'mass_balance_kg_s': mass_balance,
        'species_balances_kg_s': species_balances,
        'energy_balance_W': energy_balance,
        'fuel_energy_W': fuel_energy,
        'asu_gross_work_consumed_W': asu_work_consumed,
        'asu_work_generated_W': asu_work_generated,
        'asu_work_W': asu_work,
        'asu_specific_work_J_kg': specific_work,
        'asu_specific_work_kJ_kg_O2': specific_work/1e3,
        'asu_specific_work_kWh_Nm3_O2': specific_work_normal_volume,
        'pure_o2_flow_kg_s': pure_o2_flow,
        'o2_recovery': o2_recovery,
        'total_work_consumed_W': results['gross_work_consumed_W'],
        'net_efficiency': results['net_work_W']/fuel_energy,
    })

    print('\nBoundary-stream enthalpies:')
    for state in (*feeds, *products):
        print(
            f'[{state.name:<40}] - Absolute enthalpy: '
            f'{state.m_dot*state.h/1e3:>10.2f} (kW)'
        )

    print('\nIntegrated balances and performance:')
    print(f'Global Mass Balance: {mass_balance:.2e} (kg/s)')
    for species, balance in species_balances.items():
        print(f'{species.name} Mass Balance: {balance:.2e} (kg/s)')
    print(f'Energy Balance: {energy_balance/1e3:.2e} (kW)')
    print(
        'Energy Balance relative error (LHV): '
        f'{energy_balance/fuel_energy*100:.6e} (%)'
    )
    print(f'ASU Gross Work Consumed: {asu_work_consumed/1e6:.3f} (MW)')
    print(f'ASU Work Generated (air turbine): {asu_work_generated/1e6:.3f} (MW)')
    print(f'ASU Total Work (net input): {asu_work/1e6:.3f} (MW)')
    print(f'ASU Specific Consumption: {specific_work/1e3:.3f} (kJ/kg_O2)')
    print(
        'ASU Specific Consumption: '
        f'{specific_work_normal_volume:.6f} (kWh/Nm3_O2)'
    )
    print('Normal O2 volume reference: 273.15 K, 101325 Pa')
    print(f'MAC Intake Mass Flow: {air.m_dot:.6f} (kg/s)')
    print(f'O2 Product Mass Flow: {oxygen.m_dot:.6f} (kg/s)')
    print(f'Pure O2 Component Flow: {pure_o2_flow:.6f} (kg/s)')
    print(f'O2 Product Purity: {oxygen.z[o2_index]*100:.6f} (mol%)')
    print(f'O2 Recovery: {o2_recovery*100:.6f} (%)')
    print(f'Fuel Energy (LHV): {fuel_energy/1e6:.2f} (MW)')
    print(
        'Total Work Generated (turbines): '
        f'{results["work_generated_W"]/1e6:.2f} (MW)'
    )
    print(
        'Total Work Consumed (compressors/pumps): '
        f'{results["total_work_consumed_W"]/1e6:.2f} (MW)'
    )
    print(
        'Total Heat Output (intercoolers/condenser): '
        f'{abs(results["external_heat_W"])/1e6:.2f} (MW)'
    )
    print(f'Net Work: {results["net_work_W"]/1e6:.2f} (MW)')
    print(f'Efficiency (LHV): {results["net_efficiency"]*100:.2f} (%)')
    return results
