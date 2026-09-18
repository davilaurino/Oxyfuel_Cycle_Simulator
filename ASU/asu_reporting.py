"""Reporting and CSV output for ASU solutions."""

import numpy as np
import pandas as pd


def print_states(states):
    """Print the solved state table to the console."""
    for state in states.values():
        vapor_fraction = '-'
        if 0.0 <= state.vapor_fraction <= 1.0:
            vapor_fraction = f'{state.vapor_fraction:.3f}'

        print(
            f'{state.name:<32} '
            f'm_dot={state.m_dot:9.4f} kg/s  '
            f'T={state.T:8.3f} K  '
            f'P={state.P/1e6:7.5f} MPa  '
            f'z={state.z}  '
            f'phase={state.phase:<9}  '
            f'VF={vapor_fraction}'
        )


def asu_report(components, feeds, products, o2_product, n2_product):
    """Report calculations shared by all ASU configurations."""
    print('\nComponent duties and work:')

    work_generated = 0.0
    work_consumed = 0.0
    external_heat = 0.0

    for component in components.values():
        if hasattr(component, 'Q_condenser'):
            print(
                f'[{component.name:<32}] '
                f'Q_condenser={component.Q_condenser/1e3:11.3f} kW  '
                f'Q_reboiler={component.Q_reboiler/1e3:11.3f} kW'
            )
        elif hasattr(component, 'Q_ext') and component.Q_ext is not None:
            print(
                f'[{component.name:<32}] '
                f'Q_ext={component.Q_ext/1e3:11.3f} kW'
            )
        elif hasattr(component, 'Q') and component.Q is not None:
            print(f'[{component.name:<32}] Q={component.Q/1e3:11.3f} kW')

        if hasattr(component, 'Q_ext') and component.Q_ext is not None:
            external_heat += component.Q_ext

        if hasattr(component, 'W') and component.W is not None:
            print(f'[{component.name:<32}] W={component.W/1e3:11.3f} kW')
            if component.W >= 0.0:
                work_generated += component.W
            else:
                work_consumed -= component.W

    reference_state = feeds[0]
    species_balances = np.zeros(len(reference_state.species))
    mass_in = 0.0
    mass_out = 0.0
    enthalpy_in = 0.0
    enthalpy_out = 0.0

    for state in feeds:
        mass_in += state.m_dot
        species_balances += state.m_dot*state.w
        enthalpy_in += state.m_dot*state.h

    for state in products:
        mass_out += state.m_dot
        species_balances -= state.m_dot*state.w
        enthalpy_out += state.m_dot*state.h

    mass_balance = mass_in - mass_out
    net_power_input = work_consumed - work_generated
    energy_balance = (
        enthalpy_in - enthalpy_out + external_heat + net_power_input
    )

    o2_index = next(
        i for i, species in enumerate(reference_state.species)
        if species.name == 'O2'
    )
    n2_index = next(
        i for i, species in enumerate(reference_state.species)
        if species.name == 'N2'
    )

    pure_o2_feed = 0.0
    for state in feeds:
        pure_o2_feed += state.m_dot*state.w[o2_index]

    pure_o2_product = o2_product.m_dot*o2_product.w[o2_index]
    o2_recovery = pure_o2_product/pure_o2_feed
    specific_work = net_power_input/pure_o2_product

    normal_temperature = 273.15
    normal_pressure = 101325.0
    universal_gas_constant = 8.31446261815324
    o2_molar_mass = reference_state.species[o2_index].MW
    pure_o2_molar_flow = pure_o2_product/o2_molar_mass
    normal_o2_volume_flow = (
        pure_o2_molar_flow
        * universal_gas_constant
        * normal_temperature
        / normal_pressure
    )
    specific_work_normal_volume = (
        net_power_input/normal_o2_volume_flow/3.6e6
    )

    print('\nASU balances and performance:')
    print(f'Global mass balance:          {mass_balance: .6e} kg/s')
    for species, balance in zip(reference_state.species, species_balances):
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
    print(
        'ASU specific work:            '
        f'{specific_work_normal_volume: .6f} kWh/Nm3_O2'
    )
    print(f'O2 product flow:              {o2_product.m_dot: .6f} kg/s')
    print(f'O2 product pressure:          {o2_product.P/1e6: .6f} MPa')
    print(f'O2 product temperature:       {o2_product.T: .6f} K')
    print(f'Pure O2 component flow:       {pure_o2_product: .6f} kg/s')
    print(f'O2 purity:                    {100.0*o2_product.z[o2_index]: .6f} mol%')
    print(f'O2 recovery:                  {100.0*o2_recovery: .6f} %')
    print(f'N2 product purity:            {100.0*n2_product.z[n2_index]: .6f} mol%')

    return {
        'mass_balance_kg_s': mass_balance,
        'species_balances_kg_s': species_balances,
        'energy_balance_W': energy_balance,
        'external_heat_W': external_heat,
        'gross_work_consumed_W': work_consumed,
        'work_generated_W': work_generated,
        'net_power_input_W': net_power_input,
        'specific_work_kJ_kg_O2': specific_work/1e3,
        'specific_work_kWh_Nm3_O2': specific_work_normal_volume,
        'o2_purity_mol': o2_product.z[o2_index],
        'n2_purity_mol': n2_product.z[n2_index],
        'o2_recovery': o2_recovery,
    }


def gox_report(states, components):
    """Report GOX-ASU balances, duties, and performance."""
    results = asu_report(
        components,
        feeds=(states['MAC air intake'],),
        products=(states['Warm O2 product'], states['Warm N2 product']),
        o2_product=states['Warm O2 product'],
        n2_product=states['Warm N2 product'],
    )

    mac_split = states['MAC air to HPC path'].m_dot/states['MAC outlet'].m_dot
    hpc_reflux_split = (
        states['HPC reflux'].m_dot/states['Condensed HPC N2'].m_dot
    )
    hpc_reflux_ratio = (
        states['HPC reflux'].m_dot/states['HPC N2 to HX2'].m_dot
    )

    print('\nGOX configuration:')
    print(f'MAC split to HPC:             {mac_split: .6f}')
    print(f'HPC condensate reflux split:  {hpc_reflux_split: .6f}')
    print(f'HPC reflux ratio (R/D):       {hpc_reflux_ratio: .6f}')

    results.update({
        'mac_split_to_hpc': mac_split,
        'hpc_reflux_split': hpc_reflux_split,
        'hpc_reflux_ratio': hpc_reflux_ratio,
    })
    return results


def lox_report(states, components):
    """Report LOX-ASU balances, duties, and performance."""
    results = asu_report(
        components,
        feeds=(states['MAC air intake'],),
        products=(states['O2 product'], states['N2 product']),
        o2_product=states['O2 product'],
        n2_product=states['N2 product'],
    )

    mac_power = -sum(
        component.W
        for name, component in components.items()
        if name.startswith('MAC - Compressor')
    )
    mp_booster_power = -sum(
        component.W
        for name, component in components.items()
        if name.startswith('MP booster - Compressor')
    )
    hp_booster_power = -sum(
        component.W
        for name, component in components.items()
        if name.startswith('HP booster - Compressor')
    )
    lox_pump_power = -components['LOX pump'].W

    mac_outlet = states['MAC outlet']
    mp_booster_outlet = states['MP booster outlet']
    lp_air_fraction = states['LP air to HX1'].m_dot/mac_outlet.m_dot
    boosted_air_fraction = (
        states['Air to next compression train'].m_dot/mac_outlet.m_dot
    )
    mp_air_fraction = (
        states['MP air to HX1'].m_dot/mp_booster_outlet.m_dot
    )
    hp_air_fraction = (
        states['Air to HP compression train'].m_dot/mp_booster_outlet.m_dot
    )
    hpc_reflux_split = (
        states['HPC reflux'].m_dot/states['Condensed HPC N2'].m_dot
    )
    hpc_reflux_ratio = (
        states['HPC reflux'].m_dot/states['HPC N2 to HX2'].m_dot
    )

    print('\nLOX configuration:')
    print(f'MAC outlet pressure:          {mac_outlet.P/1e6: .6f} MPa')
    print(f'MAC split to LP air:          {100.0*lp_air_fraction: .6f} %')
    print(f'MAC split to next compressor: {100.0*boosted_air_fraction: .6f} %')
    print(f'MAC power input:              {mac_power/1e3: .6f} kW')
    print(
        'MP booster outlet pressure:   '
        f'{mp_booster_outlet.P/1e6: .6f} MPa'
    )
    print(f'MP booster split to MP air:   {100.0*mp_air_fraction: .6f} %')
    print(f'MP booster split to HP air:   {100.0*hp_air_fraction: .6f} %')
    print(f'MP booster power input:       {mp_booster_power/1e3: .6f} kW')
    print(
        'HP booster outlet pressure:   '
        f"{states['HP booster outlet'].P/1e6: .6f} MPa"
    )
    print(f'HP booster power input:       {hp_booster_power/1e3: .6f} kW')
    print(f'LOX pump power input:         {lox_pump_power/1e3: .6f} kW')
    print(f'HPC condensate reflux split:  {hpc_reflux_split: .6f}')
    print(f'HPC reflux ratio (R/D):       {hpc_reflux_ratio: .6f}')

    results.update({
        'mac_power_W': mac_power,
        'mac_split_to_lp': lp_air_fraction,
        'mp_booster_power_W': mp_booster_power,
        'mp_booster_split_to_mp': mp_air_fraction,
        'hp_booster_power_W': hp_booster_power,
        'lox_pump_power_W': lox_pump_power,
        'hpc_reflux_split': hpc_reflux_split,
        'hpc_reflux_ratio': hpc_reflux_ratio,
    })
    return results


def results_table(states):
    """Return a tabular representation of all solved ASU states."""
    results = {}

    for state in states.values():
        row = {
            'm_dot (kg/s)': round(state.m_dot, 2),
            'T (K)': round(state.T, 2),
            'P (MPa)': round(state.P/1e6, 2),
            'phase': state.phase,
            'vapor fraction': round(state.vapor_fraction, 4),
            'h (J/kg)': round(state.h, 2),
            's (J/kg K)': round(state.s, 4),
            'molar mass (kg/mol)': round(state.molar_mass, 6),
        }

        for species, mole_fraction, mass_fraction in zip(
            state.species,
            state.z,
            state.w,
        ):
            row[f'z_{species.name}'] = round(mole_fraction, 4)
            row[f'w_{species.name}'] = round(mass_fraction, 4)

        results[state.name] = row

    return pd.DataFrame.from_dict(results, orient='index').rename_axis('State')
