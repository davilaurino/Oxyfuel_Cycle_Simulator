"""Build the gas-O2 (GOX) cryogenic ASU and air-compression train."""

import numpy as np
from common.builders import build_column, build_compressor_train
from common.classes import (
    Input,
    Mod_MSHX,
    O2Specification,
    ReboilerCondenser,
    Splitter,
    State,
    Turbine,
    Valve,
)


def build_asu():
    species = ['O2', 'N2', 'AR']
    states = {}
    components = {}

    # Main air compressor intake
    mac_air_in = State(
        name = 'MAC air intake',
        species = species,
        m_dot = 10.0,             # kg/s
        T = 298.15,               # K
        P = 101325.0,             # Pa
        z = [0.209390, 0.780848, 0.009762], # mole fractions [O2, N2, AR]
        phase = 'vapor',
    )
    states[mac_air_in.name] = mac_air_in

    mac_air_input = Input(
        name = 'MAC air intake - Input',
        T = mac_air_in.T,         # K
        P = mac_air_in.P,         # Pa
        z = mac_air_in.z,         # mole fractions [O2, N2, AR]
        outlet = mac_air_in,
        m_dot = mac_air_in.m_dot, # kg/s
    )
    components[mac_air_input.name] = mac_air_input

    mac_states, mac_components = build_compressor_train(
        name = 'MAC',
        inlet = mac_air_in,
        P_end = 6.00e5,           # Pa
        cooling_temperature = 300.0, # K
        n_stages = 2,
        efficiency = 0.87,
        pressure_drop_percent = 0.1, # % per cooler
    )
    states.update(mac_states)
    components.update(mac_components)
    mac_air = mac_states['MAC outlet']

    # Main air split between the HPC and LPC paths
    mac_air_to_hpc = State(
        name = 'MAC air to HPC path',
        species = species,
        m_dot = 8.0,              # kg/s
        T = 300.0,                # K
        P = 6.00e5,               # Pa
        z = [0.209, 0.7909, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'vapor',
    )
    states[mac_air_to_hpc.name] = mac_air_to_hpc

    mac_air_to_lpc = State(
        name = 'MAC air to LPC path',
        species = species,
        m_dot = 2.0,              # kg/s
        T = 300.0,                # K
        P = 6.00e5,               # Pa
        z = [0.209, 0.7909, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'vapor',
    )
    states[mac_air_to_lpc.name] = mac_air_to_lpc

    mac_air_splitter = Splitter(
        name = 'MAC air splitter',
        inlet = mac_air,
        outlets = [mac_air_to_hpc, mac_air_to_lpc],
    )
    components[mac_air_splitter.name] = mac_air_splitter

    # LPC-path booster compressor
    lpc_booster_states, lpc_booster_components = build_compressor_train(
        name = 'LPC booster',
        inlet = mac_air_to_lpc,
        P_end = 7.5e5,           # Pa
        cooling_temperature = 300.0, # K
        n_stages = 1,
        efficiency = 0.87,
        pressure_drop_percent = 0.1, # % per cooler
    )
    states.update(lpc_booster_states)
    components.update(lpc_booster_components)
    lpc_air = lpc_booster_states['LPC booster outlet']

    # Cooled air entering the bottom of the HPC
    cooled_hpc_air = State(
        name = 'Cooled HPC air',
        species = species,
        m_dot = 8.0,              # kg/s
        T = 100.0369,             # K, saved-solution initial guess
        P = 6.00e5,               # Pa
        z = [0.209, 0.7909, 0.0001], # mole fractions [O2, N2, AR]
    )
    states[cooled_hpc_air.name] = cooled_hpc_air

    # HPC reflux
    hpc_reflux = State(
        name = 'HPC reflux',
        species = species,
        m_dot = 3.8368,           # kg/s, saved-solution initial guess
        T = 96.0,                 # K
        P = 5.90e5,               # Pa
        z = [0.0006662, 0.9992338, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'liquid',
    )
    states[hpc_reflux.name] = hpc_reflux

    # HPC tray streams
    hpc_states, hpc_components = build_column(
        name = 'HPC',
        species = species,
        n_trays = 15,
        P_top = 5.90e5,             # Pa
        P_bottom = 6.00e5,          # Pa
        L_top = 4.043,                # kg/s, saved-solution initial guess
        L_bottom = 5.4421,            # kg/s
        V_top = 6.395,                # kg/s
        V_bottom = 6.398,             # kg/s
        x_top = [0.00163, 0.99827, 0.0001], # mole fractions [O2, N2, AR]
        x_bottom = [0.3112455, 0.6886545, 0.0001], # mole fractions [O2, N2, AR]
        feeds = {
            1: [hpc_reflux],
            15: [cooled_hpc_air],
        },
    )
    states.update(hpc_states)
    components.update(hpc_components)

    # Condensed HPC nitrogen
    condensed_hpc_n2 = State(
        name = 'Condensed HPC N2',
        species = species,
        m_dot = 6.3947,           # kg/s, saved-solution initial guess
        T = 92.0,                 # K
        P = 5.90e5,               # Pa
        z = [0.0006662, 0.9992338, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'liquid',
    )
    states[condensed_hpc_n2.name] = condensed_hpc_n2

    # HPC nitrogen to HX2
    hpc_n2_to_hx2 = State(
        name = 'HPC N2 to HX2',
        species = species,
        m_dot = 2.5579,           # kg/s, saved-solution initial guess
        T = 96.0,                 # K
        P = 5.90e5,               # Pa
        z = [0.0006662, 0.9992338, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'liquid',
    )
    states[hpc_n2_to_hx2.name] = hpc_n2_to_hx2

    hpc_cond_splitter = Splitter(
        name = 'HPC condenser splitter',
        inlet = condensed_hpc_n2,
        outlets = [hpc_reflux, hpc_n2_to_hx2],
        split_fractions = [0.52, 0.48],
    )
    components[hpc_cond_splitter.name] = hpc_cond_splitter

    # LPC nitrogen feed after the HPC N2 valve
    lpc_n2_feed = State(
        name = 'LPC N2 feed',
        species = species,
        m_dot = 2.5579,           # kg/s, saved-solution initial guess
        T = 78.8241,              # K, flashed valve-outlet initial guess
        P = 1.20e5,               # Pa
        z = [0.0006662, 0.9992338, 0.0001], # mole fractions [O2, N2, AR]
    )
    states[lpc_n2_feed.name] = lpc_n2_feed

    # Turbine outlet feeding LPC tray 16. The state is registered later to keep
    # the main-heat-exchanger stream ordering together.
    cooled_lpc_air = State(
        name = 'Cooled LPC air',
        species = species,
        m_dot = lpc_air.m_dot, # kg/s
        T = 83.2275,              # K, last-converged solution initial guess
        P = 1.30e5,               # Pa
        z = lpc_air.z,
    )

    # HPC oxygen-enriched liquid after its valve / LPC tray-5 feed
    lpc_o2_feed = State(
        name = 'LPC enriched O2 feed',
        species = species,
        m_dot = 5.4421,           # kg/s, saved-solution initial guess
        T = 82.1326,              # K
        P = 1.28e5,               # Pa
        z = [0.3112455, 0.6886545, 0.0001], # mole fractions [O2, N2, AR]
    )
    states[lpc_o2_feed.name] = lpc_o2_feed

    # LPC oxygen product
    lpc_o2_product = State(
        name = 'LPC O2 product',
        species = species,
        m_dot = 1.9779,           # kg/s, saved-solution initial guess
        T = 92.6406,              # K
        P = 1.30e5,               # Pa
        z = [0.99, 0.0099, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'liquid',
    )
    states[lpc_o2_product.name] = lpc_o2_product

    o2_specification = O2Specification(
        name = 'O2 Specification',
        o2_product = lpc_o2_product,
        purity_target = 0.98,
    )
    components[o2_specification.name] = o2_specification

    # LPC vapor returned from the reboiler
    lpc_boilup = State(
        name = 'LPC boilup',
        species = species,
        m_dot = 4.8430,           # kg/s, saved-solution initial guess
        T = 92.6406,              # K
        P = 1.30e5,               # Pa
        z = [0.9998969, 0.0000031, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'vapor',
    )
    states[lpc_boilup.name] = lpc_boilup

    # LPC tray streams
    lpc_states, lpc_components = build_column(
        name = 'LPC',
        species = species,
        n_trays = 20,
        P_top = 1.20e5,           # Pa
        P_bottom = 1.30e5,        # Pa
        L_top = 2.420,             # kg/s, saved-solution initial guess
        L_bottom = 6.821,          # kg/s
        V_top = 8.022,             # kg/s
        V_bottom = 4.841,           # kg/s
        x_top = [0.12857, 0.87133, 0.0001], # mole fractions [O2, N2, AR]
        x_bottom = [0.9998976, 0.0000024, 0.0001], # mole fractions [O2, N2, AR]
        feeds = {
            1: [lpc_n2_feed],
            8: [lpc_o2_feed],
            12: [cooled_lpc_air],
            20: [lpc_boilup],
        },
    )

    reboiler_condenser = ReboilerCondenser(
        name = 'HPC condenser - LPC reboiler',
        condenser_vapor_in = hpc_states['HPC T1 - vapor'],
        condenser_liquid_out = condensed_hpc_n2,
        reboiler_liquid_in = lpc_states['LPC T20 - liquid'],
        reboiler_liquid_out = lpc_o2_product,
        reboiler_vapor_out = lpc_boilup,
        T_cold_pinch = 3.5,        # K
    )
    components[reboiler_condenser.name] = reboiler_condenser
    states.update(lpc_states)
    components.update(lpc_components)

    # HPC nitrogen after HX2
    cooled_hpc_n2 = State(
        name = 'Cooled HPC N2',
        species = species,
        m_dot = 2.5579,           # kg/s, saved-solution initial guess
        T = 81.7486,              # K
        P = 5.90e5,               # Pa
        z = [0.0006662, 0.9992338, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'liquid',
    )
    states[cooled_hpc_n2.name] = cooled_hpc_n2

    # HPC oxygen-enriched liquid after HX2
    cooled_hpc_o2 = State(
        name = 'Cooled HPC enriched O2',
        species = species,
        m_dot = 5.4421,           # kg/s, saved-solution initial guess
        T = 96.4484,              # K
        P = 6.00e5,               # Pa
        z = [0.3112455, 0.6886545, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'liquid',
    )
    states[cooled_hpc_o2.name] = cooled_hpc_o2

    # LPC nitrogen after HX2
    warmed_lpc_n2 = State(
        name = 'Warmed LPC N2',
        species = species,
        m_dot = 8.0221,           # kg/s, saved-solution initial guess
        T = 90.0,                 # K
        P = 1.20e5,               # Pa
        z = [0.0373603, 0.9625397, 0.0001], # mole fractions [O2, N2, AR]
        phase = 'vapor',
    )
    states[warmed_lpc_n2.name] = warmed_lpc_n2

    # Pump-free baseline: the LPC liquid oxygen product is sent directly to
    # HX1. Any downstream external O2 compression is outside this ASU model.
    hx2 = Mod_MSHX(
        name = 'HX2',
        pressure_drop_percent = 0.0, # %
        stream_pairs = [
            (hpc_n2_to_hx2, cooled_hpc_n2),
            (hpc_states['HPC T15 - liquid'], cooled_hpc_o2),
            (lpc_states['LPC T1 - vapor'], warmed_lpc_n2),
        ],
        approaches = [
            (cooled_hpc_n2, lpc_states['LPC T1 - vapor'], 2.0),
            (hpc_n2_to_hx2, warmed_lpc_n2, 2.0),
        ],
        hot_streams = [
            (hpc_n2_to_hx2, cooled_hpc_n2),
            (hpc_states['HPC T15 - liquid'], cooled_hpc_o2),
        ],
    )
    components[hx2.name] = hx2

    # Main heat exchanger outlets
    lpc_air_to_turbine = State(
        name = 'LPC air to turbine',
        species = species,
        m_dot = lpc_air.m_dot, # kg/s
        T = 115.0,                 # K
        P = lpc_air.P, # Pa
        z = lpc_air.z,
        phase = 'vapor',
    )
    states[lpc_air_to_turbine.name] = lpc_air_to_turbine

    states[cooled_lpc_air.name] = cooled_lpc_air

    lpc_air_turbine = Turbine(
        name = 'LPC air turbine',
        efficiency = 0.92,                # Allam setup value
        P_out = 1.30e5,                    # Pa, LPC tray-6 feed pressure
        inlet = lpc_air_to_turbine,
        outlet = cooled_lpc_air,
    )
    components[lpc_air_turbine.name] = lpc_air_turbine

    warm_n2_product = State(
        name = 'Warm N2 product',
        species = species,
        m_dot = warmed_lpc_n2.m_dot, # kg/s
        T = 298.0,                 # K, initial guess
        P = warmed_lpc_n2.P,       # Pa
        z = warmed_lpc_n2.z,
        phase = 'vapor',
    )
    states[warm_n2_product.name] = warm_n2_product

    warm_o2_product = State(
        name = 'Warm O2 product',
        species = species,
        m_dot = lpc_o2_product.m_dot, # kg/s
        T = 298.0,                 # K, initial guess
        P = lpc_o2_product.P,       # Pa; external O2 compression is not modeled here
        z = lpc_o2_product.z,
        phase = 'vapor',
    )
    states[warm_o2_product.name] = warm_o2_product

    hx1 = Mod_MSHX(
        name = 'HX1',
        pressure_drop_percent = 0.0, # %
        stream_pairs = [
            (mac_air_to_hpc, cooled_hpc_air),
            (lpc_air, lpc_air_to_turbine),
            (warmed_lpc_n2, warm_n2_product),
            (lpc_o2_product, warm_o2_product),
        ],
        fixed_temperatures = [
            (lpc_air_to_turbine, 115.0),
        ],
        approaches = [
            (mac_air_to_hpc, warm_n2_product, 6.0),
            (lpc_air, warm_o2_product, 6.0),
        ],
        hot_streams = [
            (mac_air_to_hpc, cooled_hpc_air),
            (lpc_air, lpc_air_to_turbine),
        ],
    )
    components[hx1.name] = hx1

    hpc_n2_valve = Valve(
        name = 'HPC N2 valve',
        P_out = 1.20e5,           # Pa
        inlet = cooled_hpc_n2,
        outlet = lpc_n2_feed,
    )
    components[hpc_n2_valve.name] = hpc_n2_valve

    hpc_o2_valve = Valve(
        name = 'HPC enriched O2 valve',
        P_out = 1.28e5,           # Pa
        inlet = cooled_hpc_o2,
        outlet = lpc_o2_feed,
    )
    components[hpc_o2_valve.name] = hpc_o2_valve

    x0 = []
    for state in states.values():
        x0.extend(state.flatten_vars())

    return states, components, np.asarray(x0)
