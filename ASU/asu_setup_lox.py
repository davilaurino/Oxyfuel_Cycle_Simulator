"""Liquid-O2 internal-compression ASU setup."""

import numpy as np

from common.builders import build_column, build_compressor_train
from common.classes import (
    Input,
    Pump,
    Mod_MSHX,
    O2Specification,
    ReboilerCondenser,
    Splitter,
    State,
    Turbine,
    Valve,
)


def build_asu(o2_mass_flow=29.24):
    """Build the LOX ASU, optionally leaving product flow unspecified."""
    species = ['O2', 'N2', 'AR']
    air_composition = [0.209390, 0.780848, 0.009762]
    states = {}
    components = {}

    # Main air compressor: ambient air to the HPC pressure level.
    mac_air_in = State(
        name = 'MAC air intake',
        species = species,
        m_dot = 126.5,             # kg/s
        T = 298.1,               # K
        P = 101325.0,             # Pa
        z = [0.2094, 0.7808, 0.0098],
        phase = 'vapor',
    )
    states[mac_air_in.name] = mac_air_in

    mac_air_input = Input(
        name = 'MAC air intake - Input',
        T = mac_air_in.T,
        P = mac_air_in.P,
        z = mac_air_in.z,
        outlet = mac_air_in,
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

    lp_air_to_hx1 = State(
        name = 'LP air to HX1',
        species = species,
        m_dot = 12.9,              # kg/s
        T = 300.0,                # K
        P = 600000.0,               # Pa
        z = [0.2094, 0.7808, 0.0098],
        phase = 'vapor',
    )
    states[lp_air_to_hx1.name] = lp_air_to_hx1

    air_to_next_compression = State(
        name = 'Air to next compression train',
        species = species,
        m_dot = 113.6,              # kg/s
        T = 300.0,                # K
        P = 600000.0,               # Pa
        z = [0.2094, 0.7808, 0.0098],
        phase = 'vapor',
    )
    states[air_to_next_compression.name] = air_to_next_compression

    mac_air_splitter = Splitter(
        name = 'MAC air splitter',
        inlet = mac_air,
        outlets = [lp_air_to_hx1, air_to_next_compression],
    )
    components[mac_air_splitter.name] = mac_air_splitter

    # Two-stage MP booster. Its aftercoolers return the stream to 300 K
    # before the split between the MP expansion path and future HP booster.
    mp_booster_states, mp_booster_components = build_compressor_train(
        name = 'MP booster',
        inlet = air_to_next_compression,
        P_end = 1.80e6,           # Pa
        cooling_temperature = 300.0, # K
        n_stages = 2,
        efficiency = 0.87,
        pressure_drop_percent = 0.1, # % per cooler
    )
    states.update(mp_booster_states)
    components.update(mp_booster_components)
    mp_air = mp_booster_states['MP booster outlet']

    mp_air_to_hx1 = State(
        name = 'MP air to HX1',
        species = species,
        m_dot = 82.2,              # kg/s
        T = 300.0,                # K
        P = 1800000.0,               # Pa
        z = [0.2094, 0.7808, 0.0098],
        phase = 'vapor',
    )
    states[mp_air_to_hx1.name] = mp_air_to_hx1

    air_to_hp_compression = State(
        name = 'Air to HP compression train',
        species = species,
        m_dot = 31.4,              # kg/s
        T = 300.0,                # K
        P = 1800000.0,               # Pa
        z = [0.2094, 0.7808, 0.0098],
        phase = 'vapor',
    )
    states[air_to_hp_compression.name] = air_to_hp_compression

    mp_air_splitter = Splitter(
        name = 'MP air splitter',
        inlet = mp_air,
        outlets = [mp_air_to_hx1, air_to_hp_compression],
    )
    components[mp_air_splitter.name] = mp_air_splitter

    # Single-stage HP booster. The 4 MPa outlet supplies the condensing
    # air path through HX1 and the HP throttle into the HPC.
    hp_booster_states, hp_booster_components = build_compressor_train(
        name = 'HP booster',
        inlet = air_to_hp_compression,
        P_end = 4.00e6,           # Pa
        cooling_temperature = 300.0, # K
        n_stages = 1,
        efficiency = 0.87,
        pressure_drop_percent = 0.1, # % per cooler
    )
    states.update(hp_booster_states)
    components.update(hp_booster_components)
    hp_air = hp_booster_states['HP booster outlet']

    # Cold air states are declared before the column so the real downstream
    # compressor/HX paths can feed the HPC directly.
    hp_air_to_hpc = State(
        name = 'Throttled HP air',
        species = species,
        m_dot = 31.4,              # kg/s
        T = 98.7,           # K, converged initial guess
        P = 600000.0,               # Pa
        z = [0.2094, 0.7808, 0.0098],
    )
    states[hp_air_to_hpc.name] = hp_air_to_hpc

    mp_air_to_hpc = State(
        name = 'Expanded MP air',
        species = species,
        m_dot = 82.2,              # kg/s
        T = 100.6,          # K, converged initial guess
        P = 600000.0,               # Pa
        z = [0.2094, 0.7808, 0.0098],
    )
    states[mp_air_to_hpc.name] = mp_air_to_hpc

    cooled_lp_air = State(
        name = 'Cooled LP air',
        species = species,
        m_dot = 12.9, # kg/s
        T = 102.7,          # K, converged initial guess
        P = 600000.0,      # Pa; HPC tray-15 pressure
        z = [0.2094, 0.7808, 0.0098],
        phase = 'vapor',
    )
    states[cooled_lp_air.name] = cooled_lp_air

    # HPC reflux from the coupled condenser.
    hpc_reflux = State(
        name = 'HPC reflux',
        species = species,
        m_dot = 50.2,           # kg/s
        T = 96.0,                 # K
        P = 590000.0,               # Pa
        z = [0.0066, 0.9910, 0.0024],
        phase = 'liquid',
    )
    states[hpc_reflux.name] = hpc_reflux

    # Feed locations follow the converged tray composition profile: the
    # two-phase HP air matches tray 10, while the cooler LP vapor enters above
    # the hotter MP vapor on the bottom two trays.
    hpc_states, hpc_components = build_column(
        name = 'HPC',
        species = species,
        n_trays = 15,
        P_top = 5.90e5,           # Pa
        P_bottom = 6.00e5,        # Pa
        L_top = 50.3,             # kg/s
        L_bottom = 80.2,          # kg/s
        V_top = 96.5,             # kg/s
        V_bottom = 92.0,          # kg/s
        x_top = [0.0160, 0.9796, 0.0044],
        x_bottom = [0.3325, 0.6532, 0.0143],
        feeds = {
            1: [hpc_reflux],
            12: [hp_air_to_hpc],
            15: [cooled_lp_air, mp_air_to_hpc],
        },
    )
    states.update(hpc_states)
    components.update(hpc_components)

    condensed_hpc_n2 = State(
        name = 'Condensed HPC N2',
        species = species,
        m_dot = 96.5,           # kg/s
        T = 96.0,                 # K
        P = 590000.0,               # Pa
        z = [0.0066, 0.9910, 0.0024],
        phase = 'liquid',
    )
    states[condensed_hpc_n2.name] = condensed_hpc_n2

    hpc_n2_to_hx2 = State(
        name = 'HPC N2 to HX2',
        species = species,
        m_dot = 46.3,           # kg/s
        T = 96.0,                 # K
        P = 590000.0,               # Pa
        z = [0.0066, 0.9910, 0.0024],
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

    # Two inter-column LPC feeds: liquid nitrogen reflux from the HPC top and
    # oxygen-enriched liquid from the HPC bottom. Reboiler vapor is returned
    # separately at the LPC bottom and is not counted as a process feed.
    lpc_n2_feed = State(
        name = 'LPC N2 feed',
        species = species,
        m_dot = 46.3,           # kg/s
        T = 78.9,              # K
        P = 120000.0,               # Pa
        z = [0.0066, 0.9910, 0.0024],
    )
    states[lpc_n2_feed.name] = lpc_n2_feed

    lpc_o2_feed = State(
        name = 'LPC enriched O2 feed',
        species = species,
        m_dot = 80.2,           # kg/s
        T = 82.5,              # K
        P = 128000.0,               # Pa
        z = [0.3325, 0.6532, 0.0143],
    )
    states[lpc_o2_feed.name] = lpc_o2_feed

    lpc_o2_product = State(
        name = 'LPC O2 product',
        species = species,
        m_dot = 29.2,           # kg/s
        T = 92.5,              # K
        P = 130000.0,               # Pa
        z = [0.9800, 0.0001, 0.0199],
        phase = 'liquid',
    )
    states[lpc_o2_product.name] = lpc_o2_product

    o2_specification = O2Specification(
        name = 'O2 Specification',
        o2_product = lpc_o2_product,
        purity_target = 0.98,
        m_dot = o2_mass_flow,
    )
    components[o2_specification.name] = o2_specification

    lpc_boilup = State(
        name = 'LPC boilup',
        species = species,
        m_dot = 78.6,           # kg/s
        T = 92.5,              # K
        P = 130000.0,               # Pa
        z = [0.9690, 0.0005, 0.0305],
        phase = 'vapor',
    )
    states[lpc_boilup.name] = lpc_boilup

    lpc_states, lpc_components = build_column(
        name = 'LPC',
        species = species,
        n_trays = 20,
        P_top = 1.20e5,           # Pa
        P_bottom = 1.30e5,        # Pa
        L_top = 45.2,             # kg/s
        L_bottom = 107.8,         # kg/s
        V_top = 97.3,             # kg/s
        V_bottom = 78.8,          # kg/s
        x_top = [0.0252, 0.9573, 0.0175],
        x_bottom = [0.9720, 0.0004, 0.0276],
        feeds = {
            1: [lpc_n2_feed],
            12: [lpc_o2_feed],
            20: [lpc_boilup],
        },
    )
    states.update(lpc_states)
    components.update(lpc_components)

    reboiler_condenser = ReboilerCondenser(
        name = 'HPC condenser - LPC reboiler',
        condenser_vapor_in = hpc_states['HPC T1 - vapor'],
        condenser_liquid_out = condensed_hpc_n2,
        reboiler_liquid_in = lpc_states['LPC T20 - liquid'],
        reboiler_liquid_out = lpc_o2_product,
        reboiler_vapor_out = lpc_boilup,
        T_cold_pinch = 3.5,   # K
    )
    components[reboiler_condenser.name] = reboiler_condenser

    cooled_hpc_n2 = State(
        name = 'Cooled HPC N2',
        species = species,
        m_dot = 46.3,           # kg/s
        T = 81.1,              # K
        P = 590000.0,               # Pa
        z = [0.0066, 0.9910, 0.0024],
        phase = 'liquid',
    )
    states[cooled_hpc_n2.name] = cooled_hpc_n2

    cooled_hpc_o2 = State(
        name = 'Cooled HPC enriched O2',
        species = species,
        m_dot = 80.2,           # kg/s
        T = 99.2,              # K
        P = 600000.0,               # Pa
        z = [0.3325, 0.6532, 0.0143],
        phase = 'liquid',
    )
    states[cooled_hpc_o2.name] = cooled_hpc_o2

    warmed_lpc_n2 = State(
        name = 'Warmed LPC N2',
        species = species,
        m_dot = 97.3,           # kg/s
        T = 94.0,                 # K
        P = 120000.0,               # Pa
        z = [0.0068, 0.9861, 0.0071],
        phase = 'vapor',
    )
    states[warmed_lpc_n2.name] = warmed_lpc_n2

    # Direct internal compression of the LPC liquid oxygen product.
    pressurized_lox = State(
        name = 'Pressurized LOX',
        species = species,
        m_dot = 29.2, # kg/s
        T = 100.7,                # K, pump-outlet initial guess
        P = 30000000.0,               # Pa
        z = [0.9800, 0.0001, 0.0199],
        phase = 'liquid',
    )
    states[pressurized_lox.name] = pressurized_lox

    # Remaining main-exchanger outlets.
    cooled_mp_air = State(
        name = 'Cooled MP air',
        species = species,
        m_dot = 82.2, # kg/s
        T = 130.0,                # K, fixed by HX1 specification
        P = 1800000.0,      # Pa
        z = [0.2094, 0.7808, 0.0098],
    )
    states[cooled_mp_air.name] = cooled_mp_air

    cooled_hp_air = State(
        name = 'Cooled HP air',
        species = species,
        m_dot = 31.4, # kg/s
        T = 102.7,                # K, energy-balance initial guess
        P = 4000000.0, # Pa
        z = [0.2094, 0.7808, 0.0098],
        phase = 'liquid',
    )
    states[cooled_hp_air.name] = cooled_hp_air

    n2_product = State(
        name = 'N2 product',
        species = species,
        m_dot = 97.3, # kg/s
        T = 298.0,                # K, 6 K warm-end approach
        P = 120000.0,      # Pa
        z = [0.0068, 0.9861, 0.0071],
        phase = 'vapor',
    )
    states[n2_product.name] = n2_product

    o2_product = State(
        name = 'O2 product',
        species = species,
        m_dot = 29.2, # kg/s
        T = 298.0,                # K, 6 K warm-end approach
        P = 30000000.0,    # Pa
        z = [0.9800, 0.0001, 0.0199],
        phase = 'vapor',
    )
    states[o2_product.name] = o2_product

    lox_pump = Pump(
        name = 'LOX pump',
        efficiency = 0.88,
        P_out = 30.0e6,
        inlet = lpc_o2_product,
        outlet = pressurized_lox,
    )

    hx2 = Mod_MSHX(
        name = 'HX2',
        pressure_drop_percent = 0.0,
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

    hpc_n2_valve = Valve(
        name = 'HPC N2 valve',
        P_out = 1.20e5,
        inlet = cooled_hpc_n2,
        outlet = lpc_n2_feed,
    )
    components[hpc_n2_valve.name] = hpc_n2_valve

    hpc_o2_valve = Valve(
        name = 'HPC enriched O2 valve',
        P_out = 1.28e5,
        inlet = cooled_hpc_o2,
        outlet = lpc_o2_feed,
    )
    components[hpc_o2_valve.name] = hpc_o2_valve

    components[lox_pump.name] = lox_pump

    hx1 = Mod_MSHX(
        name = 'HX1',
        pressure_drop_percent = 0.0, # %
        stream_pairs = [
            (lp_air_to_hx1, cooled_lp_air),
            (mp_air_to_hx1, cooled_mp_air),
            (hp_air, cooled_hp_air),
            (warmed_lpc_n2, n2_product),
            (pressurized_lox, o2_product),
        ],
        fixed_temperatures = [
            (cooled_mp_air, 130.0),
        ],
        approaches = [
            (hp_air, n2_product, 2.0),
            (lp_air_to_hx1, o2_product, 2.0),
            (cooled_lp_air, pressurized_lox, 2.0),
            (cooled_hp_air, pressurized_lox, 2.0),
        ],
        hot_streams = [
            (lp_air_to_hx1, cooled_lp_air),
            (mp_air_to_hx1, cooled_mp_air),
            (hp_air, cooled_hp_air),
        ],
    )
    components[hx1.name] = hx1

    hp_air_valve = Valve(
        name = 'HP air valve',
        P_out = 6.00e5,           # Pa
        inlet = cooled_hp_air,
        outlet = hp_air_to_hpc,
    )
    components[hp_air_valve.name] = hp_air_valve

    mp_air_turbine = Turbine(
        name = 'MP air turbine',
        efficiency = 0.92,
        P_out = 6.00e5,           # Pa
        inlet = cooled_mp_air,
        outlet = mp_air_to_hpc,
    )
    components[mp_air_turbine.name] = mp_air_turbine

    x0 = []
    for state in states.values():
        x0.extend(state.flatten_vars())

    return states, components, np.asarray(x0)
