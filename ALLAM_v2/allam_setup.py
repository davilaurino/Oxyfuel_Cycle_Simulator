"""Allam cycle with internal compression ASU."""

import numpy as np

from common.utils import (
    CO2, H2O, O2, N2, AR, CH4, C2H6, C3H8,
    fuel_requirements
)

from common.builders import build_column, build_compressor_train
from common.classes import (
    Input,
    Compressor,
    Pump,
    Mod_MSHX,
    Splitter,
    TurbineCoolingSplitter,
    Mixer,
    State,
    Combustor,
    Turbine,
    Condenser,
)

P_MIN  = 3e6  # Pa
P_ITM = 8e6  # Pa
P_MAX = 3e7  # Pa (max cycle pressure)

ETA_PUMP = 0.88

ETA_COMPRESSOR = 0.87
N_COMPRESSOR_STAGES = 4

ETA_FUEL_COMPRESSOR = 0.85

ETA_TURBINE = 0.92
N_TURBINE_STAGES = 3
TURBINE_PRESSURE_RATIO = (P_MAX/P_MIN)**(1/N_TURBINE_STAGES)

IC_OUTLET_TEMP = 300.0  # K
CONDENSER_OUTLET_TEMP = 300 # K

INTERCOOLER_PRESSURE_DROP_PERCENT = 0.1  # %
RECUPERATOR_PRESSURE_DROP_PERCENT = 0.5  # %

FUEL_TEMP = 298.15  # K
FUEL_PRESSURE = 7.0e6  # Pa
FUEL_SPECIES = [CH4, C2H6, C3H8, CO2]
FUEL_MOLE_FRACTIONS = [0.9400, 0.0300, 0.0200, 0.0100]

O2_TEMP = 298  # K
O2_SPECIES = [O2, N2, AR]
O2_MOLE_FRACTIONS = [0.9800, 0.0020, 0.0180]

K_COOLING = 0.06  # El-Masri cooling coefficient
BLADE_TEMP = 1120  # K — blade metal temperature limit
COOLANT_TEMP = 473  # K — coolant temperature

RECUPERATOR_PINCH_TEMP = 5.0  # K
FLUE_GAS_OUTLET_TEMP = 360.0  # K
CONDENSER_OUTLET_TEMP = 300.0  # K

DEFAULT_EXCESS_O2 = 1.002
DEFAULT_CO2_O2_RATIO = 8.0
DEFAULT_TURBINE_INLET_TEMPERATURE = 1420.0  # K
DEFAULT_FUEL_MASS_FLOW = 7.4  # Kg/s

PRODUCT_SPECIES = [CO2, H2O, O2, N2, AR]

def build_cycle(o2_feed=None):
    """Build the ALLAM cycle with an optional external oxygen stream."""
    states = {}
    components = {}

    # Fuel Input
    fuel_compressor_inlet = State(
        name = 'Fuel Compressor - Inlet',
        species = FUEL_SPECIES,
        m_dot = DEFAULT_FUEL_MASS_FLOW,
        T = FUEL_TEMP,
        P = FUEL_PRESSURE,
        z = np.array(FUEL_MOLE_FRACTIONS),
        phase = 'vapor',
    )
    states[fuel_compressor_inlet.name] = fuel_compressor_inlet

    fuel_inlet = Input(
        name = 'Fuel Inlet',
        m_dot = DEFAULT_FUEL_MASS_FLOW,  # Kg/s
        T = FUEL_TEMP,
        P = FUEL_PRESSURE,
        z = np.array(FUEL_MOLE_FRACTIONS),
        outlet = fuel_compressor_inlet,
    )
    components[fuel_inlet.name] = fuel_inlet

    compressed_fuel = State(
        name = 'Fuel Compressor - Outlet',
        species = FUEL_SPECIES,
        m_dot = DEFAULT_FUEL_MASS_FLOW,
        T = 429.7,  # K
        P = P_MAX,
        z = np.array(FUEL_MOLE_FRACTIONS),
        phase = 'vapor',
    )
    states[compressed_fuel.name] = compressed_fuel

    fuel_compressor = Compressor(
        name = 'Fuel Compressor',
        efficiency = ETA_FUEL_COMPRESSOR,
        pressure_ratio = P_MAX/FUEL_PRESSURE,
        inlet = fuel_compressor_inlet,
        outlet = compressed_fuel,
    )
    components[fuel_compressor.name] = fuel_compressor

    # Mostly temporary calculations
    o2_stream_molar_mass = sum(z_i * species_i.MW for z_i, species_i in zip(O2_MOLE_FRACTIONS, O2_SPECIES))

    m_o2_sto, n_o2_sto = fuel_requirements(fuel_compressor_inlet)
    n_o2_pure = n_o2_sto*DEFAULT_EXCESS_O2
    n_o2_stream = n_o2_pure/O2_MOLE_FRACTIONS[0]
    m_o2_stream = n_o2_stream*o2_stream_molar_mass
    m_co2_oxidant = m_o2_stream*DEFAULT_CO2_O2_RATIO

    # Standalone runs define the O2 boundary here. Integrated runs reuse the
    # ASU product State directly so there is only one set of stream variables.
    if o2_feed is None:
        o2_inlet_stream = State(
            name = "O2 - Inlet Stream",
            species = O2_SPECIES,
            m_dot = m_o2_stream, # Kg/s
            T = O2_TEMP,
            P = P_MAX,
            z = np.array(O2_MOLE_FRACTIONS),
            phase = 'vapor',
        )
        states[o2_inlet_stream.name] = o2_inlet_stream

        o2_inlet = Input(
            name = 'O2 Inlet',
            T = O2_TEMP,
            P = P_MAX,
            z = np.array(O2_MOLE_FRACTIONS),
            outlet = o2_inlet_stream,
        )
        components[o2_inlet.name] = o2_inlet
    else:
        expected_species = [species.name for species in O2_SPECIES]
        actual_species = [species.name for species in o2_feed.species]
        if actual_species != expected_species:
            raise ValueError(
                'External O2 feed species must be ordered as '
                f'{expected_species}; received {actual_species}'
            )
        o2_inlet_stream = o2_feed

    # The condenser vapor closes the cycle and feeds the recycle compressor.
    condenser_vapor = State(
        name = 'Condenser Vapor Outlet',
        species = PRODUCT_SPECIES,
        m_dot = 536.7,  # kg/s
        T = CONDENSER_OUTLET_TEMP,
        P = P_MIN*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'vapor',
    )
    states[condenser_vapor.name] = condenser_vapor

    co2_compressor_states, co2_compressor_components = build_compressor_train(
        name='CO2 Compressor',
        inlet=condenser_vapor,
        P_end=P_ITM,
        cooling_temperature=IC_OUTLET_TEMP,
        n_stages=N_COMPRESSOR_STAGES,
        efficiency=ETA_COMPRESSOR,
        pressure_drop_percent=INTERCOOLER_PRESSURE_DROP_PERCENT,
        phase='vapor',
        final_cooler_phase='liquid',
    )
    states.update(co2_compressor_states)
    components.update(co2_compressor_components)
    co2_compressed = co2_compressor_states['CO2 Compressor outlet']

    co2_recycled = State(
        name = 'Recycled CO2',
        species = PRODUCT_SPECIES,
        m_dot = 516.4,  # kg/s
        T = IC_OUTLET_TEMP,
        P = P_ITM,
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'liquid',
    )
    states[co2_recycled.name] = co2_recycled

    co2_purge = State(
        name = 'Purged CO2',
        species = PRODUCT_SPECIES,
        m_dot = 20.3,  # Kg/s
        T = IC_OUTLET_TEMP,
        P = P_ITM,
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'liquid',
    )
    states[co2_purge.name] = co2_purge

    co2_purge_splitter = Splitter(
        name = 'CO2 Purge Splitter',
        inlet = co2_compressed,
        outlets = [co2_recycled, co2_purge],
    )
    components[co2_purge_splitter.name] = co2_purge_splitter

    co2_max_pressure = State(
        name = 'CO2 Max. Pressure',
        species = PRODUCT_SPECIES,
        m_dot = 516.4,  # kg/s
        T = 343.5,  # K
        P = P_MAX,
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'liquid',
    )
    states[co2_max_pressure.name] = co2_max_pressure

    co2_pump = Pump(
        name = 'CO2 Pump',
        efficiency = ETA_PUMP,
        P_out = P_MAX,
        inlet = co2_recycled,
        outlet = co2_max_pressure,
    )
    components[co2_pump.name] = co2_pump

    co2_combustion = State(
        name = 'CO2 Combustion',
        species = PRODUCT_SPECIES,
        m_dot = m_co2_oxidant,  # kg/s
        T = 343.5,  # K
        P = P_MAX,
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'liquid',
    )
    states[co2_combustion.name] = co2_combustion

    co2_dilution = State(
        name = 'CO2 Dilution',
        species = PRODUCT_SPECIES,
        m_dot = 260.4,  # Kg/s
        T = 343.5,  # K
        P = P_MAX,
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'liquid',
    )
    states[co2_dilution.name] = co2_dilution

    co2_cooling = State(
        name = 'CO2 Cooling',
        species = PRODUCT_SPECIES,
        m_dot = 21.6,  # Kg/s
        T = 343.5,  # K
        P = P_MAX,
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'liquid',
    )
    states[co2_cooling.name] = co2_cooling

    co2_recycle_splitter = Splitter(
        name = 'CO2 Recycle Splitter',
        inlet = co2_max_pressure,
        outlets = [co2_combustion, co2_dilution, co2_cooling],
    )
    components[co2_recycle_splitter.name] = co2_recycle_splitter

    oxidant = State(
        name = 'Oxidant',
        species = PRODUCT_SPECIES,
        m_dot = 263.7,  # Kg/s
        T = 331.1,  # K
        P = P_MAX,
        z = np.array([0.8193, 0.0015, 0.1428, 0.0036, 0.0328]),
        phase = 'liquid',
    )
    states[oxidant.name] = oxidant

    oxidant_mixer = Mixer(
        name = 'Oxidant mixer',
        inlets = [co2_combustion, o2_inlet_stream],
        outlet = oxidant,
        inlet_mass_ratios = [DEFAULT_CO2_O2_RATIO, 1.0],
    )
    components[oxidant_mixer.name] = oxidant_mixer

    heated_oxidant = State(
        name = 'Heated Oxidant',
        species = PRODUCT_SPECIES,
        m_dot = 263.7,  # Kg/s
        T = 860.3,  # K
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.8193, 0.0015, 0.1428, 0.0036, 0.0328]),
        phase = 'vapor',
    )
    states[heated_oxidant.name] = heated_oxidant

    heated_co2_dilution = State(
        name = 'Heated Dilution CO2',
        species = PRODUCT_SPECIES,
        m_dot = 260.4,  # Kg/s
        T = 993.2,  # K
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'vapor',
    )
    states[heated_co2_dilution.name] = heated_co2_dilution

    heated_co2_cooling = State(
        name = 'Heated Cooling CO2',
        species = PRODUCT_SPECIES,
        m_dot = 21.6,  # Kg/s
        T = COOLANT_TEMP,
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
        phase = 'vapor',
    )
    states[heated_co2_cooling.name] = heated_co2_cooling

    flue_gas = State(
        name = 'Flue Gases',
        species = PRODUCT_SPECIES,
        m_dot = 553.1,  # Kg/s
        T = 360.0,  # K
        P = P_MIN*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.8949, 0.0682, 0.0003, 0.0037, 0.0329]),
    )
    states[flue_gas.name] = flue_gas

    combustion_products = State(
        name = 'Combustion Products',
        species = PRODUCT_SPECIES,
        m_dot = 271.1,  # Kg/s
        T = 1796.0,  # K
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.8335, 0.1322, 0.0003, 0.0034, 0.0306]),
        phase = 'vapor',
    )
    states[combustion_products.name] = combustion_products

    combustor = Combustor(
        name = 'Combustor',
        fuel = compressed_fuel,
        oxidant = heated_oxidant,
        outlet = combustion_products,
        excess_o2 = DEFAULT_EXCESS_O2
    )
    components[combustor.name] = combustor

    diluted_products = State(
        name = 'Diluted Products',
        species = PRODUCT_SPECIES,
        m_dot = 531.5,  # Kg/s
        T = DEFAULT_TURBINE_INLET_TEMPERATURE,  # K
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.8924, 0.0708, 0.0003, 0.0036, 0.0329]),
        phase = 'vapor',
    )
    states[diluted_products.name] = diluted_products

    dilution_mixer = Mixer(
        name = 'Dilution Mixer',
        inlets = [combustion_products, heated_co2_dilution],
        outlet = diluted_products,
    )
    components[dilution_mixer.name] = dilution_mixer

    hp_turbine_outlet = State(
        name = 'HP Turbine - Outlet',
        species = PRODUCT_SPECIES,
        m_dot = 531.5,  # Kg/s
        T = 1275.6,  # K
        P = P_MAX/TURBINE_PRESSURE_RATIO,   # Pa
        z = np.array([0.8924, 0.0708, 0.0003, 0.0036, 0.0329]),
        phase = 'vapor',
    )
    states[hp_turbine_outlet.name] = hp_turbine_outlet

    hp_turbine = Turbine(
        name = 'HP Turbine',
        efficiency = ETA_TURBINE,
        P_out = P_MAX/TURBINE_PRESSURE_RATIO,
        inlet = diluted_products,
        outlet = hp_turbine_outlet,
        TIT = DEFAULT_TURBINE_INLET_TEMPERATURE,
    )
    components[hp_turbine.name] = hp_turbine

    hp_co2_cooling = State(
        name = 'HP Turbine cooling CO2',
        species = PRODUCT_SPECIES,
        m_dot = 14.8,  # Kg/s
        T = 437.5,
        P = P_MAX/TURBINE_PRESSURE_RATIO,
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
    )
    states[hp_co2_cooling.name] = hp_co2_cooling

    ip_turbine_inlet = State(
        name = 'IP Turbine Inlet',
        species = PRODUCT_SPECIES,
        m_dot = 546.3,  # Kg/s
        T = 1254.6,  # K
        P = P_MAX/TURBINE_PRESSURE_RATIO,
        z = np.array([0.8942, 0.0690, 0.0003, 0.0037, 0.0328]),
        phase = 'vapor',
    )
    states[ip_turbine_inlet.name] = ip_turbine_inlet

    hp_coolant_mixer = Mixer(
        name = "HP Turbine Coolant Mixer",
        inlets = [hp_turbine_outlet, hp_co2_cooling],
        outlet = ip_turbine_inlet,
    )
    components[hp_coolant_mixer.name] = hp_coolant_mixer

    ip_turbine_outlet = State(
        name = 'IP Turbine - Outlet',
        species = PRODUCT_SPECIES,
        m_dot = 546.3,  # Kg/s
        T = 1124.2,  # K
        P = P_MAX/(TURBINE_PRESSURE_RATIO**2),   # Pa
        z = np.array([0.8942, 0.0690, 0.0003, 0.0037, 0.0328]),
        phase = 'vapor',
    )
    states[ip_turbine_outlet.name] = ip_turbine_outlet

    ip_turbine = Turbine(
        name = 'IP Turbine',
        efficiency = ETA_TURBINE,
        P_out = P_MAX/(TURBINE_PRESSURE_RATIO**2),
        inlet = ip_turbine_inlet,
        outlet = ip_turbine_outlet,
    )
    components[ip_turbine.name] = ip_turbine

    ip_co2_cooling = State(
        name = 'IP Turbine cooling CO2',
        species = PRODUCT_SPECIES,
        m_dot = 6.8,  # Kg/s
        T = 408.8,
        P = P_MAX/(TURBINE_PRESSURE_RATIO**2),
        z = np.array([0.9588, 0.0017, 0.0003, 0.0039, 0.0353]),
    )
    states[ip_co2_cooling.name] = ip_co2_cooling

    coolant_splitter = TurbineCoolingSplitter(
        name = 'Turbine Coolant Splitter',
        cooling_coefficient = K_COOLING,
        blade_temperature = BLADE_TEMP,
        inlet = heated_co2_cooling,
        outlets = [hp_co2_cooling, ip_co2_cooling],
        hot_gas_inlets = [diluted_products, ip_turbine_inlet],
        turbine_outlets = [hp_turbine_outlet, ip_turbine_outlet],
    )
    components[coolant_splitter.name] = coolant_splitter

    lp_turbine_inlet = State(
        name = 'LP Turbine Inlet',
        species = PRODUCT_SPECIES,
        m_dot = 553.1,  # Kg/s
        T = 1116.2,  # K
        P = P_MAX/(TURBINE_PRESSURE_RATIO**2),
        z = np.array([0.8949, 0.0682, 0.0003, 0.0037, 0.0329]),
        phase = 'vapor',
    )
    states[lp_turbine_inlet.name] = lp_turbine_inlet

    ip_coolant_mixer = Mixer(
        name = 'IP Turbine Coolant Mixer',
        inlets = [ip_turbine_outlet, ip_co2_cooling],
        outlet = lp_turbine_inlet,
    )
    components[ip_coolant_mixer.name] = ip_coolant_mixer

    lp_turbine_outlet = State(
        name = 'LP Turbine Outlet',
        species = PRODUCT_SPECIES,
        m_dot = 553.1,  # Kg/s
        T = 998.2,  # K
        P = P_MIN,
        z = np.array([0.8949, 0.0682, 0.0003, 0.0037, 0.0329]),
        phase = 'vapor',
    )
    states[lp_turbine_outlet.name] = lp_turbine_outlet

    lp_turbine = Turbine(
        name = 'LP Turbine',
        efficiency = ETA_TURBINE,
        P_out = P_MIN,
        inlet = lp_turbine_inlet,
        outlet = lp_turbine_outlet,
    )
    components[lp_turbine.name] = lp_turbine

    # MSHX
    recuperator = Mod_MSHX(
        name = 'MSHX Recuperator',
        pressure_drop_percent = RECUPERATOR_PRESSURE_DROP_PERCENT,
        stream_pairs = [
            (lp_turbine_outlet, flue_gas),
            (oxidant, heated_oxidant),
            (co2_dilution, heated_co2_dilution),
            (co2_cooling, heated_co2_cooling),
        ],
        approaches = [
            (lp_turbine_outlet, heated_co2_dilution, RECUPERATOR_PINCH_TEMP),
        ],
        fixed_temperatures=[
            (heated_co2_cooling, COOLANT_TEMP),
            (flue_gas, FLUE_GAS_OUTLET_TEMP),
        ],
    )
    components[recuperator.name] = recuperator

    condenser_liquid = State(
        name = 'Condenser Liquid Outlet',
        species = PRODUCT_SPECIES,
        m_dot = 16.4,  # kg/s
        T = CONDENSER_OUTLET_TEMP,
        P = P_MIN*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.0148, 0.9849, 0.0001, 0.0001, 0.0001]),
        phase = 'liquid',
    )
    states[condenser_liquid.name] = condenser_liquid

    condenser = Condenser(
        name = 'Flue Gas Condenser',
        T_out = CONDENSER_OUTLET_TEMP,
        inlet = flue_gas,
        vapor_outlet = condenser_vapor,
        liquid_outlet = condenser_liquid,
    )
    components[condenser.name] = condenser


    x0 = np.asarray([
        value
        for state in states.values()
        for value in state.flatten_vars()
    ], dtype=float)

    return states, components, x0
    

