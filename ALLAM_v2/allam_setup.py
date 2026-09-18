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
    Mixer,
    State,
    Combustor,
    Turbine,
    Valve,
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
FUEL_MOLE_FRACTIONS = [0.94, 0.03, 0.02, 0.01]

O2_TEMP = 298  # K
O2_SPECIES = [O2, N2, AR]
O2_MOLE_FRACTIONS = [0.9800, 0.0020, 0.0180]

K_COOLING = 0.06  # El-Masri cooling coefficient
BLADE_TEMP = 1120  # K — blade metal temperature limit
COOLANT_TEMP = 473  # K — coolant temperature

RECUPERATOR_PINCH_TEMP = 5.0  # K
FLUE_GAS_OUTLET_TEMP = 400.0  # K
CONDENSER_OUTLET_TEMP = 300.0  # K

DEFAULT_EXCESS_O2 = 1.01
DEFAULT_CO2_O2_RATIO = 8.0
DEFAULT_TURBINE_INLET_TEMPERATURE = 1500.0  # K
DEFAULT_FUEL_MASS_FLOW = 7.4  # Kg/s

PRODUCT_SPECIES = [CO2, H2O, O2, N2, AR]

def build_cycle():
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
        T = 429.70, # K
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

    # O2 Input
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
        m_dot = m_o2_stream,  # Kg/s
        T = O2_TEMP,
        P = P_MAX,
        z = np.array(O2_MOLE_FRACTIONS),
        outlet = o2_inlet_stream,
    )
    components[o2_inlet.name] = o2_inlet

    # This stream will eventually be connected to the condenser outlet.  It
    # is an unconstrained initial state for now, so no Input component is
    # added at this stage.
    co2_compressor_inlet = State(
        name = 'CO2 Compressor - Inlet',
        species = PRODUCT_SPECIES,
        m_dot = 500, # Kg/s
        T = CONDENSER_OUTLET_TEMP,
        P = P_MIN,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        phase = 'vapor',
    )
    states[co2_compressor_inlet.name] = co2_compressor_inlet

    # Associated fake input
    co2_fake = Input(
        name = 'Fake CO2 Input',
        m_dot = 500,  # Kg/s
        T = CONDENSER_OUTLET_TEMP,
        P = P_MIN,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        outlet = co2_compressor_inlet,
    )
    components[co2_fake.name] = co2_fake

    co2_compressor_states, co2_compressor_components = build_compressor_train(
        name='CO2 Compressor',
        inlet=co2_compressor_inlet,
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
        m_dot = 475,  # Kg/s
        T = IC_OUTLET_TEMP,
        P = P_ITM,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        phase = 'liquid',
    )
    states[co2_recycled.name] = co2_recycled

    co2_purge = State(
        name = 'Purged CO2',
        species = PRODUCT_SPECIES,
        m_dot = 25,  # Kg/s
        T = IC_OUTLET_TEMP,
        P = P_ITM,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        phase = 'liquid',
    )
    states[co2_purge.name] = co2_purge

    co2_purge_splitter = Splitter(
        name = 'CO2 Purge Splitter',
        inlet = co2_compressed,
        outlets = [co2_recycled, co2_purge],
        split_fractions = [0.95, 0.05],
    )
    components[co2_purge_splitter.name] = co2_purge_splitter

    co2_max_pressure = State(
        name = 'CO2 Max. Pressure',
        species = PRODUCT_SPECIES,
        m_dot = 475,  # Kg/s
        T = 328.59,  # K
        P = P_MAX,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
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
        m_dot = m_co2_oxidant,  # Kg/s
        T = 328.59,  # K
        P = P_MAX,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        phase = 'liquid',
    )
    states[co2_combustion.name] = co2_combustion

    co2_dilution = State(
        name = 'CO2 Dilution',
        species = PRODUCT_SPECIES,
        m_dot = 213.77,  # Kg/s
        T = 328.59,  # K
        P = P_MAX,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        phase = 'liquid',
    )
    states[co2_dilution.name] = co2_dilution

    co2_cooling = State(
        name = 'CO2 Cooling',
        species = PRODUCT_SPECIES,
        m_dot = 25,  # Kg/s
        T = 328.59,  # K
        P = P_MAX,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        phase = 'liquid',
    )
    states[co2_cooling.name] = co2_cooling

    co2_recycle_splitter = Splitter(
        name = 'CO2 Recycle Splitter',
        inlet = co2_max_pressure,
        outlets = [co2_combustion, co2_dilution, co2_cooling],
        outlet_flows = [m_co2_oxidant, None, 25],
    )
    components[co2_recycle_splitter.name] = co2_recycle_splitter

    oxidant = State(
        name = 'Oxidant',
        species = PRODUCT_SPECIES,
        m_dot = 265.76,  # Kg/s
        T = 317.03,  # K
        P = P_MAX,
        z = np.array([0.853034, 0.000598, 0.143275, 0.000378, 0.002715]),
        phase = 'liquid',
    )
    states[oxidant.name] = oxidant

    oxidant_mixer = Mixer(
        name = 'Oxidant mixer',
        inlets = [co2_combustion, o2_inlet_stream],
        outlet = oxidant,
    )
    components[oxidant_mixer.name] = oxidant_mixer

    heated_oxidant = State(
        name = 'Heated Oxidant',
        species = PRODUCT_SPECIES,
        m_dot = 265.76,  # Kg/s
        T = 1193.33,  # K
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.853034, 0.000598, 0.143275, 0.000378, 0.002715]),
        phase = 'vapor',
    )
    states[heated_oxidant.name] = heated_oxidant

    heated_co2_dilution = State(
        name = 'Heated Dilution CO2',
        species = PRODUCT_SPECIES,
        m_dot = 213.77,  # Kg/s
        T = 1195,  # K
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        phase = 'vapor',
    )
    states[heated_co2_dilution.name] = heated_co2_dilution

    heated_co2_cooling = State(
        name = 'Heated Cooling CO2',
        species = PRODUCT_SPECIES,
        m_dot = 25,  # Kg/s
        T = COOLANT_TEMP,
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
        phase = 'vapor',
    )
    states[heated_co2_cooling.name] = heated_co2_cooling

    lp_turbine_outlet = State(
        name = 'LP Turbine Outlet',
        species = PRODUCT_SPECIES,
        m_dot = 600,  # Kg/s
        T = 1200,  # K
        P = P_MIN,
        z = np.array([0.8973, 0.0839, 0.0011, 0.0177, 0.0001]),
        phase = 'vapor',
    )
    states[lp_turbine_outlet.name] = lp_turbine_outlet

    fake_products = Input(
        name = 'Fake hot products',
        m_dot = 600,  # Kg/s (Temporary)
        T = 1200,  # K (Temporary)
        P = P_MIN,
        z = np.array([0.8973, 0.0839, 0.0011, 0.0177, 0.0001]),
        outlet = lp_turbine_outlet,
    )
    components[fake_products.name] = fake_products

    flue_gas = State(
        name = 'Flue Gases',
        species = PRODUCT_SPECIES,
        m_dot = 600,  # Kg/s
        T = 400,  # K
        P = P_MIN*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.8973, 0.0839, 0.0011, 0.0177, 0.0001]),
        phase = 'vapor',
    )
    states[flue_gas.name] = flue_gas

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

    combustion_products = State(
        name = 'Combustion Products',
        species = PRODUCT_SPECIES,
        m_dot = 273.16,  # Kg/s
        T = 2055.79,  # K
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.864920, 0.130786, 0.001404, 0.000353, 0.002537]),
    )
    states[combustion_products.name] = combustion_products

    combustor = Combustor(
        name = 'Combustor',
        fuel = compressed_fuel,
        oxidant = heated_oxidant,
        outlet = combustion_products,
    )
    components[combustor.name] = combustor

    diluted_products = State(
        name = 'Diluted Products',
        species = PRODUCT_SPECIES,
        m_dot = 486.93,  # Kg/s
        T = 1695.01,  # K
        P = P_MAX*(1 - RECUPERATOR_PRESSURE_DROP_PERCENT/100),
        z = np.array([0.921133, 0.076247, 0.000857, 0.000247, 0.001516]),
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
        m_dot = 486.93,  # Kg/s
        T = 1531.41,  # K
        P = P_MAX/TURBINE_PRESSURE_RATIO,   # Pa
        z = np.array([0.921133, 0.076247, 0.000857, 0.000247, 0.001516]),
    )
    states[hp_turbine_outlet.name] = hp_turbine_outlet

    hp_turbine = Turbine(
        name = 'HP Turbine',
        efficiency = ETA_TURBINE,
        P_out = P_MAX/TURBINE_PRESSURE_RATIO,
        inlet = diluted_products,
        outlet = hp_turbine_outlet,
    )
    components[hp_turbine.name] = hp_turbine

    ip_turbine_inlet = State(
        name = 'IP Turbine Inlet',
        species = PRODUCT_SPECIES,
        m_dot = 500,  # Kg/s
        T = 1500,  # K
        P = P_MAX/TURBINE_PRESSURE_RATIO,
        z = np.array([0.921133, 0.076247, 0.000857, 0.000247, 0.001516]),
    )
    states[ip_turbine_inlet.name] = ip_turbine_inlet

    hp_co2_cooling = State(
        name = 'HP Turbine cooling CO2',
        species = PRODUCT_SPECIES,
        m_dot = 15,  # Kg/s
        T = COOLANT_TEMP,
        P = P_MAX/TURBINE_PRESSURE_RATIO,
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
    )
    states[hp_co2_cooling.name] = hp_co2_cooling

    ip_co2_cooling = State(
        name = 'IP Turbine cooling CO2',
        species = PRODUCT_SPECIES,
        m_dot = 10,  # Kg/s
        T = COOLANT_TEMP,
        P = P_MAX/(2*TURBINE_PRESSURE_RATIO),
        z = np.array([0.9990, 0.0007, 0.0001, 0.0001, 0.0001]),
    )
    states[ip_co2_cooling.name] = ip_co2_cooling


    x0 = np.asarray([
        value
        for state in states.values()
        for value in state.flatten_vars()
    ], dtype=float)

    return states, components, x0
    

