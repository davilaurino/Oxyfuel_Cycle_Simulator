"""Reusable builders for ASU process structures."""

import CoolProp.CoolProp as CP
import numpy as np
import utils

from ASU.asu_classes import Compressor, Intercooler, State, Tray


def build_compressor_train(
    name,
    inlet,
    P_start,
    P_end,
    cooling_temperature,
    n_stages,
    efficiency,
    pressure_drop_percent=0.0,
):
    """Create equal-ratio gas-compression stages with cooling after each stage."""
    if not isinstance(n_stages, (int, np.integer)) or n_stages < 1:
        raise ValueError('n_stages must be a positive integer')
    if P_end <= P_start:
        raise ValueError('P_end must be greater than P_start')
    if not np.isclose(inlet.P, P_start):
        raise ValueError('The inlet pressure must match P_start')
    if not 0.0 < efficiency <= 1.0:
        raise ValueError('efficiency must be greater than 0 and at most 1')
    if not 0.0 <= pressure_drop_percent < 100.0:
        raise ValueError('pressure_drop_percent must be between 0 and 100')

    states = {}
    components = {}
    stage_inlet = inlet
    species = [sp.name for sp in inlet.species]
    cooler_pressure_ratio = 1.0 - pressure_drop_percent/100.0
    compressor_pressure_ratio = (
        P_end/(P_start*cooler_pressure_ratio**n_stages)
    )**(1.0/n_stages)

    for i in range(1, n_stages + 1):
        compressor_pressure = stage_inlet.P*compressor_pressure_ratio
        # Diatomic ideal-gas estimate used only to initialize the solver.
        T_iso_guess = stage_inlet.T*compressor_pressure_ratio**(2.0/7.0)
        compressor_temperature = (
            stage_inlet.T + (T_iso_guess - stage_inlet.T)/efficiency
        )

        compressor_out = State(
            name = f'{name} - Stage {i} compressor outlet',
            species = species,
            m_dot = stage_inlet.m_dot,
            T = compressor_temperature,
            P = compressor_pressure,
            z = stage_inlet.z,
            phase = 'vapor',
        )
        states[compressor_out.name] = compressor_out

        compressor = Compressor(
            name = f'{name} - Compressor {i}',
            efficiency = efficiency,
            pressure_ratio = compressor_pressure_ratio,
            inlet = stage_inlet,
            outlet = compressor_out,
        )
        components[compressor.name] = compressor

        cooler_name = (
            f'{name} outlet'
            if i == n_stages
            else f'{name} - Stage {i} cooler outlet'
        )
        cooler_out = State(
            name = cooler_name,
            species = species,
            m_dot = compressor_out.m_dot,
            T = cooling_temperature,
            P = compressor_pressure*cooler_pressure_ratio,
            z = compressor_out.z,
            phase = 'vapor',
        )
        states[cooler_out.name] = cooler_out

        cooler = Intercooler(
            name = f'{name} - Cooler {i}',
            T_out = cooling_temperature,
            pressure_drop_percent = pressure_drop_percent,
            inlet = compressor_out,
            outlet = cooler_out,
        )
        components[cooler.name] = cooler
        stage_inlet = cooler_out

    return states, components


def build_column(
    name,
    species,
    n_trays,
    P_top,
    P_bottom,
    L_top,
    L_bottom,
    V_top,
    V_bottom,
    x_top,
    x_bottom,
    feeds,
):
    """Create and connect equilibrium trays from top to bottom."""
    pressures = np.linspace(P_top, P_bottom, n_trays)
    liquid_flows = np.linspace(L_top, L_bottom, n_trays)
    vapor_flows = np.linspace(V_top, V_bottom, n_trays)
    liquid_compositions = np.linspace(x_top, x_bottom, n_trays)

    fluid_names = []
    for species_name in species:
        fluid_names.append(utils.SPS[species_name].fluid)
    equilibrium = CP.AbstractState('HEOS', '&'.join(fluid_names))

    temperatures = []
    vapor_compositions = []
    for i in range(n_trays):
        equilibrium.set_mole_fractions(liquid_compositions[i])
        equilibrium.update(CP.PQ_INPUTS, pressures[i], 0.0)
        temperatures.append(equilibrium.T())
        vapor_compositions.append(equilibrium.mole_fractions_vapor())
    states = {}
    components = {}
    liquid_states = []
    vapor_states = []

    for i in range(n_trays):
        tray_name = f'{name} - T{i + 1}'

        liquid = State(
            f'{tray_name} - liquid', species, liquid_flows[i],
            temperatures[i], pressures[i], liquid_compositions[i],
            phase='liquid',
        )
        states[liquid.name] = liquid
        liquid_states.append(liquid)

        vapor = State(
            f'{tray_name} - vapor', species, vapor_flows[i],
            temperatures[i], pressures[i], vapor_compositions[i],
            phase='vapor',
        )
        states[vapor.name] = vapor
        vapor_states.append(vapor)

    for i in range(n_trays):
        tray_name = f'{name} T{i + 1}'
        liquid_in = liquid_states[i - 1] if i > 0 else None
        vapor_in = vapor_states[i + 1] if i < n_trays - 1 else None

        tray = Tray(
            tray_name,
            pressures[i],
            liquid_in,
            vapor_in,
            liquid_states[i],
            vapor_states[i],
            feeds=feeds.get(i + 1, []),
        )
        components[tray.name] = tray

    return states, components
