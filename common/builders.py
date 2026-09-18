"""Reusable builders for process structures."""

import numpy as np
from common import utils

from common.classes import Compressor, Intercooler, State, Tray
from common.thermo import (
    bubble_point,
    new_heos,
    ps_gas_temperature,
    ps_temperature,
)

def build_compressor_train(
    name,
    inlet,
    P_end,
    cooling_temperature,
    n_stages,
    efficiency,
    pressure_drop_percent=0.0,
    phase='vapor',
    final_cooler_phase='vapor',
):
    """Create equal-ratio compression stages with cooling after each stage.

    The default gas mode preserves the ASU behavior.  A separate phase can be
    assigned to the final cooler outlet for dense-CO2 cycles.
    """
    if not isinstance(n_stages, (int, np.integer)) or n_stages < 1:
        raise ValueError('n_stages must be a positive integer')
    if P_end <= inlet.P:
        raise ValueError('P_end must be greater than inlet pressure')
    if not 0.0 < efficiency <= 1.0:
        raise ValueError('efficiency must be greater than 0 and at most 1')

    states = {}
    components = {}
    stage_inlet = inlet
    species = [sp.name for sp in inlet.species]
    thermo_state = new_heos(inlet.species)
    cooler_pressure_ratio = 1.0 - pressure_drop_percent/100.0
    compressor_pressure_ratio = (P_end/(inlet.P*cooler_pressure_ratio**n_stages))**(1.0/n_stages)

    for i in range(1, n_stages + 1):
        compressor_pressure = stage_inlet.P*compressor_pressure_ratio
        temperature_flash = (
            ps_gas_temperature if phase == 'vapor' else ps_temperature
        )
        T_iso_guess = temperature_flash(
            thermo_state, compressor_pressure, stage_inlet.s, stage_inlet.z,
        )
        compressor_temperature = (stage_inlet.T + (T_iso_guess - stage_inlet.T)/efficiency)

        compressor_out = State(
            name = f'{name} - Stage {i} compressor outlet',
            species = species,
            m_dot = stage_inlet.m_dot,
            T = compressor_temperature,
            P = compressor_pressure,
            z = stage_inlet.z,
            phase = phase,
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

        cooler_name = (f'{name} outlet' if i == n_stages else f'{name} - Stage {i} cooler outlet')
        cooler_phase = (
            final_cooler_phase
            if i == n_stages
            else phase
        )
        cooler_out = State(
            name = cooler_name,
            species = species,
            m_dot = compressor_out.m_dot,
            T = cooling_temperature,
            P = compressor_pressure*cooler_pressure_ratio,
            z = compressor_out.z,
            phase = cooler_phase,
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

    column_species = []
    for species_name in species:
        column_species.append(utils.SPS[species_name])
    thermo_state = new_heos(column_species)

    temperatures = []
    vapor_compositions = []
    for pressure, liquid_composition in zip(pressures, liquid_compositions):
        temperature, vapor_composition = bubble_point(thermo_state, pressure, liquid_composition)
        temperatures.append(temperature)
        vapor_compositions.append(vapor_composition)
        
    states = {}
    components = {}
    liquid_states = []
    vapor_states = []

    for i in range(n_trays):
        tray_name = f'{name} T{i + 1}'

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
