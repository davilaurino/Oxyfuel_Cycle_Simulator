"""CoolProp helpers for process models."""

import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import brentq

from common import utils


T_REF = 298.15  # K
P_REF = 101325  # Pa

# Standard molar formation enthalpies at 298.15 K and 101325 Pa (J/mol).
FORMATION_ENTHALPIES = {
    'CO2': -393520.0,
    'H2O': -285830.0,
    'O2': 0.0,
    'N2': 0.0,
    'AR': 0.0,
    'CH4': -74600.0,
    'C2H6': -83800.0,
    'C3H8': -104700.0,
}


def _configure_reference_states():
    """Set a common CoolProp reference before creating any HEOS states."""
    for species in utils.ALL_SPECIES:
        CP.set_reference_state(species.fluid, 'DEF')

    for species in utils.ALL_SPECIES:
        molar_density = CP.PropsSI(
            'Dmolar',
            'T', T_REF,
            'P', P_REF,
            species.fluid,
        )
        CP.set_reference_state(
            species.fluid,
            T_REF,
            molar_density,
            FORMATION_ENTHALPIES[species.name],
            0.0,
        )


_configure_reference_states()


PHASE_NAMES = {
    CP.iphase_liquid: 'liquid',
    CP.iphase_gas: 'vapor',
    CP.iphase_twophase: 'two-phase',
    CP.iphase_supercritical: 'supercritical',
    CP.iphase_supercritical_liquid: 'supercritical liquid',
    CP.iphase_supercritical_gas: 'supercritical vapor',
    CP.iphase_critical_point: 'critical point',
    CP.iphase_unknown: 'unknown',
}


class CoolPropFailure(RuntimeError):
    """A CoolProp calculation failed for the current trial state."""


def new_heos(species):
    """Create a reusable HEOS state for a list of species."""
    fluids = '&'.join(sp.fluid for sp in species)
    return CP.AbstractState('HEOS', fluids)


def bubble_point(thermo_state, P, z):
    """Return saturated-liquid temperature and equilibrium vapor composition."""
    try:
        thermo_state.set_mole_fractions(z)
        thermo_state.update(CP.PQ_INPUTS, P, 0.0)

        temperature = thermo_state.T()
        vapor_composition = np.asarray(thermo_state.mole_fractions_vapor(), dtype=float)
        return temperature, vapor_composition
    
    except ValueError as error:
        print(f'CoolProp failure: {error}')
        raise CoolPropFailure


def ps_gas_enthalpy(thermo_state, P, s, z):
    """Return gas-phase enthalpy at the specified pressure and entropy."""
    try:
        thermo_state.set_mole_fractions(z)
        thermo_state.unspecify_phase()
        thermo_state.specify_phase(CP.iphase_gas)
        thermo_state.update(CP.PSmass_INPUTS, P, s)
        return thermo_state.hmass()
    except ValueError as error:
        print(f'CoolProp failure: {error}')
        raise CoolPropFailure


def ps_gas_temperature(thermo_state, P, s, z):
    """Return gas-phase temperature at the specified pressure and entropy."""
    try:
        thermo_state.set_mole_fractions(z)
        thermo_state.unspecify_phase()
        thermo_state.specify_phase(CP.iphase_gas)
        thermo_state.update(CP.PSmass_INPUTS, P, s)
        return thermo_state.T()
    except ValueError as error:
        print(f'CoolProp failure: {error}')
        raise CoolPropFailure


def ps_enthalpy(thermo_state, P, s, z):
    """Return enthalpy at pressure and entropy without imposing a phase.

    This is useful for dense CO2 cycle streams, which can cross the critical
    region during compression and expansion and therefore cannot reliably be
    forced into the gas or liquid phase.
    """
    try:
        thermo_state.set_mole_fractions(z)
        thermo_state.unspecify_phase()
        thermo_state.update(CP.PSmass_INPUTS, P, s)
        return thermo_state.hmass()
    except ValueError as error:
        print(f'CoolProp failure: {error}')
        raise CoolPropFailure


def ps_temperature(thermo_state, P, s, z):
    """Return temperature at pressure and entropy without imposing a phase."""
    try:
        thermo_state.set_mole_fractions(z)
        thermo_state.unspecify_phase()
        thermo_state.update(CP.PSmass_INPUTS, P, s)
        return thermo_state.T()
    except ValueError as error:
        print(f'CoolProp failure: {error}')
        raise CoolPropFailure


def ps_liquid_enthalpy(thermo_state, P, s, z):
    """Return liquid-phase enthalpy at the specified pressure and entropy."""
    try:
        thermo_state.set_mole_fractions(z)
        thermo_state.unspecify_phase()
        thermo_state.specify_phase(CP.iphase_liquid)
        thermo_state.update(CP.PSmass_INPUTS, P, s)
        return thermo_state.hmass()
    except ValueError as error:
        print(f'CoolProp failure: {error}')
        raise CoolPropFailure


def ps_flash_enthalpy(thermo_state, P, s, z):
    """Return phase-aware enthalpy at the specified pressure and entropy."""
    try:
        thermo_state.set_mole_fractions(z)
        thermo_state.unspecify_phase()
        thermo_state.update(CP.PQ_INPUTS, P, 1.0)
        s_dew = thermo_state.smass()

        if s >= s_dew:
            thermo_state.specify_phase(CP.iphase_gas)
            thermo_state.update(CP.PSmass_INPUTS, P, s)
            return thermo_state.hmass()

        def entropy_residual(Q):
            thermo_state.unspecify_phase()
            thermo_state.update(CP.PQ_INPUTS, P, Q)
            return thermo_state.smass() - s

        Q = brentq(entropy_residual, 0.0, 1.0)
        thermo_state.unspecify_phase()
        thermo_state.update(CP.PQ_INPUTS, P, Q)
        return thermo_state.hmass()

    except ValueError as error:
        print(f'CoolProp failure: {error}')
        raise CoolPropFailure


def update_state(thermo_state, species, T, P, z, imposed_phase):
    """Calculate state properties using a reusable CoolProp state.

    Returns ``(phase, h, s, molar_mass, w, vapor_fraction, phi)``.
    """
    try:
        thermo_state.set_mole_fractions(z)
        if imposed_phase == 'liquid':
            thermo_state.specify_phase(CP.iphase_liquid)
        elif imposed_phase == 'vapor':
            thermo_state.specify_phase(CP.iphase_gas)
        else:
            thermo_state.unspecify_phase()

        thermo_state.update(CP.PT_INPUTS, P, T)

        phase = PHASE_NAMES.get(thermo_state.phase(), 'unknown')
        h = thermo_state.hmass()
        s = thermo_state.smass()
        molar_mass = thermo_state.molar_mass()

        w = np.zeros(len(species))
        for i, (z_i, sp) in enumerate(zip(z, species)):
            w[i] = z_i*sp.MW/molar_mass

        vapor_fraction = thermo_state.Q()
        phi = None
        if imposed_phase in ('liquid', 'vapor'):
            phi = np.zeros(len(species))
            for i in range(len(species)):
                phi[i] = thermo_state.fugacity_coefficient(i)

        return phase, h, s, molar_mass, w, vapor_fraction, phi
    
    except ValueError as error:
        print(f'CoolProp failure: {error}')
        raise CoolPropFailure
