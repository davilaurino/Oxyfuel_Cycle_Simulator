from CoolProp.CoolProp import PropsSI
from dataclasses import dataclass
from scipy.constants import R
import numpy as np

MW_CO2  = PropsSI('M', 'CarbonDioxide')
MW_H2O  = PropsSI('M', 'Water')
MW_O2   = PropsSI('M', 'Oxygen')
MW_N2   = PropsSI('M', 'Nitrogen')
MW_AR   = PropsSI('M', 'Argon')
MW_CH4  = PropsSI('M', 'Methane')
MW_C2H6 = PropsSI('M', 'Ethane')
MW_C3H8 = PropsSI('M', 'Propane')

Tc_CO2 = PropsSI('Tcrit', 'PR::CarbonDioxide')
Tc_H2O = PropsSI('Tcrit', 'PR::Water')
Tc_O2  = PropsSI('Tcrit', 'PR::Oxygen')
Tc_N2  = PropsSI('Tcrit', 'PR::Nitrogen')
Tc_Ar = PropsSI('Tcrit', 'PR::Argon')
Tc_CH4 = PropsSI('Tcrit', 'PR::Methane')
Tc_C2H6 = PropsSI('Tcrit', 'PR::Ethane')
Tc_C3H8 = PropsSI('Tcrit', 'PR::Propane')

Pc_CO2 = PropsSI('Pcrit', 'PR::CarbonDioxide')
Pc_H2O = PropsSI('Pcrit', 'PR::Water')
Pc_O2  = PropsSI('Pcrit', 'PR::Oxygen')
Pc_N2  = PropsSI('Pcrit', 'PR::Nitrogen')
Pc_Ar = PropsSI('Pcrit', 'PR::Argon')
Pc_CH4 = PropsSI('Pcrit', 'PR::Methane')
Pc_C2H6 = PropsSI('Pcrit', 'PR::Ethane')
Pc_C3H8 = PropsSI('Pcrit', 'PR::Propane')

omega_CO2 = PropsSI('acentric', 'CarbonDioxide')
omega_H2O = PropsSI('acentric', 'Water')
omega_O2  = PropsSI('acentric', 'Oxygen')
omega_N2  = PropsSI('acentric', 'Nitrogen')
omega_Ar = PropsSI('acentric', 'Argon')
omega_CH4 = PropsSI('acentric', 'Methane')
omega_C2H6 = PropsSI('acentric', 'Ethane')
omega_C3H8 = PropsSI('acentric', 'Propane')

@dataclass(frozen=True, slots=True)
class Species:
    """Immutable thermodynamic and elemental data for one species."""

    name: str
    fluid: str
    MW: float
    Tc: float
    Pc: float
    omega: float
    LHV: float = 0.0
    n_C: int = 0
    n_H: int = 0

CO2  = Species('CO2',  'CarbonDioxide', MW_CO2,  Tc_CO2,  Pc_CO2,  omega_CO2)
H2O  = Species('H2O',  'Water',         MW_H2O,  Tc_H2O,  Pc_H2O,  omega_H2O)
O2   = Species('O2',   'Oxygen',        MW_O2,   Tc_O2,   Pc_O2,   omega_O2)
N2   = Species('N2',   'Nitrogen',      MW_N2,   Tc_N2,   Pc_N2,   omega_N2)
AR   = Species('AR',   'Argon',         MW_AR,   Tc_Ar,   Pc_Ar,   omega_Ar)
CH4  = Species('CH4',  'Methane',       MW_CH4,  Tc_CH4,  Pc_CH4,  omega_CH4, LHV=50.05e6, n_C=1, n_H=4)
C2H6 = Species('C2H6', 'Ethane',        MW_C2H6, Tc_C2H6, Pc_C2H6, omega_C2H6, LHV=47.52e6, n_C=2, n_H=6)
C3H8 = Species('C3H8', 'Propane',       MW_C3H8, Tc_C3H8, Pc_C3H8, omega_C3H8, LHV=46.34e6, n_C=3, n_H=8)

SPS = {
    'CO2': CO2,
    'H2O': H2O,
    'O2': O2,
    'N2': N2,
    'AR': AR,
    'CH4': CH4,
    'C2H6': C2H6,
    'C3H8': C3H8,
}

# Canonical species groups used by the legacy models.  The tuples contain the
# same singleton objects exposed above, so identity checks against SPS remain
# valid while the groups cannot be modified accidentally.
ALL_SPECIES = (CO2, H2O, O2, N2, AR, CH4, C2H6, C3H8)
PRODUCT_SPECIES = (CO2, H2O, O2, N2)
FUEL_SPECIES = (CH4, C2H6, C3H8, CO2)
AIR_SPECIES = (O2, N2)

def ylny(y):
    if (y>0):
        t = y*np.log(y)
    else:
        t = 0
    return t

def mass_fraction(y, species):
    pairs = list(zip(y, species))
    MW_mix = sum(yi * sp.MW for (yi, sp) in pairs)
    x = list((yi*sp.MW)/MW_mix for (yi, sp) in pairs)

    return MW_mix, x

def stoichiometry(S_fuel):
    if hasattr(S_fuel, 'species'):
        fuel_fractions = S_fuel.z
        fuel_species = S_fuel.species
    else:
        fuel_fractions = S_fuel.y
        fuel_species = S_fuel.spc

    MW_fuel, x_fuel = mass_fraction(fuel_fractions, fuel_species)
    n_fuel = S_fuel.m_dot/MW_fuel

    n_C = 0
    n_H = 0
    for (spc, yi) in zip(fuel_species, fuel_fractions):
        n_C += spc.n_C*yi*n_fuel
        n_H += spc.n_H*yi*n_fuel

    return MW_fuel, n_fuel, n_C, n_H

def fuel_requirements(S_fuel):
    """Return stoichiometric O2 mass and molar flow rates."""
    MW_fuel, n_fuel, n_C, n_H = stoichiometry(S_fuel)
    n_O2_sto = n_C + n_H/4
    m_O2_sto = n_O2_sto*SPS['O2'].MW

    return m_O2_sto, n_O2_sto

def oxygen_separation_work(p, work_95=720e3):
    """Cryogenic oxygen-separation work in J/kg_O2 at mole fraction p."""
    purity = 100*p

    # CMU/IECM piecewise purity correction, normalized to 95 mol% O2.
    if purity <= 97.5:
        work_multiplier = (
            3.0e-5*purity**2 + 1.7e-3*purity + 0.5923
        )/1.02455
    else:
        work_multiplier = (
            -0.0457*purity**2 + 9.1372*purity - 455.82
        )/0.62318

    return work_95*work_multiplier
