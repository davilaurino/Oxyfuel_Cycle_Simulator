from CoolProp.CoolProp import PropsSI
from scipy.constants import R
import numpy as np

T_ref = 298.15
P_ref = 101325

# Van Wylen LHV (J/kg)
LHV = {
    'CH4': 50.05e6,
    'C2H6': 47.52e6,
    'C3H8': 46.34e6,
}

MW_CO2  = PropsSI('M', 'CarbonDioxide')
MW_H2O  = PropsSI('M', 'Water')
MW_O2   = PropsSI('M', 'Oxygen')
MW_N2   = PropsSI('M', 'Nitrogen')
MW_AR   = PropsSI('M', 'Argon')
MW_CH4  = PropsSI('M', 'Methane')
MW_C2H6 = PropsSI('M', 'Ethane')
MW_C3H8 = PropsSI('M', 'Propane')

h_ref_CO2 = PropsSI('H', 'P', P_ref, 'T', T_ref, 'CarbonDioxide')
h_ref_H2O = PropsSI('H', 'P', P_ref, 'T', T_ref, 'Water')
h_ref_O2  = PropsSI('H', 'P', P_ref, 'T', T_ref, 'Oxygen')
h_ref_N2  = PropsSI('H', 'P', P_ref, 'T', T_ref, 'Nitrogen')
h_ref_AR  = PropsSI('H', 'P', P_ref, 'T', T_ref, 'Argon')
h_ref_CH4  = PropsSI('H', 'P', P_ref, 'T', T_ref, 'Methane')
h_ref_C2H6 = PropsSI('H', 'P', P_ref, 'T', T_ref, 'Ethane')
h_ref_C3H8 = PropsSI('H', 'P', P_ref, 'T', T_ref, 'Propane')

s_ref_CO2 = PropsSI('S', 'P', P_ref, 'T', T_ref, 'CarbonDioxide')
s_ref_H2O = PropsSI('S', 'P', P_ref, 'T', T_ref, 'Water')
s_ref_O2  = PropsSI('S', 'P', P_ref, 'T', T_ref, 'Oxygen')
s_ref_N2  = PropsSI('S', 'P', P_ref, 'T', T_ref, 'Nitrogen')
s_ref_AR  = PropsSI('S', 'P', P_ref, 'T', T_ref, 'Argon')
s_ref_CH4  = PropsSI('S', 'P', P_ref, 'T', T_ref, 'Methane')
s_ref_C2H6 = PropsSI('S', 'P', P_ref, 'T', T_ref, 'Ethane')
s_ref_C3H8 = PropsSI('S', 'P', P_ref, 'T', T_ref, 'Propane')

# NIST Formation Enthalpies
h_form_CH4  = -74600/MW_CH4
h_form_C2H6 = -83800/MW_C2H6
h_form_C3H8 = -104700/MW_C3H8
h_form_CO2  = -393520/MW_CO2
h_form_H2O  = -285830/MW_H2O
h_form_O2   = 0
h_form_N2   = 0
h_form_AR   = 0

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

class Species:
    def __init__(self, name, fluid, MW, h_ref, h_form, s_ref, Tc, Pc, omega, n_C=0, n_H=0):
        self.name = name
        self.fluid = fluid
        self.MW = MW
        self.h_ref = h_ref
        self.h_form = h_form
        self.s_ref = s_ref

        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega

        self.n_C = n_C
        self.n_H = n_H

SPS = {
    'CO2':  Species('CO2',  'CarbonDioxide', MW_CO2,  h_ref_CO2,  h_form_CO2,  s_ref_CO2,  Tc_CO2,  Pc_CO2,  omega_CO2),
    'H2O':  Species('H2O',  'Water',         MW_H2O,  h_ref_H2O,  h_form_H2O,  s_ref_H2O,  Tc_H2O,  Pc_H2O,  omega_H2O),
    'O2':   Species('O2',   'Oxygen',        MW_O2,   h_ref_O2,   h_form_O2,   s_ref_O2,   Tc_O2,   Pc_O2,   omega_O2),
    'N2':   Species('N2',   'Nitrogen',      MW_N2,   h_ref_N2,   h_form_N2,   s_ref_N2,   Tc_N2,   Pc_N2,   omega_N2),
    'AR':   Species('AR',   'Argon',         MW_AR,   h_ref_AR,   h_form_AR,   s_ref_AR,   Tc_Ar,   Pc_Ar,   omega_Ar),
    'CH4':  Species('CH4',  'Methane',       MW_CH4,  h_ref_CH4,  h_form_CH4,  s_ref_CH4,  Tc_CH4,  Pc_CH4,  omega_CH4,  n_C=1, n_H=4),
    'C2H6': Species('C2H6', 'Ethane',        MW_C2H6, h_ref_C2H6, h_form_C2H6, s_ref_C2H6, Tc_C2H6, Pc_C2H6, omega_C2H6, n_C=2, n_H=6),
    'C3H8': Species('C3H8', 'Propane',       MW_C3H8, h_ref_C3H8, h_form_C3H8, s_ref_C3H8, Tc_C3H8, Pc_C3H8, omega_C3H8, n_C=3, n_H=8),
}

PRODUCT_SPECIES = [SPS['CO2'], SPS['H2O'], SPS['O2'], SPS['N2']]
FUEL_SPECIES = [SPS['CH4'], SPS['C2H6'], SPS['C3H8'], SPS['CO2']]
AIR_SPECIES = [SPS['O2'], SPS['N2']]

def ylny(y):
    if (y>0):
        t = y*np.log(y)
    else:
        t = 0
    return t

def flow_exergy(m_dot, h, s, T0=T_ref):
    B = m_dot*(h - T0*s)
    
    return B

def y_H2O_sat(T, P):
    P_sat = PropsSI('P', 'T', T, 'Q', 0, 'Water')
    y = P_sat/P
    return y

def CO2_volume(T, P):
    d = PropsSI('D', 'P', P, 'T', T, 'CarbonDioxide')
    v = 1/d
    return v

def mass_fraction(y, species):
    pairs = list(zip(y, species))
    MW_mix = sum(yi * sp.MW for (yi, sp) in pairs)
    x = list((yi*sp.MW)/MW_mix for (yi, sp) in pairs)

    return MW_mix, x

def stoichiometry(S_fuel):
    MW_fuel, x_fuel = mass_fraction(S_fuel.y, S_fuel.spc)
    n_fuel = S_fuel.m_dot/MW_fuel

    n_C = 0
    n_H = 0
    for (spc, yi) in zip(S_fuel.spc, S_fuel.y):
        n_C += spc.n_C*yi*n_fuel
        n_H += spc.n_H*yi*n_fuel

    return MW_fuel, n_fuel, n_C, n_H

def fuel_requirements(S_fuel):
    MW_fuel, n_fuel, n_C, n_H = stoichiometry(S_fuel)
    n_O2_sto = n_C + n_H/4
    m_O2_sto = n_O2_sto*SPS['O2'].MW

    n_CO2_fuel = 0
    for spc, yi in zip(S_fuel.spc, S_fuel.y):
            if spc is SPS['CO2']:
                n_CO2_fuel += yi*n_fuel
        
    m_CO2_add = (n_C + n_CO2_fuel)*SPS['CO2'].MW
    m_H2O_add = (n_H/2)*SPS['H2O'].MW

    return m_CO2_add, m_H2O_add, m_O2_sto

def asu_work(p, work_95=720e3):
    """Cryogenic-ASU specific work in J/kg_O2 for oxygen mole fraction p."""
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
