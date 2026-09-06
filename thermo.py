"""Thermodynamic properties for fluid mixtures.

Pure-fluid properties are obtained from HEOS through CoolProp. Peng--Robinson
excess-property corrections will be added to account for non-ideal mixing.
"""

import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from scipy.constants import R
from scipy.optimize import minimize_scalar
from math import log, sqrt
import numpy as np

import utils

KIJ = {
    frozenset(('CO2', 'H2O')): 0.152780,
    frozenset(('CO2', 'O2')): 0.160723,
    frozenset(('CO2', 'N2')): 0.054330,
    frozenset(('O2', 'N2')): -0.009991,
    frozenset(('CH4', 'C2H6')): 0.004868,
    frozenset(('CH4', 'C3H8')): 0.024310,
    frozenset(('CH4', 'CO2')): 0.093698,
}

PR_A = 0.4572355289213822
PR_B = 0.0777960739038846

def mixture_critical_point(y, species):
    """Return the maximum stable HEOS critical temperature and pressure."""
    fractions = [yi for yi in y if yi > 1e-5]
    fluids = [sp.fluid for yi, sp in zip(y, species) if yi > 1e-5]

    if len(fluids) == 1:
        return (
            PropsSI('Tcrit', f'HEOS::{fluids[0]}'),
            PropsSI('Pcrit', f'HEOS::{fluids[0]}'),
        )

    state = CP.AbstractState('HEOS', '&'.join(fluids))
    state.set_mole_fractions(fractions)

    critical_points = [
        point for point in state.all_critical_points() if point.stable
    ]

    if not critical_points:
        return np.nan, np.nan

    Tc_mix = max(point.T for point in critical_points)
    Pc_mix = max(point.p for point in critical_points)
    return Tc_mix, Pc_mix

def pseudo_critical_temperature(P, y, species):
    """Return the HEOS constant-pressure heat-capacity maximum in K."""
    fractions = [yi for yi in y if yi > 1e-5]
    fluids = [sp.fluid for yi, sp in zip(y, species) if yi > 1e-5]
    Tc_mix, _ = mixture_critical_point(y, species)

    state = CP.AbstractState('HEOS', '&'.join(fluids))
    state.set_mole_fractions(fractions)

    def negative_cp(T):
        state.update(CP.PT_INPUTS, P, T)
        return -state.cpmass()

    result = minimize_scalar(
        negative_cp,
        bounds=(Tc_mix*(1 + 1e-6), 1.5*Tc_mix),
        method='bounded',
    )
    return result.x

def mixture_phase(T, P, y, species):
    """Return the HEOS mixture phase, critical limits, and density."""
    fractions = [yi for yi in y if yi > 1e-5]
    fluids = [sp.fluid for yi, sp in zip(y, species) if yi > 1e-5]
    Tc_mix, Pc_mix = mixture_critical_point(y, species)

    state = CP.AbstractState('HEOS', '&'.join(fluids))
    state.set_mole_fractions(fractions)
    state.update(CP.PT_INPUTS, P, T)

    if state.phase() == CP.iphase_twophase:
        phase = 'two-phase'
    elif np.isfinite(Tc_mix) and T > Tc_mix and P > Pc_mix:
        T_pseudo = pseudo_critical_temperature(P, y, species)
        if T < T_pseudo:
            phase = 'supercritical liquid-like'
        else:
            phase = 'supercritical gas-like'
    elif state.phase() in (CP.iphase_liquid, CP.iphase_supercritical_liquid):
        phase = 'liquid'
    else:
        phase = 'gas'

    rho_mix = state.rhomass()
    return phase, Tc_mix, Pc_mix, rho_mix

def heos_mixture_properties(T, P, y, species, phase):
    """Return phase-imposed HEOS h and s in the project's reference."""
    active = [(yi, sp) for yi, sp in zip(y, species) if yi > 1e-5]
    y_total = sum(yi for yi, _ in active)
    fractions = [yi/y_total for yi, _ in active]
    fluids = [sp.fluid for _, sp in active]
    active_species = [sp for _, sp in active]

    state = CP.AbstractState('HEOS', '&'.join(fluids))
    state.set_mole_fractions(fractions)

    if phase == 'liquid':
        state.specify_phase(CP.iphase_liquid)
    elif phase == 'gas':
        state.specify_phase(CP.iphase_gas)
    elif phase in ('supercritical liquid-like', 'supercritical gas-like'):
        state.specify_phase(CP.iphase_supercritical)

    state.update(CP.PT_INPUTS, P, T)

    _, x = utils.mass_fraction(fractions, active_species)
    h = state.hmass() + sum(
        xi*(sp.h_form - sp.h_ref) for xi, sp in zip(x, active_species)
    )
    s = state.smass() - sum(
        xi*sp.s_ref for xi, sp in zip(x, active_species)
    )

    return h, s

def solve_pr_cubic(A, B):
    coefficients = [
        1,
        B - 1,
        A - 3*B**2 - 2*B,
        B**3 + B**2 - A*B,
    ]

    roots = np.roots(coefficients)

    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-8 and root.real > B:
            valid_roots.append(root.real)

    if len(valid_roots) == 0:
        raise ValueError("No physical root found for the PR cubic")

    if len(valid_roots) == 1:
        return valid_roots[0]

    residual_gibbs = []
    for Z in valid_roots:
        log_ratio = log((Z + (1 + sqrt(2))*B)/(Z + (1 - sqrt(2))*B))
        residual_gibbs.append(Z - 1 - log(Z - B) - A*log_ratio/(2*sqrt(2)*B))

    Z = valid_roots[residual_gibbs.index(min(residual_gibbs))]
    
    return Z

def calculate_pr_excess(T, P, y, species):
    """Calculate PR excess enthalpy and entropy in J/kg and J/(kg K)."""
    y_total = sum(y)
    y = [yi/y_total for yi in y]

    a = []
    b = []
    da_dT = []

    for sp in species:
        kappa = 0.37464 + 1.54226*sp.omega - 0.26992*sp.omega**2
        alpha_base = 1 + kappa*(1 - sqrt(T/sp.Tc))
        alpha = alpha_base**2
        a_constant = PR_A*R**2*sp.Tc**2/sp.Pc

        a.append(a_constant*alpha)
        b.append(PR_B*R*sp.Tc/sp.Pc)
        da_dT.append(-a_constant*kappa*alpha_base/sqrt(T*sp.Tc))

    a_mix = 0
    da_mix_dT = 0
    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            kij = KIJ.get(frozenset((sp_i.name, sp_j.name)), 0)
            aij = (1 - kij)*sqrt(a[i]*a[j])
            a_mix += y[i]*y[j]*aij

            daij_dT = aij*(da_dT[i]/a[i] + da_dT[j]/a[j])/2
            da_mix_dT += y[i]*y[j]*daij_dT

    b_mix = 0
    for i, sp_i in enumerate(species):
        b_mix += y[i]*b[i]

    A = a_mix*P/(R**2*T**2)
    B = b_mix*P/(R*T)
    Z_mix = solve_pr_cubic(A, B)

    B_pure = []
    Z_pure = []
    H_res_pure = []
    S_res_pure = []
    for i in range(len(species)):
        Ai = a[i]*P/(R**2*T**2)
        Bi = b[i]*P/(R*T)
        Zi = solve_pr_cubic(Ai, Bi)

        B_pure.append(Bi)
        Z_pure.append(Zi)

        log_ratio_i = log((Zi + (1 + sqrt(2))*Bi)/(Zi + (1 - sqrt(2))*Bi))
        H_res_i = R*T*(Zi - 1) + (T*da_dT[i] - a[i])*log_ratio_i/(2*sqrt(2)*b[i])
        S_res_i = R*log(Zi - Bi) + da_dT[i]*log_ratio_i/(2*sqrt(2)*b[i])
        H_res_pure.append(H_res_i)
        S_res_pure.append(S_res_i)

    log_ratio = log((Z_mix + (1 + sqrt(2))*B)/(Z_mix + (1 - sqrt(2))*B))
    H_res_mix = R*T*(Z_mix - 1) + (T*da_mix_dT - a_mix)*log_ratio/(2*sqrt(2)*b_mix)
    S_res_mix = R*log(Z_mix - B) + da_mix_dT*log_ratio/(2*sqrt(2)*b_mix)

    H_excess = H_res_mix
    S_excess = S_res_mix
    for i in range(len(species)):
        H_excess -= y[i]*H_res_pure[i]
        S_excess -= y[i]*S_res_pure[i]

    MW_mix = sum(yi*sp.MW for yi, sp in zip(y, species))
    h_excess = H_excess/MW_mix
    s_excess = S_excess/MW_mix

    return h_excess, s_excess

def mixture_enthalpy(T, P, y, species, include_excess=True):
    MW, x = utils.mass_fraction(y, species)
    pairs = list(zip(x, species))
    h = 0
    for (xi, sp) in pairs:
        h += xi*(PropsSI('H', 'P', P, 'T', T, sp.fluid) - sp.h_ref + sp.h_form)

    if include_excess:
        h_excess, _ = calculate_pr_excess(T, P, y, species)
        h += 1*h_excess

    return h

def mixture_entropy(T, P, y, species, include_excess=True):
    MW, x = utils.mass_fraction(y, species)
    pairs = list(zip(x, species))

    t = sum(utils.ylny(yi) for yi in y)
    s_mix = (-R*t)/MW

    s = s_mix
    for (xi, sp) in pairs:
        s += xi*(PropsSI('S', 'P', P, 'T', T, sp.fluid) - sp.s_ref)

    if include_excess:
        _, s_excess = calculate_pr_excess(T, P, y, species)
        s += 1*s_excess

    return s

if __name__ == '__main__':
    T_test = 1200
    P_test = 30e6
    y_test = [0.98, 0.02]
    species_test = [utils.SPS['CO2'], utils.SPS['N2']]
    MW_test, x_test = utils.mass_fraction(y_test, species_test)

    h_excess_pr, s_excess_pr = calculate_pr_excess(
        T_test, P_test, y_test, species_test
    )

    h_pr = h_excess_pr
    for xi, sp in zip(x_test, species_test):
        h_ref_pr = PropsSI(
            'H', 'T', utils.T_ref, 'P', utils.P_ref, f'PR::{sp.fluid}'
        )
        h_pure_pr = PropsSI('H', 'T', T_test, 'P', P_test, f'PR::{sp.fluid}')
        h_pr += xi*(h_pure_pr - h_ref_pr + sp.h_form)

    fluid_heos = 'HEOS::' + '&'.join(
        f'{sp.fluid}[{yi}]' for yi, sp in zip(y_test, species_test)
    )
    h_heos = PropsSI('H', 'T', T_test, 'P', P_test, fluid_heos)
    h_heos += sum(
        xi*(sp.h_form - sp.h_ref) for xi, sp in zip(x_test, species_test)
    )

    h_mass_weighted = 0
    for xi, sp in zip(x_test, species_test):
        h_pure_heos = PropsSI('H', 'T', T_test, 'P', P_test, sp.fluid)
        h_mass_weighted += xi*(h_pure_heos - sp.h_ref + sp.h_form)

    h_heos_pr = mixture_enthalpy(T_test, P_test, y_test, species_test)

    s_mix_ideal = -R*sum(utils.ylny(yi) for yi in y_test)/MW_test

    s_pr = s_mix_ideal + s_excess_pr
    for xi, sp in zip(x_test, species_test):
        s_ref_pr = PropsSI(
            'S', 'T', utils.T_ref, 'P', utils.P_ref, f'PR::{sp.fluid}'
        )
        s_pure_pr = PropsSI('S', 'T', T_test, 'P', P_test, f'PR::{sp.fluid}')
        s_pr += xi*(s_pure_pr - s_ref_pr)

    s_heos = PropsSI('S', 'T', T_test, 'P', P_test, fluid_heos)
    s_heos -= sum(xi*sp.s_ref for xi, sp in zip(x_test, species_test))

    s_mass_weighted = s_mix_ideal
    for xi, sp in zip(x_test, species_test):
        s_pure_heos = PropsSI('S', 'T', T_test, 'P', P_test, sp.fluid)
        s_mass_weighted += xi*(s_pure_heos - sp.s_ref)

    s_heos_pr = mixture_entropy(T_test, P_test, y_test, species_test)

    print(f'T = {T_test:.2f} K, P = {P_test/1e6:.2f} MPa')
    print(f'Mass-weighted HEOS:   {h_mass_weighted:.6f} J/kg')
    print(f'Our PR:               {h_pr:.6f} J/kg')
    print(f'HEOS mixture:         {h_heos:.6f} J/kg')
    print(f'HEOS + PR excess:     {h_heos_pr:.6f} J/kg')
    print(f'PR excess:            {h_excess_pr:.6f} J/kg')
    print(f'Mass-weighted error:  {h_mass_weighted - h_heos:.6f} J/kg')
    print(f'PR-adjusted error:    {h_heos_pr - h_heos:.6f} J/kg')

    print(f'T = {T_test:.2f} K, P = {P_test/1e6:.2f} MPa')
    print(f'Mass-weighted HEOS:   {s_mass_weighted:.6f} J/(kg K)')
    print(f'Our PR:               {s_pr:.6f} J/(kg K)')
    print(f'HEOS mixture:         {s_heos:.6f} J/(kg K)')
    print(f'HEOS + PR excess:     {s_heos_pr:.6f} J/(kg K)')
    print(f'PR excess:            {s_excess_pr:.6f} J/(kg K)')
    print(f'Mass-weighted error:  {s_mass_weighted - s_heos:.6f} J/(kg K)')
    print(f'PR-adjusted error:    {s_heos_pr - s_heos:.6f} J/(kg K)')
