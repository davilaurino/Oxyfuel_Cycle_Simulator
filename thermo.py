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

from common import utils

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

def calculate_pr_parameters(T, P, y, species):
    """Return pure-component and mixture parameters for the PR equation."""
    y = np.asarray(y, dtype=float)
    if len(y) != len(species):
        raise ValueError("Composition and species must have the same length")
    if np.any(y < 0) or y.sum() <= 0:
        raise ValueError("Composition must be nonnegative and have a positive sum")
    y = y/y.sum()

    a = np.empty(len(species))
    b = np.empty(len(species))
    da_dT = np.empty(len(species))

    for i, sp in enumerate(species):
        kappa = 0.37464 + 1.54226*sp.omega - 0.26992*sp.omega**2
        alpha_base = 1 + kappa*(1 - sqrt(T/sp.Tc))
        a_constant = PR_A*R**2*sp.Tc**2/sp.Pc

        a[i] = a_constant*alpha_base**2
        b[i] = PR_B*R*sp.Tc/sp.Pc
        da_dT[i] = -a_constant*kappa*alpha_base/sqrt(T*sp.Tc)

    aij = np.empty((len(species), len(species)))
    daij_dT = np.empty_like(aij)
    for i, sp_i in enumerate(species):
        for j, sp_j in enumerate(species):
            kij = KIJ.get(frozenset((sp_i.name, sp_j.name)), 0)
            aij[i, j] = (1 - kij)*sqrt(a[i]*a[j])
            daij_dT[i, j] = aij[i, j]*(
                da_dT[i]/a[i] + da_dT[j]/a[j]
            )/2

    a_mix = 0
    b_mix = 0
    da_mix_dT = 0
    for i in range(len(species)):
        b_mix += y[i]*b[i]
        for j in range(len(species)):
            a_mix += y[i]*y[j]*aij[i, j]
            da_mix_dT += y[i]*y[j]*daij_dT[i, j]

    return {
        'y': y,
        'a': a,
        'b': b,
        'da_dT': da_dT,
        'aij': aij,
        'a_mix': a_mix,
        'b_mix': b_mix,
        'da_mix_dT': da_mix_dT,
        'A': a_mix*P/(R**2*T**2),
        'B': b_mix*P/(R*T),
    }

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

    valid_roots.sort()
    if len(valid_roots) == 1:
        return (valid_roots[0],)

    # The middle root is mechanically unstable. Only the liquid-like minimum
    # and vapor-like maximum roots are useful for phase calculations.
    return valid_roots[0], valid_roots[-1]

def select_stable_pr_root(A, B, roots):
    """Select the physical PR root with the lowest residual Gibbs energy."""
    if len(roots) == 1:
        return roots[0]

    residual_gibbs = []
    for Z in roots:
        log_ratio = log(
            (Z + (1 + sqrt(2))*B)/(Z + (1 - sqrt(2))*B)
        )
        residual_gibbs.append(
            Z - 1 - log(Z - B)
            - A*log_ratio/(2*sqrt(2)*B)
        )

    return roots[residual_gibbs.index(min(residual_gibbs))]

def calculate_pr_excess(T, P, y, species):
    """Calculate PR excess enthalpy and entropy in J/kg and J/(kg K)."""
    pr = calculate_pr_parameters(T, P, y, species)
    y = pr['y']
    a = pr['a']
    b = pr['b']
    da_dT = pr['da_dT']
    a_mix = pr['a_mix']
    b_mix = pr['b_mix']
    da_mix_dT = pr['da_mix_dT']
    A = pr['A']
    B = pr['B']
    Z_mix = select_stable_pr_root(A, B, solve_pr_cubic(A, B))

    B_pure = []
    Z_pure = []
    H_res_pure = []
    S_res_pure = []
    for i in range(len(species)):
        Ai = a[i]*P/(R**2*T**2)
        Bi = b[i]*P/(R*T)
        Zi = select_stable_pr_root(Ai, Bi, solve_pr_cubic(Ai, Bi))

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

# =============================================================================
# ASU thermodynamic functions
# =============================================================================

def pr_fugacity_coefficients(T, P, y, species, phase):
    """Return PR fugacity coefficients for a liquid or vapor mixture."""
    if phase not in ('liquid', 'vapor'):
        raise ValueError("phase must be 'liquid' or 'vapor'")

    pr = calculate_pr_parameters(T, P, y, species)
    y = pr['y']
    a = pr['a']
    b = pr['b']
    aij = pr['aij']
    a_mix = pr['a_mix']
    b_mix = pr['b_mix']
    A = pr['A']
    B = pr['B']

    roots = solve_pr_cubic(A, B)
    Z = roots[0] if phase == 'liquid' else roots[-1]

    log_ratio = log(
        (Z + (1 + sqrt(2))*B)/(Z + (1 - sqrt(2))*B)
    )
    interaction_sum = np.zeros(len(species))
    for i in range(len(species)):
        for j in range(len(species)):
            interaction_sum[i] += aij[i, j]*y[j]

    ln_phi = (
        b/b_mix*(Z - 1)
        - np.log(Z - B)
        - A/(2*sqrt(2)*B)
        * (2*interaction_sum/a_mix - b/b_mix)
        * log_ratio
    )

    return np.exp(ln_phi)


def test_o2_n2_fugacity():
    """Compare PR and HEOS fugacities at an O2-N2 equilibrium state."""
    T = 100.0
    P = 6.0e5
    feed = [0.21, 0.79]
    species = utils.AIR_SPECIES
    fluid_string = '&'.join(sp.fluid for sp in species)

    flash = CP.AbstractState('HEOS', fluid_string)
    flash.set_mole_fractions(feed)
    flash.update(CP.PT_INPUTS, P, T)

    if flash.phase() != CP.iphase_twophase:
        raise AssertionError("The HEOS reference state is not two-phase")

    phase_data = {
        'liquid': (flash.mole_fractions_liquid(), CP.iphase_liquid),
        'vapor': (flash.mole_fractions_vapor(), CP.iphase_gas),
    }

    print(f'O2-N2 fugacity comparison at T = {T:.2f} K, P = {P/1e5:.2f} bar')
    print(f'HEOS vapor fraction = {flash.Q():.6f}')
    print(
        f"{'Phase':<8} {'Species':<8} {'z_i':>9} "
        f"{'phi HEOS':>12} {'phi PR':>12} {'error':>10} "
        f"{'f HEOS':>12} {'f PR':>12}"
    )

    maximum_relative_error = 0
    for phase, (composition, imposed_phase) in phase_data.items():
        heos = CP.AbstractState('HEOS', fluid_string)
        heos.set_mole_fractions(composition)
        heos.specify_phase(imposed_phase)
        heos.update(CP.PT_INPUTS, P, T)

        phi_heos = np.array([
            heos.fugacity_coefficient(i) for i in range(len(species))
        ])
        phi_pr = pr_fugacity_coefficients(T, P, composition, species, phase)

        for i, sp in enumerate(species):
            relative_error = (phi_pr[i] - phi_heos[i])/phi_heos[i]
            maximum_relative_error = max(
                maximum_relative_error, abs(relative_error)
            )
            fugacity_heos = composition[i]*phi_heos[i]*P
            fugacity_pr = composition[i]*phi_pr[i]*P
            print(
                f'{phase:<8} {sp.name:<8} {composition[i]:>9.6f} '
                f'{phi_heos[i]:>12.6f} {phi_pr[i]:>12.6f} '
                f'{100*relative_error:>9.3f}% '
                f'{fugacity_heos/1e5:>10.6f} bar '
                f'{fugacity_pr/1e5:>10.6f} bar'
            )

    if maximum_relative_error > 0.02:
        raise AssertionError(
            'PR fugacity coefficient differs from HEOS by more than 2%'
        )

    print(f'Maximum relative error: {100*maximum_relative_error:.3f}%')


if __name__ == '__main__':
    test_o2_n2_fugacity()
