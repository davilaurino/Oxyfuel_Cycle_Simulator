"""Fit constant CO2-X Peng--Robinson interaction coefficients to HEOS."""

import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from scipy.constants import R
from scipy.optimize import minimize_scalar

import thermo
from common import utils


TP_PAIRS = [
    (300, 8e6, CP.iphase_liquid),
    (336, 30e6, CP.iphase_supercritical),
    (473, 29.85e6, CP.iphase_supercritical),
    (700, 29.85e6, CP.iphase_supercritical),
    (1000, 3e6, CP.iphase_supercritical_gas),
    (1100, 6.5e6, CP.iphase_supercritical_gas),
    (1250, 14e6, CP.iphase_supercritical),
    (1400, 30e6, CP.iphase_supercritical),
    (1700, 30e6, CP.iphase_supercritical),
]
MAJOR_SPECIES_FRACTIONS = [0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
CO2_OTHER_SPECIES = [utils.SPS['H2O'], utils.SPS['O2'], utils.SPS['N2']]

FUEL_TP_PAIRS = [
    (298.15, 0.7e6, CP.iphase_gas),
    (550, 30e6, CP.iphase_supercritical_gas),
    (600, 30e6, CP.iphase_supercritical_gas),
    (650, 30e6, CP.iphase_supercritical_gas),
    (700, 30e6, CP.iphase_supercritical_gas),
]
FUEL_OTHER_SPECIES = [
    utils.SPS['C2H6'], utils.SPS['C3H8'], utils.SPS['CO2']
]

OXIDANT_TP_PAIRS = [
    (280, 30e6, CP.iphase_supercritical),
    (300, 30e6, CP.iphase_supercritical),
    (320, 30e6, CP.iphase_supercritical),
    (350, 30e6, CP.iphase_supercritical),
]


def heos_excess(T, P, y, species, phase):
    MW_mix, x = utils.mass_fraction(y, species)
    state = CP.AbstractState('HEOS', '&'.join(sp.fluid for sp in species))
    state.set_mole_fractions(y)
    state.specify_phase(phase)
    state.update(CP.PT_INPUTS, P, T)

    h_mix = state.hmass()
    s_mix = state.smass()

    h_pure = 0
    s_pure = 0
    for xi, sp in zip(x, species):
        h_pure += xi*PropsSI('H', 'T', T, 'P', P, sp.fluid)
        s_pure += xi*PropsSI('S', 'T', T, 'P', P, sp.fluid)

    s_ideal_mix = -R*sum(utils.ylny(yi) for yi in y)/MW_mix
    return h_mix - h_pure, s_mix - s_pure - s_ideal_mix


def reference_points(species, tp_pairs, first_fractions):
    points = []
    for T, P, phase in tp_pairs:
        for first_fraction in first_fractions:
            y = [first_fraction, 1 - first_fraction]
            try:
                h_heos, s_heos = heos_excess(T, P, y, species, phase)
            except ValueError:
                continue
            points.append((T, P, y, h_heos, s_heos))
    return points


def property_errors(kij, pair, species, points):
    thermo.KIJ[pair] = kij
    h_errors = []
    h_errors_over_T = []
    s_errors = []

    for T, P, y, h_heos, s_heos in points:
        h_pr, s_pr = thermo.calculate_pr_excess(T, P, y, species)
        h_errors.append(h_pr - h_heos)
        h_errors_over_T.append((h_pr - h_heos)/T)
        s_errors.append(s_pr - s_heos)

    return (
        np.asarray(h_errors),
        np.asarray(h_errors_over_T),
        np.asarray(s_errors),
    )


def fit_interaction(first_species, second_species, tp_pairs,
                    first_fractions, bounds):
    species = [first_species, second_species]
    pair = frozenset((first_species.name, second_species.name))
    points = reference_points(species, tp_pairs, first_fractions)
    original_kij = thermo.KIJ.get(pair)

    def combined_rmse(kij):
        _, h_errors_over_T, s_errors = property_errors(
            kij, pair, species, points
        )
        return np.sqrt(np.mean(
            0.5*h_errors_over_T**2 + 0.5*s_errors**2
        ))

    try:
        baseline_score = combined_rmse(0)
        baseline_h_errors, _, baseline_s_errors = property_errors(
            0, pair, species, points
        )
        baseline_h_rmse = np.sqrt(np.mean(baseline_h_errors**2))
        baseline_s_rmse = np.sqrt(np.mean(baseline_s_errors**2))
        baseline_h_max = np.max(np.abs(baseline_h_errors))
        baseline_s_max = np.max(np.abs(baseline_s_errors))

        result = minimize_scalar(
            combined_rmse,
            bounds=bounds,
            method='bounded',
            options={'xatol': 1e-6},
        )

        h_errors, _, s_errors = property_errors(
            result.x, pair, species, points
        )
        h_rmse = np.sqrt(np.mean(h_errors**2))
        s_rmse = np.sqrt(np.mean(s_errors**2))
        h_max = np.max(np.abs(h_errors))
        s_max = np.max(np.abs(s_errors))
    finally:
        if original_kij is None:
            thermo.KIJ.pop(pair, None)
        else:
            thermo.KIJ[pair] = original_kij

    return (
        result.x, result.fun, h_rmse, s_rmse, h_max, s_max,
        baseline_score, baseline_h_rmse, baseline_s_rmse,
        baseline_h_max, baseline_s_max, len(points),
    )


def print_result(label, result):
    (
        kij, score, h_rmse, s_rmse, h_max, s_max,
        baseline_score, baseline_h_rmse, baseline_s_rmse,
        baseline_h_max, baseline_s_max, points,
    ) = result

    score_improvement = 100*(baseline_score - score)/baseline_score
    h_improvement = 100*(baseline_h_rmse - h_rmse)/baseline_h_rmse
    s_improvement = 100*(baseline_s_rmse - s_rmse)/baseline_s_rmse

    print(
        f'{label}: kij = {kij:.6f}, points = {points}\n'
        f'  kij = 0: score = {baseline_score:.4f} J/(kg K), '
        f's_RMSE = {baseline_s_rmse:.4f} J/(kg K), '
        f's_max = {baseline_s_max:.4f} J/(kg K), '
        f'h_RMSE = {baseline_h_rmse:.1f} J/kg, '
        f'h_max = {baseline_h_max:.1f} J/kg\n'
        f'  fitted: score = {score:.4f} J/(kg K), '
        f's_RMSE = {s_rmse:.4f} J/(kg K), '
        f's_max = {s_max:.4f} J/(kg K), '
        f'h_RMSE = {h_rmse:.1f} J/kg, '
        f'h_max = {h_max:.1f} J/kg\n'
        f'  improvement: score = {score_improvement:.1f}%, '
        f's_RMSE = {s_improvement:.1f}%, '
        f'h_RMSE = {h_improvement:.1f}%'
    )


def main():
    print('CO2-rich cycle interactions')
    for other_species in CO2_OTHER_SPECIES:
        print_result(
            f'CO2-{other_species.name}',
            fit_interaction(
                utils.SPS['CO2'], other_species,
                TP_PAIRS, MAJOR_SPECIES_FRACTIONS, (0, 0.40),
            ),
        )

    print('\nFuel interactions')
    for other_species in FUEL_OTHER_SPECIES:
        print_result(
            f'CH4-{other_species.name}',
            fit_interaction(
                utils.SPS['CH4'], other_species,
                FUEL_TP_PAIRS, MAJOR_SPECIES_FRACTIONS, (-0.25, 0.40),
            ),
        )

    print('\nOxidant interaction')
    print_result(
        'O2-N2',
        fit_interaction(
            utils.SPS['O2'], utils.SPS['N2'],
            OXIDANT_TP_PAIRS, MAJOR_SPECIES_FRACTIONS, (-0.25, 0.40),
        ),
    )


if __name__ == '__main__':
    main()
