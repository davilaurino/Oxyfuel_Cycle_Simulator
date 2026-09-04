"""Manually test PR kij values against HEOS excess properties."""

import CoolProp.CoolProp as CP
import numpy as np
from scipy.constants import R


TEMPERATURES = [320.0]  # K
PRESSURES = [10.0e6]  # Pa
X_MINOR_VALUES = [0.10, 0.05, 0.01]
LOW_PRESSURE = 10  # Pa, approximates the ideal-gas limit

# Change these values manually and run the file again.
KIJ = {
    "Water": 0.13,
    "Oxygen": 0.19,
    "Nitrogen": 0.06,
}

SHORT_NAME = {
    "Water": "H2O",
    "Oxygen": "O2",
    "Nitrogen": "N2",
}


def pure_pr_properties(fluid, temperature, pressure):
    state = CP.AbstractState("PR", fluid)

    if pressure == LOW_PRESSURE:
        phase = CP.iphase_gas
    elif fluid == "Water" and temperature < state.T_critical():
        phase = CP.iphase_liquid
    else:
        phase = CP.iphase_gas

    state.specify_phase(phase)
    state.update(CP.PT_INPUTS, pressure, temperature)
    return state.hmolar(), state.smolar()


def raw_pr_mixing(minor_fluid, kij, temperature, pressure, x_minor):
    x = np.array([1.0 - x_minor, x_minor])
    mixture = CP.AbstractState("PR", f"CarbonDioxide&{minor_fluid}")
    mixture.set_mole_fractions(x)
    mixture.set_binary_interaction_double(0, 1, "kij", kij)

    if pressure == LOW_PRESSURE:
        mixture.specify_phase(CP.iphase_gas)

    mixture.update(CP.PT_INPUTS, pressure, temperature)

    h_co2, s_co2 = pure_pr_properties("CarbonDioxide", temperature, pressure)
    h_minor, s_minor = pure_pr_properties(minor_fluid, temperature, pressure)

    h_mixing = mixture.hmolar() - x[0] * h_co2 - x[1] * h_minor
    s_mixing = (
        mixture.smolar()
        - x[0] * s_co2
        - x[1] * s_minor
        + R * np.sum(x * np.log(x))
    )
    return np.array([h_mixing, s_mixing])


def pr_excess(minor_fluid, kij, temperature, pressure, x_minor):
    real_pressure = raw_pr_mixing(
        minor_fluid, kij, temperature, pressure, x_minor
    )
    ideal_limit = raw_pr_mixing(
        minor_fluid, kij, temperature, LOW_PRESSURE, x_minor
    )
    return real_pressure - ideal_limit


def heos_excess(minor_fluid, temperature, pressure, x_minor):
    state = CP.AbstractState("HEOS", f"CarbonDioxide&{minor_fluid}")
    state.set_mole_fractions([1.0 - x_minor, x_minor])
    state.update(CP.PT_INPUTS, pressure, temperature)
    excess = np.array([state.hmolar_excess(), state.smolar_excess()])
    return excess / state.molar_mass()


def calculate_grid(minor_fluid, kij):
    rows = []
    for x_minor in X_MINOR_VALUES:
        for temperature in TEMPERATURES:
            for pressure in PRESSURES:
                heos = heos_excess(
                    minor_fluid, temperature, pressure, x_minor
                )
                pr = pr_excess(
                    minor_fluid, kij, temperature, pressure, x_minor
                )

                state = CP.AbstractState(
                    "HEOS", f"CarbonDioxide&{minor_fluid}"
                )
                state.set_mole_fractions([1.0 - x_minor, x_minor])
                molecular_weight = state.molar_mass()
                pr /= molecular_weight

                # Plain utils.py assumes zero excess properties.
                utils = np.zeros(2)
                rows.append(
                    (x_minor, temperature, pressure, heos, utils, pr)
                )
    return rows


def print_property_table(rows, property_index, symbol, units):
    print(f"\n{symbol} [{units}]")
    print(
        f"{'x_minor':>9}{'T [K]':>8}{'P [MPa]':>10}"
        f"{'HEOS':>13}{'utils.py':>13}"
        f"{'PR':>13}{'utils error':>15}{'corrected error':>18}"
    )
    for x_minor, temperature, pressure, heos, utils, pr in rows:
        utils_error = utils[property_index] - heos[property_index]
        corrected_error = pr[property_index] - heos[property_index]
        print(
            f"{x_minor:>9.3f}{temperature:>8.1f}"
            f"{pressure / 1e6:>10.1f}"
            f"{heos[property_index]:>13.3f}"
            f"{utils[property_index]:>13.3f}"
            f"{pr[property_index]:>13.3f}"
            f"{utils_error:>15.3f}{corrected_error:>18.3f}"
        )


def print_error_summary(rows):
    utils_errors = np.array(
        [utils - heos for _, _, _, heos, utils, _ in rows]
    )
    corrected_errors = np.array(
        [pr - heos for _, _, _, heos, _, pr in rows]
    )

    print("\nGrid error summary")
    print(
        f"{'Method':<22}{'h MAE':>12}{'h max':>12}"
        f"{'s MAE':>12}{'s max':>12}"
    )
    for method, errors in (
        ("utils.py", utils_errors),
        ("utils.py + PR excess", corrected_errors),
    ):
        absolute = np.abs(errors)
        print(
            f"{method:<22}{absolute[:, 0].mean():>12.3f}"
            f"{absolute[:, 0].max():>12.3f}"
            f"{absolute[:, 1].mean():>12.3f}"
            f"{absolute[:, 1].max():>12.3f}"
        )


print(f"x_minor values = {X_MINOR_VALUES} (mole fractions)")

for minor_fluid, kij in KIJ.items():
    print("\n" + "=" * 96)
    print(f"CO2-{SHORT_NAME[minor_fluid]}, kij = {kij:+.4f}")
    grid = calculate_grid(minor_fluid, kij)
    print_property_table(grid, 0, "hE", "J/kg")
    print_property_table(grid, 1, "sE", "J/(kg K)")
    print_error_summary(grid)
