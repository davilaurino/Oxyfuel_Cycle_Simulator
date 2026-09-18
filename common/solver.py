"""Numerical helpers and process-model solver orchestration."""

import time

import numpy as np
from scipy.optimize import fsolve, least_squares

from ASU.asu_config import (
    COMPOSITION_SCALE,
    COMPOSITION_BOUNDS,
    FD_RELATIVE_STEP,
    FLOW_SCALE,
    FLOW_BOUNDS,
    FSOLVE_FACTOR,
    FSOLVE_MAXFEV,
    FSOLVE_XTOL,
    LEAST_SQUARES_FTOL,
    LEAST_SQUARES_GTOL,
    LEAST_SQUARES_MAX_NFEV,
    LEAST_SQUARES_XTOL,
    PRESSURE_SCALE,
    PRESSURE_BOUNDS,
    TEMPERATURE_SCALE,
    TEMPERATURE_BOUNDS,
)
from common.residuals import evaluate_residuals
from common.thermo import CoolPropFailure


def build_variable_scales(states, config=None):
    """Return fixed engineering scales for each state variable."""
    if config is None:
        import ASU.asu_config as config

    scales = []

    for state in states.values():
        n_composition = len(state.z) - 1
        scales.extend([
            config.FLOW_SCALE,
            config.TEMPERATURE_SCALE,
            config.PRESSURE_SCALE,
            *([config.COMPOSITION_SCALE]*n_composition),
        ])

    return np.array(scales, dtype=float)

def build_bounds(states, config=None):
    """Return physical bounds for least-squares and Jacobian probes."""
    if config is None:
        import ASU.asu_config as config

    lower = []
    upper = []

    for state in states.values():
        n = len(state.z) - 1
        lower.extend([
            config.FLOW_BOUNDS[0],
            config.TEMPERATURE_BOUNDS[0],
            config.PRESSURE_BOUNDS[0],
            *([config.COMPOSITION_BOUNDS[0]]*n),
        ])
        upper.extend([
            config.FLOW_BOUNDS[1],
            config.TEMPERATURE_BOUNDS[1],
            config.PRESSURE_BOUNDS[1],
            *([config.COMPOSITION_BOUNDS[1]]*n),
        ])
    return np.array(lower), np.array(upper)

def color_columns(pattern):
    """Greedily group Jacobian columns whose nonzeros do not share rows."""
    conflicts = pattern.T @ pattern                              # Variable pairs sharing a residual row
    np.fill_diagonal(conflicts, False)                           # A variable does not conflict with itself
    conflict_counts = np.count_nonzero(conflicts, axis=1)        # Count of conflicts for each variable
    order = np.argsort(-conflict_counts, kind='stable')          # Most conflicts first
    colors = np.full_like(order, -1)                             # -1 means uncolored

    for column in order:
        conflicting_columns = np.flatnonzero(conflicts[column])  # Columns sharing a residual
        unavailable = set(colors[conflicting_columns])           # Colors already assigned to conflicting columns

        color = 0
        while color in unavailable:                              # Find the lowest color not used by conflicting columns   
            color += 1

        colors[column] = color                                   # Assign the color to the column

    groups = []
    for color in range(colors.max() + 1):
        columns = np.flatnonzero(colors == color)
        groups.append(columns)

    return groups

def build_jacobian_sparsity(states, components):
    """Build a conservative equation/variable dependency matrix."""
    state_columns = {}
    column = 0
    for state in states.values():
        width = len(state.flatten_vars())
        state_columns[state] = np.arange(column, column + width)
        column += width

    row_counts = [len(component.residuals()) for component in components.values()]
    pattern = np.zeros((sum(row_counts), column), dtype=bool)

    row = 0
    for component, count in zip(components.values(), row_counts):
        for state in component.state_dependencies:
            pattern[row:row + count, state_columns[state]] = True
        row += count

    return pattern

class ColoredFiniteDifferenceJacobian:
    """Dense finite-difference Jacobian using sparse column coloring."""

    def __init__(self, function, pattern, relative_step=1.0e-6, upper_bounds=None):
        self.function = function
        self.pattern = pattern
        self.relative_step = relative_step
        self.groups = color_columns(pattern)
        self.upper_bounds = upper_bounds

    def __call__(self, variables):
        base = self.function(variables)
        jacobian = np.zeros(self.pattern.shape)

        for columns in self.groups:
            steps = self.relative_step*np.maximum(np.abs(variables[columns]), 1.0)

            if self.upper_bounds is not None:
                use_backward = (variables[columns] + steps > self.upper_bounds[columns])
                steps[use_backward] *= -1.0

            perturbed = variables.copy()
            perturbed[columns] += steps
            difference = self.function(perturbed) - base

            for column, step in zip(columns, steps):
                rows = self.pattern[:, column]
                jacobian[rows, column] = difference[rows]/step

        return jacobian

def solve(states, components, x0, solver='fsolve', config=None):
    """Solve a process model and return ``(physical_solution, success)``."""

    start = time.perf_counter()
    counter = [0]

    if config is None:
        import ASU.asu_config as config

    variable_scales = build_variable_scales(states, config)
    scaled_x0 = x0/variable_scales

    def scaled_residuals(scaled_variables):
        try:
            return evaluate_residuals(
                scaled_variables*variable_scales,
                states,
                components,
                start,
                counter,
            )
        except CoolPropFailure:
            # Reject thermodynamically invalid trial points without aborting.
            counter[0] += 1
            return np.full(len(scaled_variables), 1.0e4)

    lower, upper = build_bounds(states, config)
    lower_scaled = lower/variable_scales
    upper_scaled = upper/variable_scales
    jacobian_pattern = build_jacobian_sparsity(states, components)
    jacobian = ColoredFiniteDifferenceJacobian(
        scaled_residuals,
        jacobian_pattern,
        relative_step=config.FD_RELATIVE_STEP,
        upper_bounds=upper_scaled,
    )

    if solver == 'least_squares':
        result = least_squares(
            scaled_residuals,
            scaled_x0,
            jac=jacobian,
            bounds=(lower_scaled, upper_scaled),
            xtol=config.LEAST_SQUARES_XTOL,
            gtol=config.LEAST_SQUARES_GTOL,
            ftol=config.LEAST_SQUARES_FTOL,
            max_nfev=config.LEAST_SQUARES_MAX_NFEV,
        )
        x_final = result.x*variable_scales
        success = result.success
        message = result.message
        solver_calls = result.nfev

    elif solver == 'fsolve':
        scaled_final, info, ier, message = fsolve(
            scaled_residuals,
            scaled_x0,
            fprime=jacobian,
            xtol=config.FSOLVE_XTOL,
            full_output=True,
            maxfev=config.FSOLVE_MAXFEV,
            factor=config.FSOLVE_FACTOR,
        )
        x_final = scaled_final*variable_scales
        success = ier == 1
        solver_calls = info['nfev']

    elapsed = time.perf_counter() - start

    # Ensure the model objects contain the returned solution.
    residuals = evaluate_residuals(x_final, states, components)
    residual_norm = np.linalg.norm(residuals)
    max_residual = np.max(np.abs(residuals))

    print('Solver:', solver)
    print('Success:', success)
    print('Message:', message)
    print('Solver function calls:', solver_calls)
    print('Residual evaluations:', counter[0])
    print('Residual norm:', residual_norm)
    print('Maximum residual:', max_residual)
    print(f'Execution time: {elapsed:.3f} s')

    return x_final, success
