"""Numerical helpers and solver orchestration for the ASU model."""

import time

import numpy as np
from scipy.optimize import fsolve, least_squares

from ASU.asu_classes import (
    COMPOSITION_SCALE,
    FLOW_SCALE,
    PRESSURE_SCALE,
    TEMPERATURE_SCALE,
    State,
)
from ASU.asu_cycle_residuals import asu_residuals


def build_variable_scales(states):
    """Return fixed engineering scales for each state variable."""
    scales = []

    for state in states.values():
        n_composition = len(state.z) - 1
        scales.extend([
            FLOW_SCALE,
            TEMPERATURE_SCALE,
            PRESSURE_SCALE,
            *([COMPOSITION_SCALE]*n_composition),
        ])

    return np.array(scales, dtype=float)

def build_bounds(states, components):
    """Return physical lower and upper bounds for least-squares variables."""
    lower = []
    upper = []

    for state in states.values():
        n = len(state.z) - 1
        lower.extend([0.1, 70.0, 1e5, *([1e-7]*n)])
        upper.extend([12.0, 500.0, 1e8, *([1.0 - 1e-7]*n)])

    for component in components.values():
        n = len(component.flatten_vars())
        lower.extend([-np.inf]*n)
        upper.extend([np.inf]*n)

    return np.array(lower, dtype=float), np.array(upper, dtype=float)

def _component_states(component):
    """Collect State objects referenced by a component."""
    found = set()

    def visit(value):
        if isinstance(value, State):
            found.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for value in vars(component).values():
        visit(value)

    return found

def _color_columns(pattern):
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

class ColoredFiniteDifferenceJacobian:
    """Dense Jacobian for MINPACK, evaluated using sparse column coloring."""

    def __init__(
        self,
        function,
        pattern,
        relative_step=1.0e-6,
    ):
        self.function = function
        self.pattern = pattern
        self.relative_step = relative_step
        self.groups = _color_columns(pattern)

    def __call__(self, variables):
        base = self.function(variables)
        jacobian = np.zeros(self.pattern.shape)

        for columns in self.groups:
            steps = self.relative_step*np.maximum(np.abs(variables[columns]), 1.0)

            perturbed = variables.copy()
            perturbed[columns] += steps
            difference = self.function(perturbed) - base

            for column, step in zip(columns, steps):
                rows = self.pattern[:, column]
                jacobian[rows, column] = difference[rows]/step

        return jacobian

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
        for state in _component_states(component):
            pattern[row:row + count, state_columns[state]] = True
        row += count

    if pattern.shape[0] != pattern.shape[1]:
        raise ValueError(f'fsolve requires a square system; Jacobian shape is {pattern.shape}')

    return pattern

def solve_asu(states, components, x0, solver='fsolve'):
    """Solve the ASU and return ``(physical_solution, success)``."""

    start = time.perf_counter()
    counter = [0]

    variable_scales = build_variable_scales(states)
    scaled_x0 = x0/variable_scales

    def scaled_residuals(scaled_variables):
        try:
            return asu_residuals(
                scaled_variables*variable_scales,
                states,
                components,
                start,
                counter,
            )
        except ValueError:
            # MINPACK has no bounds. Reject thermodynamically invalid trial
            # points so that its trust region contracts instead of aborting.
            counter[0] += 1
            return np.full(len(scaled_variables), 1.0e4)

    if solver == 'least_squares':
        lower, upper = build_bounds(states, components)
        result = least_squares(
            scaled_residuals,
            scaled_x0,
            bounds=(lower/variable_scales, upper/variable_scales),
            x_scale='jac',
            xtol=1e-14,
            max_nfev=200,
        )
        x_final = result.x*variable_scales
        success = result.success or result.status == 0
        message = result.message
        solver_calls = result.nfev

    elif solver == 'fsolve':
        jacobian_pattern = build_jacobian_sparsity(states, components)
        jacobian = ColoredFiniteDifferenceJacobian(
            scaled_residuals,
            jacobian_pattern,
        )
        scaled_final, info, ier, message = fsolve(
            scaled_residuals,
            scaled_x0,
            fprime=jacobian,
            xtol=1e-10,
            full_output=True,
            maxfev=5000,
            factor=0.1,
        )
        x_final = scaled_final*variable_scales
        success = ier == 1
        solver_calls = info['nfev']

    else:
        raise ValueError("SOLVER must be 'fsolve' or 'least_squares'")

    elapsed = time.perf_counter() - start

    # Ensure the model objects contain the returned solution.
    residuals = asu_residuals(x_final, states, components)
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
