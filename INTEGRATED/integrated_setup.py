"""Coupled LOX-ASU and ALLAM cycle setup."""

from pathlib import Path

import numpy as np

from ALLAM_v2.allam_setup import build_cycle as build_allam
from ASU.asu_setup_lox import build_asu as build_lox_asu


def _merge_unique(first, second, collection_name):
    duplicates = set(first).intersection(second)
    if duplicates:
        duplicate_names = ', '.join(sorted(duplicates))
        raise ValueError(
            f'Duplicate {collection_name} names in integrated model: '
            f'{duplicate_names}'
        )

    merged = dict(first)
    merged.update(second)
    return merged


def build_integrated_cycle():
    """Build one residual system for the LOX ASU and ALLAM cycle.

    The ASU oxygen product is the ALLAM oxidant-mixer feed. Its mass flow is
    determined by the ALLAM combustor oxygen requirement, while the ASU air
    intake flow adjusts to supply it.
    """
    asu_states, asu_components, _ = build_lox_asu(o2_mass_flow=None)
    oxygen_product = asu_states['O2 product']

    allam_states, allam_components, _ = build_allam(
        o2_feed=oxygen_product,
    )

    # Preserve ownership for subsystem power accounting after the merge.
    for component in asu_components.values():
        component.subsystem = 'ASU'
    for component in allam_components.values():
        component.subsystem = 'ALLAM'

    states = _merge_unique(asu_states, allam_states, 'state')
    components = _merge_unique(
        asu_components,
        allam_components,
        'component',
    )

    x0 = np.asarray([
        value
        for state in states.values()
        for value in state.flatten_vars()
    ], dtype=float)

    return states, components, x0


def load_solution(path, expected_size):
    """Load a finite physical solution vector matching the current model."""
    values = np.load(path, allow_pickle=False)
    if values.shape != (expected_size,):
        raise ValueError(
            f'{path}: expected a solution of shape ({expected_size},), '
            f'received {values.shape}'
        )
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f'{path}: solution must contain real numbers')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{path}: solution contains non-finite values')
    return values.astype(float, copy=True)


def standalone_initial_guess(states, asu_path=None, allam_path=None):
    """Map saved standalone solutions into the coupled state ordering.

    The standalone ALLAM oxygen boundary is replaced by the ASU product.
    Scale all ASU flows to that boundary's demand, retaining the ASU's solved
    temperatures, pressures, and compositions as the starting point.
    This supplies guesses only; no inlet or product flow is fixed.
    """
    root = Path(__file__).resolve().parent.parent
    asu_path = Path(asu_path) if asu_path is not None else root / 'ASU/0_solution_lox.npy'
    allam_path = Path(allam_path) if allam_path is not None else root / 'ALLAM_v2/0_solution.npy'
    asu_states, _, asu_x0 = build_lox_asu()
    allam_states, _, allam_x0 = build_allam()

    guesses = {}
    for model_states, path, x0 in (
        (asu_states, asu_path, asu_x0),
        (allam_states, allam_path, allam_x0),
    ):
        values = load_solution(path, len(x0))
        offset = 0
        for name, state in model_states.items():
            width = len(state.flatten_vars())
            guesses[name] = values[offset:offset + width].copy()
            offset += width

    oxygen_flow = guesses['O2 product'][0]
    demanded_flow = guesses['O2 - Inlet Stream'][0]
    if oxygen_flow <= 0.0 or demanded_flow <= 0.0:
        raise ValueError('Saved oxygen flows must be positive')
    flow_ratio = demanded_flow/oxygen_flow
    for name in asu_states:
        guesses[name][0] *= flow_ratio

    return np.concatenate([guesses[name] for name in states])
