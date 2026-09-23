"""Coupled LOX-ASU and ALLAM cycle setup."""

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
