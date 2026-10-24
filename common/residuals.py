from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True, slots=True)
class ResidualScaling:
    """Multipliers that convert dimensional equations to scaled residuals."""

    energy: float
    specific_enthalpy: float
    flow: float
    temperature: float
    pressure: float
    composition: float

    @classmethod
    def from_config(cls, config):
        """Build residual multipliers from a numerical configuration module."""
        return cls(
            energy=1.0/config.ENERGY_SCALE,
            specific_enthalpy=1.0/config.SPECIFIC_ENTHALPY_SCALE,
            flow=1.0/config.FLOW_SCALE,
            temperature=1.0/config.TEMPERATURE_SCALE,
            pressure=1.0/config.PRESSURE_SCALE,
            composition=1.0/config.COMPOSITION_SCALE,
        )


def evaluate_residuals(
    variables,
    states,
    components,
    scaling,
    start=None,
    counter=None,
):
    """Update a process model and return its scaled residual vector."""
    i = 0

    for state in states.values():
        i = state.unpack_vars(variables, i)

    if i != len(variables):
        raise ValueError(f'Solver vector has {len(variables)} values, but the model used {i}')

    eqs = []
    for component in components.values():
        eqs.extend(component.residuals(scaling))

    eqs = np.array(eqs)

    if counter is not None and counter[0] % 50 == 0:
        print('-'*20)
        run = time.perf_counter() - start
        print('Run Time:', round(run, 2), '(s)')
        print('Residuals Norm:', np.linalg.norm(eqs))
        print('Residual Evaluations:', counter[0] + 1)

    if counter is not None:
        counter[0] += 1

    return eqs
