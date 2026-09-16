"""Numerical configuration for the ASU solver."""

# Reference scales for solver variables and residuals.
FLOW_SCALE = 1.0                    # kg/s
TEMPERATURE_SCALE = 10.0            # K
PRESSURE_SCALE = 1.0e5              # Pa
COMPOSITION_SCALE = 1.0e-2          # mole fraction
ENERGY_SCALE = 1.0e6                # W
SPECIFIC_ENTHALPY_SCALE = 1.0e5     # J/kg

# Physical bounds for state variables used by the least-squares solver and
# finite-difference Jacobian probes.
FLOW_BOUNDS = (1.0e-3, 12.0)                 # kg/s
TEMPERATURE_BOUNDS = (60.0, 500.0)           # K
PRESSURE_BOUNDS = (1.0e5, 1.0e8)             # Pa
COMPOSITION_BOUNDS = (1.0e-6, 1.0 - 1.0e-6)  # mole fraction
