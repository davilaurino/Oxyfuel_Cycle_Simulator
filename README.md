# Oxyfuel Sim

Research-oriented Python models for an oxy-fuel power system. The repository
currently contains a standalone Allam-cycle model, gas- and liquid-oxygen
cryogenic air separation unit (ASU) models, and an early coupled LOX-ASU/Allam
flowsheet.

The models formulate component mass, energy, pressure, composition, and phase
equilibrium constraints as nonlinear residual systems. Thermodynamic properties
come from CoolProp's HEOS backend; SciPy solves the systems with a colored
finite-difference Jacobian.

## Current state

- `ALLAM_v2` is the current standalone Allam-cycle implementation. It includes
  fuel and recycle compression, combustion, turbine cooling, three turbine
  stages, recuperation, condensation, and CO2 recycle/purge handling.
- `ASU` provides two runnable double-column ASU configurations: `lox` uses
  internal liquid-oxygen pumping to deliver oxygen at 30 MPa, while `gox`
  delivers warm gaseous oxygen at low pressure.
- `INTEGRATED` can assemble a shared LOX-ASU/Allam residual system, connecting
  the ASU oxygen product directly to the Allam oxidant feed. It does not yet
  have a command-line runner or integrated report.
- Saved solution vectors (`0_solution*.npy`) and state tables
  (`0_results*.csv`) are included as convergence starting points and reference
  snapshots.
- This is active research code: inputs are defined in source, dependencies are
  not pinned, and there is currently no automated test suite or packaged API.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the runtime dependencies:

```bash
python -m pip install numpy scipy pandas CoolProp
```

Run commands from the repository root so local packages resolve consistently.

## Running the models

Allam cycle:

```bash
python ALLAM_v2/allam_main.py
```

Cryogenic ASU:

```bash
python ASU/asu_main.py
```

The ASU runner first asks for `lox` or `gox`. Both runners then ask whether to
start from the previous solution (`p`) or the setup values (`s`). The saved
solution is usually the better starting point. On convergence, the program
prints solver diagnostics, state/performance results, and writes a timestamped
CSV beside the runner. Answering `y` to the final prompt replaces the canonical
solution vector and `0_results*.csv` file.

`fsolve` is the default. To try bounded nonlinear least squares, change the
`SOLVER` constant in the corresponding `*_main.py` file to
`'least_squares'`. Model assumptions and starting values live in
`allam_setup.py` and `asu_setup_{lox,gox}.py`; numerical scales, bounds, and
tolerances live in the matching config modules.

The coupled model is currently available only as a builder:

```python
from INTEGRATED.integrated_setup import build_integrated_cycle

states, components, x0 = build_integrated_cycle()
```

For a quick thermodynamic consistency check of the experimental
Peng–Robinson helpers in the root `thermo.py` module, run:

```bash
python thermo.py
```

## Repository layout

```text
ALLAM_v2/   Allam-cycle setup, runner, reporting, and saved baseline
ASU/        LOX/GOX ASU setups, runner, reporting, and saved baselines
common/     Stream/component models, HEOS helpers, builders, and solver
INTEGRATED/ Coupled LOX-ASU/Allam model assembly (work in progress)
thermo.py   Experimental Peng–Robinson mixture-property utilities
```

All process quantities use SI units internally: kg/s, K, Pa, J/kg, and W.
