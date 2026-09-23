"""Process components and thermodynamic state classes."""

import numpy as np
from common import utils

from ASU.asu_config import (
    COMPOSITION_SCALE,
    ENERGY_SCALE,
    FLOW_SCALE,
    PRESSURE_SCALE,
    SPECIFIC_ENTHALPY_SCALE,
    TEMPERATURE_SCALE,
)
from common.thermo import (
    new_heos,
    ps_flash_enthalpy,
    ps_gas_enthalpy,
    ps_liquid_enthalpy,
    update_state,
)

# Residual scaling factors.
H_mult = 1.0/ENERGY_SCALE
h_mult = 1.0/SPECIFIC_ENTHALPY_SCALE
m_mult = 1.0/FLOW_SCALE
T_mult = 1.0/TEMPERATURE_SCALE
P_mult = 1.0/PRESSURE_SCALE
z_mult = 1.0/COMPOSITION_SCALE

class State:
    """A process stream backed by a reusable CoolProp HEOS state.

    Process flow and the primary thermodynamic properties use a mass basis.
    Mass fractions and phase-equilibrium data are cached for downstream units."""

    def __init__(self, name, species, m_dot, T, P, z, phase=None):
        self.name = name

        if phase not in (None, 'liquid', 'vapor'):
            raise ValueError("State phase must be None, 'liquid', or 'vapor'")

        self.species = []
        for species_item in species:
            if isinstance(species_item, str):
                species_object = utils.SPS[species_item]
            elif isinstance(species_item, utils.Species):
                species_object = species_item

            self.species.append(species_object)

        self.imposed_phase = phase

        self.thermo_state = new_heos(self.species)
        self.update(m_dot, T, P, z)

    def flatten_vars(self):
        """Return this state's independent solver variables."""
        return [self.m_dot, self.T, self.P, *self.z[:-1]]

    def unpack_vars(self, variables, start_idx):
        """Update this state from a solver vector and return the next index."""
        m_dot = variables[start_idx]
        T = variables[start_idx + 1]
        P = variables[start_idx + 2]

        n_composition = len(self.z) - 1
        z = self.z.copy()
        z[:-1] = variables[start_idx + 3:start_idx + 3 + n_composition]
        z[-1] = 1.0 - np.sum(z[:-1])

        self.update(m_dot, T, P, z)

        return start_idx + 3 + n_composition

    def update(self, m_dot, T, P, z):
        """Update the process state and cache its thermodynamic properties."""
        self.m_dot = m_dot
        self.T = T
        self.P = P
        self.z = np.array(z, dtype=float, copy=True)

        (
            self.phase,
            self.h,
            self.s,
            self.molar_mass,
            self.w,
            self.vapor_fraction,
            self.phi,
        ) = update_state(
            self.thermo_state,
            self.species,
            self.T,
            self.P,
            self.z,
            self.imposed_phase,
        )

class Input:
    """Fixed inlet conditions applied to an outlet process stream."""

    def __init__(self, name, T, P, z, outlet, m_dot=None):
        self.name = name
        self.state_dependencies = (outlet,)
        self.T = T
        self.P = P
        self.z = np.array(z, dtype=float, copy=True)
        self.outlet = outlet
        self.m_dot = m_dot

    def residuals(self):
        eqs = []
        outlet = self.outlet

        if self.m_dot is not None:
            eqs.append((outlet.m_dot - self.m_dot)*m_mult)

        eqs.append((outlet.T - self.T)*T_mult)
        eqs.append((outlet.P - self.P)*P_mult)
        
        for i in range(len(self.z) - 1):
            eqs.append((outlet.z[i] - self.z[i])*z_mult)

        return eqs


class Mixer:
    """Adiabatically mix two or more inlet streams into one outlet stream.

    The outlet pressure is set to the lowest inlet pressure, the outlet
    composition is obtained from the inlet molar flows, and the outlet
    temperature is determined by the enthalpy balance. Optional relative
    ``inlet_mass_ratios`` add one flow specification per inlet except the
    final reference inlet.
    """

    def __init__(self, name, inlets, outlet, inlet_mass_ratios=None):
        inlets = tuple(inlets)

        if inlet_mass_ratios is not None:
            if len(inlet_mass_ratios) != len(inlets):
                raise ValueError('inlet_mass_ratios must match inlet count')
            inlet_mass_ratios = np.asarray(inlet_mass_ratios, dtype=float)
            if np.any(inlet_mass_ratios <= 0.0):
                raise ValueError('inlet_mass_ratios must be positive')

        self.name = name
        self.inlets = inlets
        self.outlet = outlet
        self.inlet_mass_ratios = inlet_mass_ratios
        self.state_dependencies = (*inlets, outlet)

    def residuals(self):
        eqs = []
        outlet = self.outlet

        if self.inlet_mass_ratios is not None:
            reference_inlet = self.inlets[-1]
            reference_ratio = self.inlet_mass_ratios[-1]
            for inlet, ratio in zip(
                self.inlets[:-1],
                self.inlet_mass_ratios[:-1],
            ):
                target_ratio = ratio/reference_ratio
                eqs.append(
                    (inlet.m_dot - target_ratio*reference_inlet.m_dot)*m_mult
                )

        species_molar_flows = {}
        total_molar_flow = 0.0
        for species in outlet.species:
            species_molar_flows[species] = 0.0
            
        for inlet in self.inlets:
            inlet_molar_flow = inlet.m_dot/inlet.molar_mass
            total_molar_flow += inlet_molar_flow

            for species, fraction in zip(inlet.species, inlet.z):
                species_molar_flows[species] += inlet_molar_flow*fraction

        outlet_z = []
        for species in outlet.species:
            outlet_z.append(species_molar_flows[species]/total_molar_flow)

        eqs.append((outlet.m_dot - sum(inlet.m_dot for inlet in self.inlets))*m_mult)
        eqs.append((outlet.P - min(inlet.P for inlet in self.inlets))*P_mult)

        inlet_enthalpy_flow = sum(inlet.m_dot*inlet.h for inlet in self.inlets)
        eqs.append((outlet.m_dot*outlet.h - inlet_enthalpy_flow)*H_mult)

        for i in range(len(outlet.z) - 1):
            eqs.append((outlet.z[i] - outlet_z[i])*z_mult)

        return eqs


class Combustor:
    """Adiabatic complete combustion of a hydrocarbon fuel."""

    def __init__(self, name, fuel, oxidant, outlet, excess_o2=None):
        self.name = name
        self.fuel = fuel
        self.oxidant = oxidant
        self.outlet = outlet
        self.excess_o2 = excess_o2
        self.state_dependencies = (fuel, oxidant, outlet)

    def residuals(self):
        fuel = self.fuel
        oxidant = self.oxidant
        outlet = self.outlet

        _, fuel_molar_flow, carbon_flow, hydrogen_flow = (utils.stoichiometry(fuel))
        stoichiometric_o2 = carbon_flow + hydrogen_flow/4.0
        oxidant_molar_flow = oxidant.m_dot/oxidant.molar_mass

        fuel_flows = {}
        for species, zi in zip(fuel.species, fuel.z):
            fuel_flows[species] = fuel_molar_flow*zi

        oxidant_flows = {}
        for species, zi in zip(oxidant.species, oxidant.z):
            oxidant_flows[species] = oxidant_molar_flow*zi

        product_flows = {}
        for species in outlet.species:
            inlet_flow = (fuel_flows.get(species, 0.0) + oxidant_flows.get(species, 0.0))

            if species is utils.CO2:
                product_flows[species] = inlet_flow + carbon_flow
            elif species is utils.H2O:
                product_flows[species] = inlet_flow + hydrogen_flow/2.0
            elif species is utils.O2:
                product_flows[species] = inlet_flow - stoichiometric_o2
            else:
                product_flows[species] = inlet_flow

        total_product_flow = sum(product_flows.values())

        product_z = {}
        for species, flow in product_flows.items():
            product_z[species] = flow/total_product_flow

        eqs = []
        if self.excess_o2 is not None:
            o2_index = oxidant.species.index(utils.O2)
            target_o2 = self.excess_o2*stoichiometric_o2/oxidant_molar_flow
            eqs.append((oxidant.z[o2_index] - target_o2)*z_mult)

        eqs.append((outlet.m_dot - fuel.m_dot - oxidant.m_dot)*m_mult)
        eqs.append((outlet.P - oxidant.P)*P_mult)

        for species, zi in zip(outlet.species[:-1], outlet.z[:-1]):
            eqs.append((zi - product_z[species])*z_mult)

        enthalpy_in = fuel.m_dot*fuel.h + oxidant.m_dot*oxidant.h
        enthalpy_out = outlet.m_dot*outlet.h
        eqs.append((enthalpy_in - enthalpy_out)*H_mult)

        return eqs


class Compressor:
    """Adiabatic gas compressor with an exact gas isentropic flash."""

    def __init__(self, name, efficiency, pressure_ratio, inlet, outlet):
        self.name = name
        self.state_dependencies = (inlet, outlet)
        self.efficiency = efficiency
        self.pressure_ratio = pressure_ratio
        self.inlet = inlet
        self.outlet = outlet
        self.W = None

        self.thermo_state = new_heos(inlet.species)

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet
        P_out = inlet.P*self.pressure_ratio

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        eqs.append((outlet.P - P_out)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        h_iso = ps_gas_enthalpy(self.thermo_state, P_out, inlet.s, outlet.z)
        eqs.append(((outlet.h - inlet.h) - (h_iso - inlet.h)/self.efficiency)*h_mult)

        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs


class Turbine:
    """Adiabatic turbine with an optional inlet-temperature specification."""

    def __init__(self, name, efficiency, P_out, inlet, outlet, TIT=None):
        self.name = name
        self.state_dependencies = (inlet, outlet)
        self.efficiency = efficiency
        self.P_out = P_out
        self.inlet = inlet
        self.outlet = outlet
        self.TIT = TIT
        self.W = None

        self.thermo_state = new_heos(inlet.species)

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        eqs.append((outlet.P - self.P_out)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        h_iso = ps_flash_enthalpy(self.thermo_state, self.P_out, inlet.s, inlet.z)
        eqs.append(((inlet.h - outlet.h) - self.efficiency*(inlet.h - h_iso))*h_mult)

        if self.TIT is not None:
            eqs.append((inlet.T - self.TIT)*T_mult)

        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs


class Pump:
    """Liquid-phase pump with an exact liquid isentropic reference state."""

    def __init__(self, name, efficiency, P_out, inlet, outlet):
        self.name = name
        self.state_dependencies = (inlet, outlet)
        self.efficiency = efficiency
        self.P_out = P_out
        self.inlet = inlet
        self.outlet = outlet
        self.W = None

        if inlet.imposed_phase != 'liquid' or outlet.imposed_phase != 'liquid':
            raise ValueError('Pump requires liquid-imposed inlet and outlet states')

        self.thermo_state = new_heos(inlet.species)

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        eqs.append((outlet.P - self.P_out)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        h_iso = ps_liquid_enthalpy(self.thermo_state, self.P_out, inlet.s, inlet.z)
        eqs.append(((outlet.h - inlet.h) - (h_iso - inlet.h)/self.efficiency)*h_mult)

        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs


class Intercooler:
    """Cooler with a specified outlet temperature and pressure drop."""

    def __init__(self, name, T_out, pressure_drop_percent, inlet, outlet):
        self.name = name
        self.state_dependencies = (inlet, outlet)
        self.T_out = T_out
        self.pressure_drop_percent = pressure_drop_percent
        self.inlet = inlet
        self.outlet = outlet
        self.Q_ext = None

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet
        pressure_ratio = 1.0 - self.pressure_drop_percent/100.0

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        eqs.append((outlet.T - self.T_out)*T_mult)
        eqs.append((outlet.P - inlet.P*pressure_ratio)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        self.Q_ext = inlet.m_dot*(outlet.h - inlet.h)
        return eqs


class Condenser:
    """Cool a process stream and separate equilibrium vapor and liquid.

    The vapor and liquid outlets are fixed at ``T_out`` and the inlet
    pressure. Component mass balances determine the two outlet flows and
    compositions. Fugacity equality for every species enforces full
    liquid-vapor equilibrium.
    """

    def __init__(self, name, T_out, inlet, vapor_outlet, liquid_outlet):
        inlet_species = set(inlet.species)
        vapor_species = set(vapor_outlet.species)
        liquid_species = set(liquid_outlet.species)

        if vapor_outlet.imposed_phase != 'vapor':
            raise ValueError('Condenser vapor outlet must impose the vapor phase')
        if liquid_outlet.imposed_phase != 'liquid':
            raise ValueError('Condenser liquid outlet must impose the liquid phase')

        self.name = name
        self.T_out = T_out
        self.inlet = inlet
        self.vapor_outlet = vapor_outlet
        self.liquid_outlet = liquid_outlet
        self.state_dependencies = (inlet, vapor_outlet, liquid_outlet)
        self.Q_ext = None

    def residuals(self):
        eqs = []
        inlet = self.inlet
        vapor = self.vapor_outlet
        liquid = self.liquid_outlet

        for inlet_index, species in enumerate(inlet.species):
            vapor_index = vapor.species.index(species)
            liquid_index = liquid.species.index(species)
            mass_in = inlet.m_dot*inlet.w[inlet_index]
            mass_out = (vapor.m_dot*vapor.w[vapor_index] + liquid.m_dot*liquid.w[liquid_index])

            eqs.append((mass_in - mass_out)*m_mult)

        eqs.append((vapor.T - self.T_out)*T_mult)
        eqs.append((liquid.T - self.T_out)*T_mult)
        eqs.append((vapor.P - inlet.P)*P_mult)
        eqs.append((liquid.P - inlet.P)*P_mult)

        for species in inlet.species:
            vapor_index = vapor.species.index(species)
            liquid_index = liquid.species.index(species)
            fugacity_vapor = (vapor.phi[vapor_index] * vapor.z[vapor_index] * vapor.P)
            fugacity_liquid = (liquid.phi[liquid_index] * liquid.z[liquid_index] * liquid.P)

            eqs.append((fugacity_vapor - fugacity_liquid)*P_mult)

        self.Q_ext = (vapor.m_dot*vapor.h + liquid.m_dot*liquid.h - inlet.m_dot*inlet.h)
        return eqs


class ReboilerCondenser:
    """Coupled total condenser and partial reboiler with equal duties."""

    def __init__(self, name, condenser_vapor_in, condenser_liquid_out, reboiler_liquid_in, reboiler_liquid_out, reboiler_vapor_out, T_cold_pinch):
        self.name = name
        self.state_dependencies = (
            condenser_vapor_in,
            condenser_liquid_out,
            reboiler_liquid_in,
            reboiler_liquid_out,
            reboiler_vapor_out,
        )
        self.condenser_vapor_in = condenser_vapor_in
        self.condenser_liquid_out = condenser_liquid_out
        self.reboiler_liquid_in = reboiler_liquid_in
        self.reboiler_liquid_out = reboiler_liquid_out
        self.reboiler_vapor_out = reboiler_vapor_out
        self.T_cold_pinch = T_cold_pinch
        self.Q_condenser = None
        self.Q_reboiler = None

    def residuals(self):
        eqs = []
        cv_in = self.condenser_vapor_in
        cl_out = self.condenser_liquid_out
        rl_in = self.reboiler_liquid_in
        rl_out = self.reboiler_liquid_out
        rv_out = self.reboiler_vapor_out

        self.Q_condenser = (cl_out.m_dot*cl_out.h - cv_in.m_dot*cv_in.h)
        self.Q_reboiler = (rl_out.m_dot*rl_out.h + rv_out.m_dot*rv_out.h - rl_in.m_dot*rl_in.h)

        eqs.append((cv_in.m_dot - cl_out.m_dot)*m_mult)

        for i in range(len(rl_in.species)):
            m_in_i = rl_in.m_dot*rl_in.w[i]
            m_out_i = (rl_out.m_dot*rl_out.w[i] + rv_out.m_dot*rv_out.w[i])
            eqs.append((m_in_i - m_out_i)*m_mult)

        # Careful! As the pinch is enforced and condenser oulet phase is imposed as liquid, the condensation must not be partial.
        eqs.append((cl_out.T - rl_in.T - self.T_cold_pinch)*T_mult)
        eqs.append((rl_out.T - rv_out.T)*T_mult)

        eqs.append((cl_out.P - cv_in.P)*P_mult)
        eqs.append((rl_out.P - rl_in.P)*P_mult)
        eqs.append((rv_out.P - rl_in.P)*P_mult)

        for i in range(len(cl_out.z) - 1):
            eqs.append((cl_out.z[i] - cv_in.z[i])*z_mult)

        for i in range(len(rl_in.species)):
            f_liquid = (rl_out.phi[i]*rl_out.z[i]*rl_out.P)
            f_vapor = (rv_out.phi[i]*rv_out.z[i]*rv_out.P)
            eqs.append((f_liquid - f_vapor)*P_mult)

        eqs.append((self.Q_reboiler + self.Q_condenser)*H_mult)
        return eqs


class O2Specification:
    """Flowsheet-level oxygen product specifications."""

    def __init__(self, name, o2_product, purity_target, m_dot=None):
        if not 0.0 < purity_target <= 1.0:
            raise ValueError('purity_target must be between 0 and 1')
        if m_dot is not None and m_dot <= 0.0:
            raise ValueError('m_dot must be positive')

        self.name = name
        self.state_dependencies = (o2_product,)
        self.o2_product = o2_product
        self.purity_target = purity_target
        self.m_dot = m_dot

    def residuals(self):
        eqs = []
        product = self.o2_product

        for i, species in enumerate(product.species):
            if species.name == 'O2':
                o2_index = i
                break

        eqs.append((product.z[o2_index] - self.purity_target)*z_mult)
        if self.m_dot is not None:
            eqs.append((product.m_dot - self.m_dot)*m_mult)

        return eqs


class Splitter:
    """Divide one process stream into any number of streams without separation.

    ``outlets`` is a sequence containing the output states.  Flow allocation
    is optional; when omitted, only the total mass balance is enforced.

    Flow allocation can be specified in one of three ways:

    * ``split_fractions``: fractions for all outlets, summing to one;
    * ``split_ratios``: relative weights for all outlets, normalized internally;
    * ``outlet_flows``: fixed mass flows for selected outlets; use ``None``
      for flows that are determined elsewhere in the flowsheet.

    Each specified allocation adds one flow equation.  The total mass balance
    is always enforced.
    """

    def __init__(
        self,
        name,
        inlet,
        outlets,
        split_fractions=None,
        split_ratios=None,
        outlet_flows=None,
    ):
        outlets = list(outlets)

        if len(outlets) < 2 or any(outlet is None for outlet in outlets):
            raise ValueError('Splitter requires at least two valid outlets')

        specifications = [
            split_fractions is not None,
            split_ratios is not None,
            outlet_flows is not None,
        ]

        if sum(specifications) > 1:
            raise ValueError(
                'Use only one of split_fractions, '
                'split_ratios, or outlet_flows'
            )

        self.name = name
        self.inlet = inlet
        self.outlets = tuple(outlets)
        self.state_dependencies = (inlet, *self.outlets)

        self._flow_targets = {}
        if split_fractions is not None:
            if len(split_fractions) != len(self.outlets):
                raise ValueError('split_fractions must match outlet count')
            fractions = np.asarray(split_fractions, dtype=float)
            if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
                raise ValueError('split_fractions must be nonnegative and sum to one')
            for outlet, fraction in zip(self.outlets[:-1], fractions[:-1]):
                self._flow_targets[outlet] = ('fraction', fraction)

        elif split_ratios is not None:
            if len(split_ratios) != len(self.outlets):
                raise ValueError('split_ratios must match outlet count')
            ratios = np.asarray(split_ratios, dtype=float)
            if np.any(ratios < 0.0) or ratios.sum() <= 0.0:
                raise ValueError('split_ratios must be nonnegative and nonzero')
            ratios /= ratios.sum()
            for outlet, fraction in zip(self.outlets[:-1], ratios[:-1]):
                self._flow_targets[outlet] = ('fraction', fraction)

        elif outlet_flows is not None:
            if len(outlet_flows) != len(self.outlets):
                raise ValueError('outlet_flows must match outlet count')
            for outlet, flow in zip(self.outlets, outlet_flows):
                if flow is not None:
                    if flow < 0.0:
                        raise ValueError('outlet flows must be nonnegative')
                    self._flow_targets[outlet] = ('flow', flow)

    def residuals(self):
        eqs = []
        inlet = self.inlet

        for outlet, (kind, target) in self._flow_targets.items():
            target_flow = (
                target*inlet.m_dot
                if kind == 'fraction'
                else target
            )
            eqs.append((outlet.m_dot - target_flow)*m_mult)

        eqs.append((inlet.m_dot - sum(outlet.m_dot for outlet in self.outlets))*m_mult)

        for outlet in self.outlets:
            eqs.append((outlet.T - inlet.T)*T_mult)
            eqs.append((outlet.P - inlet.P)*P_mult)

            for i in range(len(inlet.z) - 1):
                eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        return eqs


class TurbineCoolingSplitter:
    """Allocate and throttle turbine coolant using the El-Masri correlation.

    Each coolant outlet corresponds to one hot-gas turbine inlet and outlet.
    The required coolant flow for a stage is

    ``K * m_hot * (T_hot - T_blade) / (T_blade - T_coolant)``.

    The inlet flow is determined by the sum of the stage cooling requirements.
    Each coolant branch is throttled isenthalpically to its corresponding
    turbine outlet pressure, while composition remains unchanged.
    """

    def __init__(
        self,
        name,
        cooling_coefficient,
        blade_temperature,
        inlet,
        outlets,
        hot_gas_inlets,
        turbine_outlets,
    ):
        outlets = tuple(outlets)
        hot_gas_inlets = tuple(hot_gas_inlets)
        turbine_outlets = tuple(turbine_outlets)

        if cooling_coefficient < 0.0:
            raise ValueError('cooling_coefficient must be nonnegative')

        self.name = name
        self.cooling_coefficient = cooling_coefficient
        self.blade_temperature = blade_temperature
        self.inlet = inlet
        self.outlets = outlets
        self.hot_gas_inlets = hot_gas_inlets
        self.turbine_outlets = turbine_outlets
        self.state_dependencies = (
            inlet,
            *outlets,
            *hot_gas_inlets,
            *turbine_outlets,
        )

    def residuals(self):
        eqs = []
        inlet = self.inlet
        temperature_difference = self.blade_temperature - inlet.T

        required_flows = []
        for outlet, hot_gas_inlet in zip(
            self.outlets,
            self.hot_gas_inlets,
        ):
            required_flow = (
                self.cooling_coefficient
                * hot_gas_inlet.m_dot
                * (hot_gas_inlet.T - self.blade_temperature)
                / temperature_difference
            )
            required_flows.append(required_flow)
            eqs.append((outlet.m_dot - required_flow)*m_mult)

        eqs.append((inlet.m_dot - sum(required_flows))*m_mult)

        for outlet, turbine_outlet in zip(
            self.outlets,
            self.turbine_outlets,
        ):
            eqs.append((outlet.P - turbine_outlet.P)*P_mult)
            eqs.append((outlet.h - inlet.h)*h_mult)

            for i in range(len(inlet.z) - 1):
                eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        return eqs


class Mod_MSHX:
    """Configurable multi-stream heat exchanger.

    ``stream_pairs`` contains ``(inlet, outlet)`` tuples.  Each approach is
    specified as ``(hot_stream, cold_stream, temperature_difference)`` and
    each fixed temperature as ``(stream, temperature)``.

    ``hot_streams`` is optional and is used only to report the heat released
    by selected streams as ``Q``."""

    def __init__(self, name, pressure_drop_percent, stream_pairs, fixed_temperatures=None, approaches=None, hot_streams=None,):
        state_dependencies = []
        for inlet, outlet in stream_pairs:
            state_dependencies.extend([inlet, outlet])

        self.name = name
        self.state_dependencies = tuple(state_dependencies)
        self.pressure_drop_percent = pressure_drop_percent
        self.stream_pairs = stream_pairs
        self.fixed_temperatures = [] if fixed_temperatures is None else fixed_temperatures
        self.approaches = [] if approaches is None else approaches
        self.hot_streams = hot_streams
        self.Q = None

    def residuals(self):
        eqs = []
        pressure_ratio = 1.0 - self.pressure_drop_percent/100.0

        for inlet, outlet in self.stream_pairs:
            eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
            eqs.append((outlet.P - inlet.P*pressure_ratio)*P_mult)
            for i in range(len(inlet.z) - 1):
                            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        for stream, temperature in self.fixed_temperatures:
            eqs.append((stream.T - temperature)*T_mult)

        for hot_stream, cold_stream, temperature_difference in self.approaches:
            eqs.append((hot_stream.T - cold_stream.T - temperature_difference)*T_mult)

        energy_balance = 0.0
        for inlet, outlet in self.stream_pairs:
            energy_balance += inlet.m_dot*(outlet.h - inlet.h)
        eqs.append(energy_balance*H_mult)

        if self.hot_streams is not None:
            self.Q = 0.0
            for inlet, outlet in self.hot_streams:
                self.Q += inlet.m_dot*(inlet.h - outlet.h)

        return eqs


class Valve:
    """Isenthalpic pressure-reduction valve."""

    def __init__(self, name, P_out, inlet, outlet):
        self.name = name
        self.state_dependencies = (inlet, outlet)
        self.P_out = P_out
        self.inlet = inlet
        self.outlet = outlet

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        eqs.append((outlet.P - self.P_out)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        eqs.append((outlet.h - inlet.h)*h_mult)
        return eqs


class Tray:
    """Equilibrium stage with liquid and vapor outlet streams."""

    def __init__(self, name, P, liquid_in, vapor_in, liquid_out, vapor_out, feeds):
        state_dependencies = []
        if liquid_in is not None:
            state_dependencies.append(liquid_in)
        if vapor_in is not None:
            state_dependencies.append(vapor_in)
        state_dependencies.extend(feeds)
        state_dependencies.extend([liquid_out, vapor_out])

        self.name = name
        self.state_dependencies = tuple(state_dependencies)
        self.P = P
        self.liquid_in = liquid_in
        self.vapor_in = vapor_in
        self.liquid_out = liquid_out
        self.vapor_out = vapor_out
        self.feeds = feeds

    def residuals(self):
        eqs = []
        L_in = self.liquid_in
        V_in = self.vapor_in
        L_out = self.liquid_out
        V_out = self.vapor_out

        inlets = []
        if L_in is not None:
            inlets.append(L_in)
        if V_in is not None:
            inlets.append(V_in)
        inlets.extend(self.feeds)

        for i in range(len(L_out.species)):
            m_in_i = 0
            for inlet in inlets:
                m_in_i += inlet.m_dot*inlet.w[i]
            m_out_i = L_out.m_dot*L_out.w[i] + V_out.m_dot*V_out.w[i]
            eqs.append((m_in_i - m_out_i)*m_mult)

        eqs.append((L_out.T - V_out.T)*T_mult)
        eqs.append((L_out.P - self.P)*P_mult)
        eqs.append((V_out.P - self.P)*P_mult)

        for i in range(len(L_out.species)):
            f_liquid = L_out.phi[i]*L_out.z[i]*L_out.P
            f_vapor = V_out.phi[i]*V_out.z[i]*V_out.P
            eqs.append((f_liquid - f_vapor)*P_mult)

        H_in = sum(inlet.m_dot*inlet.h for inlet in inlets)
        H_out = L_out.m_dot*L_out.h + V_out.m_dot*V_out.h

        eqs.append((H_in - H_out)*H_mult)
        return eqs
