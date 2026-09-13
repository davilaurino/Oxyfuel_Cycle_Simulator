"""Process and thermodynamic state classes for the ASU model."""

import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import brentq
import utils

# Fixed reference scales used for both solver variables and same-unit
# residuals. Energy and specific enthalpy have no corresponding state
# variables, so they use separate engineering reference scales.
FLOW_SCALE = 1.0                    # kg/s
TEMPERATURE_SCALE = 10.0            # K
PRESSURE_SCALE = 1.0e5              # Pa
COMPOSITION_SCALE = 1.0e-2          # mole fraction
ENERGY_SCALE = 1.0e6                # W
SPECIFIC_ENTHALPY_SCALE = 1.0e5     # J/kg

H_mult = 1.0/ENERGY_SCALE
h_mult = 1.0/SPECIFIC_ENTHALPY_SCALE
m_mult = 1.0/FLOW_SCALE
T_mult = 1.0/TEMPERATURE_SCALE
P_mult = 1.0/PRESSURE_SCALE
z_mult = 1.0/COMPOSITION_SCALE

class State:
    """A process stream backed by a reusable CoolProp HEOS state.

    Process flow and the primary thermodynamic properties use a mass basis.
    Their molar equivalents are cached for phase-equilibrium calculations.
    """

    def __init__(self, name, species, m_dot, T, P, z, phase=None):
        self.name = name

        if phase not in (None, 'liquid', 'vapor'):
            raise ValueError(
                "State phase must be None, 'liquid', or 'vapor'"
            )

        self.species = []
        for species_name in species:
            self.species.append(utils.SPS[species_name])

        self.imposed_phase = phase

        fluid_names = []
        for sp in self.species:
            fluid_names.append(sp.fluid)

        self.heos = CP.AbstractState('HEOS', '&'.join(fluid_names))

        self.update(m_dot, T, P, z)

    def variable_names(self):
        """Return solver-variable names in packing order."""
        names = [
            f'{self.name}.m_dot',
            f'{self.name}.T',
            f'{self.name}.P',
        ]
        for sp in self.species[:-1]:
            names.append(f'{self.name}.z_{sp.name}')

        return names

    def flatten_vars(self):
        """Return this state's independent solver variables."""
        return [self.m_dot, self.T, self.P, *self.z[:-1]]

    def unpack_vars(self, variables, start_idx):
        """Update this state from a solver vector and return the next index."""
        m_dot = variables[start_idx]
        T = variables[start_idx + 1]
        P = variables[start_idx + 2]

        n = len(self.z) - 1
        z = self.z.copy()
        z[:-1] = variables[start_idx + 3:start_idx + 3 + n]
        z[-1] = 1.0 - np.sum(z[:-1])

        self.update(m_dot, T, P, z)

        i = start_idx + 3 + n
        return i

    def update(self, m_dot, T, P, z):
        """Update the process state and cache mass and molar properties."""
        self.m_dot = m_dot
        self.T = T
        self.P = P
        self.z = np.array(z, dtype=float, copy=True)

        self.heos.set_mole_fractions(self.z)
        self.heos.unspecify_phase()
        if self.imposed_phase == 'liquid':
            self.heos.specify_phase(CP.iphase_liquid)
        elif self.imposed_phase == 'vapor':
            self.heos.specify_phase(CP.iphase_gas)

        self.heos.update(CP.PT_INPUTS, self.P, self.T)

        phase_names = {
            CP.iphase_liquid: 'liquid',
            CP.iphase_gas: 'vapor',
            CP.iphase_twophase: 'two-phase',
            CP.iphase_supercritical: 'supercritical',
            CP.iphase_supercritical_liquid: 'supercritical liquid',
            CP.iphase_supercritical_gas: 'supercritical vapor',
            CP.iphase_critical_point: 'critical point',
            CP.iphase_unknown: 'unknown',
        }
        self.phase = phase_names.get(self.heos.phase(), 'unknown')
        self.h = self.heos.hmass()
        self.s = self.heos.smass()
        self.rho = self.heos.rhomass()
        self.h_molar = self.heos.hmolar()
        self.s_molar = self.heos.smolar()
        self.rho_molar = self.heos.rhomolar()
        self.molar_mass = self.heos.molar_mass()
        self.n_dot = self.m_dot/self.molar_mass

        self.w = np.zeros(len(self.species))
        for i, sp in enumerate(self.species):
            self.w[i] = self.z[i]*sp.MW/self.molar_mass

        self.vapor_fraction = self.heos.Q()
        if self.imposed_phase in ('liquid', 'vapor'):
            self.phi = np.zeros(len(self.species))
            for i in range(len(self.species)):
                self.phi[i] = self.heos.fugacity_coefficient(i)
        else:
            self.phi = None

class Input:
    """Fixed inlet conditions applied to an outlet process stream."""

    def __init__(self, name, T, P, z, outlet, m_dot=None):
        self.name = name
        self.T = T
        self.P = P
        self.z = np.array(z, dtype=float, copy=True)
        self.outlet = outlet
        self.m_dot = m_dot

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

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

class Compressor:
    """Adiabatic gas compressor with an exact gas isentropic flash."""

    def __init__(
        self,
        name,
        efficiency,
        pressure_ratio,
        inlet,
        outlet,
        W=None,
    ):
        self.name = name
        self.efficiency = efficiency
        self.pressure_ratio = pressure_ratio
        self.inlet = inlet
        self.outlet = outlet
        self.W = W

        fluid_names = []
        for sp in inlet.species:
            fluid_names.append(sp.fluid)
        self.heos = CP.AbstractState('HEOS', '&'.join(fluid_names))

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def isentropic_enthalpy(self, P, s, z):
        self.heos.set_mole_fractions(z)
        self.heos.unspecify_phase()
        self.heos.specify_phase(CP.iphase_gas)
        self.heos.update(CP.PSmass_INPUTS, P, s)
        return self.heos.hmass()

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet
        P_out = inlet.P*self.pressure_ratio

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        for i in range(len(inlet.species) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        eqs.append((outlet.P - P_out)*P_mult)
        h_iso = self.isentropic_enthalpy(P_out, inlet.s, outlet.z)
        eqs.append(((outlet.h - inlet.h) - (h_iso - inlet.h)/self.efficiency)*h_mult)

        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs

class Turbine:
    """Adiabatic turbine with a phase-aware isentropic reference flash."""

    def __init__(
        self,
        name,
        efficiency,
        P_out,
        inlet,
        outlet,
        W=None,
    ):
        self.name = name
        self.efficiency = efficiency
        self.P_out = P_out
        self.inlet = inlet
        self.outlet = outlet
        self.W = W

        fluid_names = []
        for sp in inlet.species:
            fluid_names.append(sp.fluid)
        self.heos = CP.AbstractState('HEOS', '&'.join(fluid_names))

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def isentropic_enthalpy(self, P, s, z):
        self.heos.set_mole_fractions(z)
        self.heos.unspecify_phase()

        self.heos.update(CP.PQ_INPUTS, P, 1.0)
        s_dew = self.heos.smass()

        if s >= s_dew:
            self.heos.specify_phase(CP.iphase_gas)
            self.heos.update(CP.PSmass_INPUTS, P, s)
            return self.heos.hmass()

        def entropy_residual(Q):
            self.heos.unspecify_phase()
            self.heos.update(CP.PQ_INPUTS, P, Q)
            return self.heos.smass() - s

        Q = brentq(entropy_residual, 0.0, 1.0)
        self.heos.unspecify_phase()
        self.heos.update(CP.PQ_INPUTS, P, Q)
        return self.heos.hmass()

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        for i in range(len(inlet.species) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        eqs.append((outlet.P - self.P_out)*P_mult)
        h_iso = self.isentropic_enthalpy(self.P_out, inlet.s, inlet.z)
        eqs.append(
            ((inlet.h - outlet.h) - self.efficiency*(inlet.h - h_iso))*h_mult
        )

        # Positive work denotes power generated by the expansion.
        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs

class LOXPump:
    """Liquid-oxygen pump with an exact liquid isentropic reference state."""

    def __init__(
        self,
        name,
        efficiency,
        P_out,
        inlet,
        outlet,
        W=None,
    ):
        self.name = name
        self.efficiency = efficiency
        self.P_out = P_out
        self.inlet = inlet
        self.outlet = outlet
        self.W = W

        if inlet.imposed_phase != 'liquid' or outlet.imposed_phase != 'liquid':
            raise ValueError('LOXPump requires liquid-imposed inlet and outlet states')

        fluid_names = []
        for sp in inlet.species:
            fluid_names.append(sp.fluid)
        self.heos = CP.AbstractState('HEOS', '&'.join(fluid_names))

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def isentropic_enthalpy(self, P, s, z):
        self.heos.set_mole_fractions(z)
        self.heos.unspecify_phase()
        self.heos.specify_phase(CP.iphase_liquid)
        self.heos.update(CP.PSmass_INPUTS, P, s)
        return self.heos.hmass()

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        for i in range(len(inlet.species) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        eqs.append((outlet.P - self.P_out)*P_mult)
        h_iso = self.isentropic_enthalpy(self.P_out, inlet.s, inlet.z)
        eqs.append(((outlet.h - inlet.h) - (h_iso - inlet.h)/self.efficiency)*h_mult)

        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs

class Intercooler:
    """Cooler with a specified outlet temperature and pressure drop."""

    def __init__(
        self,
        name,
        T_out,
        pressure_drop_percent,
        inlet,
        outlet,
        Q=None,
    ):
        self.name = name
        self.T_out = T_out
        self.pressure_drop_percent = pressure_drop_percent
        self.inlet = inlet
        self.outlet = outlet
        self.Q = Q

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet = self.outlet
        pressure_ratio = 1.0 - self.pressure_drop_percent/100.0

        eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
        for i in range(len(inlet.species) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        eqs.append((outlet.P - inlet.P*pressure_ratio)*P_mult)
        eqs.append((outlet.T - self.T_out)*T_mult)

        self.Q = inlet.m_dot*(outlet.h - inlet.h)
        return eqs

class ReboilerCondenser:
    """Total condenser coupled to a partial reboiler with a heat loss.

    Each duty is calculated from its stream enthalpy change, and their
    difference is constrained by the specified condenser heat-loss fraction.
    """

    def __init__(
        self,
        name,
        condenser_vapor_in,
        condenser_liquid_out,
        reboiler_liquid_in,
        reboiler_liquid_out,
        reboiler_vapor_out,
        T_condenser_out,
        condenser_heat_loss_fraction,
    ):
        if not 0.0 <= condenser_heat_loss_fraction < 1.0:
            raise ValueError(
                'condenser_heat_loss_fraction must be between 0 and 1'
            )

        self.name = name
        self.condenser_vapor_in = condenser_vapor_in
        self.condenser_liquid_out = condenser_liquid_out
        self.reboiler_liquid_in = reboiler_liquid_in
        self.reboiler_liquid_out = reboiler_liquid_out
        self.reboiler_vapor_out = reboiler_vapor_out
        self.T_condenser_out = T_condenser_out
        self.condenser_heat_loss_fraction = condenser_heat_loss_fraction
        self.Q_condenser = None
        self.Q_reboiler = None

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        condenser_vapor_in = self.condenser_vapor_in
        condenser_liquid_out = self.condenser_liquid_out
        reboiler_liquid_in = self.reboiler_liquid_in
        reboiler_liquid_out = self.reboiler_liquid_out
        reboiler_vapor_out = self.reboiler_vapor_out
        self.Q_condenser = (
            condenser_liquid_out.m_dot*condenser_liquid_out.h
            - condenser_vapor_in.m_dot*condenser_vapor_in.h
        )
        self.Q_reboiler = (
            reboiler_liquid_out.m_dot*reboiler_liquid_out.h
            + reboiler_vapor_out.m_dot*reboiler_vapor_out.h
            - reboiler_liquid_in.m_dot*reboiler_liquid_in.h
        )

        transferred_fraction = 1.0 - self.condenser_heat_loss_fraction
        eqs.append((
            self.Q_reboiler
            - transferred_fraction*(-self.Q_condenser)
        )*H_mult)

        eqs.append(
            (condenser_vapor_in.m_dot - condenser_liquid_out.m_dot)*m_mult
        )
        eqs.append(
            (condenser_liquid_out.T - self.T_condenser_out)*T_mult
        )
        eqs.append(
            (condenser_liquid_out.P - condenser_vapor_in.P)*P_mult
        )

        for i in range(len(condenser_liquid_out.z) - 1):
            eqs.append(
                (condenser_liquid_out.z[i] - condenser_vapor_in.z[i])*z_mult
            )

        for i in range(len(reboiler_liquid_in.species)):
            m_in_i = reboiler_liquid_in.m_dot*reboiler_liquid_in.w[i]
            m_out_i = (
                reboiler_liquid_out.m_dot*reboiler_liquid_out.w[i]
                + reboiler_vapor_out.m_dot*reboiler_vapor_out.w[i]
            )
            eqs.append((m_in_i - m_out_i)*m_mult)

        eqs.append((reboiler_liquid_out.T - reboiler_vapor_out.T)*T_mult)
        eqs.append((reboiler_liquid_out.P - reboiler_liquid_in.P)*P_mult)
        eqs.append((reboiler_vapor_out.P - reboiler_liquid_in.P)*P_mult)

        for i in range(len(reboiler_liquid_in.species)):
            f_liquid = (
                reboiler_liquid_out.phi[i]
                * reboiler_liquid_out.z[i]
                * reboiler_liquid_out.P
            )
            f_vapor = (
                reboiler_vapor_out.phi[i]
                * reboiler_vapor_out.z[i]
                * reboiler_vapor_out.P
            )
            eqs.append((f_liquid - f_vapor)*P_mult)

        return eqs


class O2Specification:
    """Flowsheet-level oxygen purity and recovery specifications."""

    def __init__(
        self,
        name,
        air_feed,
        o2_product,
        purity_target,
        recovery_target,
        o2_index=0,
    ):
        if not 0.0 < purity_target <= 1.0:
            raise ValueError('purity_target must be between 0 and 1')
        if not 0.0 < recovery_target <= 1.0:
            raise ValueError('recovery_target must be between 0 and 1')

        self.name = name
        self.air_feed = air_feed
        self.o2_product = o2_product
        self.purity_target = purity_target
        self.recovery_target = recovery_target
        self.o2_index = o2_index

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def residuals(self):
        feed_o2_flow = self.air_feed.m_dot*self.air_feed.w[self.o2_index]
        product_o2_flow = (
            self.o2_product.m_dot*self.o2_product.w[self.o2_index]
        )
        recovery = product_o2_flow/feed_o2_flow

        return [
            (self.o2_product.z[self.o2_index] - self.purity_target)*z_mult,
            (recovery - self.recovery_target)*z_mult,
        ]

class Splitter:
    """Divide one process stream into two streams without separation."""

    def __init__(self, name, inlet, outlet1, outlet2, split_fraction=None):
        self.name = name
        self.inlet = inlet
        self.outlet1 = outlet1
        self.outlet2 = outlet2
        self.split_fraction = split_fraction

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet1 = self.outlet1
        outlet2 = self.outlet2

        if self.split_fraction is not None:
            eqs.append(
                (outlet1.m_dot - self.split_fraction*inlet.m_dot)*m_mult
            )

        eqs.append((inlet.m_dot - outlet1.m_dot - outlet2.m_dot)*m_mult)
        eqs.append((outlet1.T - inlet.T)*T_mult)
        eqs.append((outlet2.T - inlet.T)*T_mult)
        eqs.append((outlet1.P - inlet.P)*P_mult)
        eqs.append((outlet2.P - inlet.P)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet1.z[i] - inlet.z[i])*z_mult)
            eqs.append((outlet2.z[i] - inlet.z[i])*z_mult)

        return eqs

class HeatExchanger:
    """Three-stream heat exchanger with two hot streams and one cold stream."""

    def __init__(
        self,
        name,
        pressure_drop_percent,
        T_diff_hot1,
        T_diff_cold,
        hot1_in,
        hot2_in,
        cold_in,
        hot1_out,
        hot2_out,
        cold_out,
    ):
        self.name = name
        self.pressure_drop_percent = pressure_drop_percent
        self.T_diff_hot1 = T_diff_hot1
        self.T_diff_cold = T_diff_cold
        self.hot1_in = hot1_in
        self.hot2_in = hot2_in
        self.cold_in = cold_in
        self.hot1_out = hot1_out
        self.hot2_out = hot2_out
        self.cold_out = cold_out
        self.Q = None

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def residuals(self):
        eqs = []

        stream_pairs = [
            (self.hot1_in, self.hot1_out),
            (self.hot2_in, self.hot2_out),
            (self.cold_in, self.cold_out),
        ]

        pressure_ratio = 1.0 - self.pressure_drop_percent/100.0
        for inlet, outlet in stream_pairs:
            eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
            eqs.append((outlet.P - inlet.P*pressure_ratio)*P_mult)

            for i in range(len(inlet.z) - 1):
                eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        eqs.append((self.hot1_out.T - self.cold_in.T - self.T_diff_hot1)*T_mult)
        eqs.append((self.cold_out.T - self.hot1_in.T + self.T_diff_cold)*T_mult)

        energy_balance = 0
        for inlet, outlet in stream_pairs:
            energy_balance += inlet.m_dot*(outlet.h - inlet.h)
        eqs.append(energy_balance*H_mult)

        self.Q = (
            self.hot1_in.m_dot*(self.hot1_in.h - self.hot1_out.h)
            + self.hot2_in.m_dot*(self.hot2_in.h - self.hot2_out.h)
        )

        return eqs

class HX1:
    """Four-stream main exchanger with warm-end product pinch targets.

    The HPC air outlet is determined by the overall energy balance and is free
    to enter the two-phase region. The LPC air outlet target protects the
    downstream turbine inlet condition. Product outlet temperatures approach
    their corresponding warm air inlet temperatures by specified differences.
    """

    def __init__(
        self,
        name,
        pressure_drop_percent,
        T_lpc_air_out,
        T_diff_n2,
        T_diff_o2,
        hpc_air_in,
        lpc_air_in,
        n2_in,
        o2_in,
        hpc_air_out,
        lpc_air_out,
        n2_out,
        o2_out,
    ):
        self.name = name
        self.pressure_drop_percent = pressure_drop_percent
        self.T_lpc_air_out = T_lpc_air_out
        self.T_diff_n2 = T_diff_n2
        self.T_diff_o2 = T_diff_o2
        self.hpc_air_in = hpc_air_in
        self.lpc_air_in = lpc_air_in
        self.n2_in = n2_in
        self.o2_in = o2_in
        self.hpc_air_out = hpc_air_out
        self.lpc_air_out = lpc_air_out
        self.n2_out = n2_out
        self.o2_out = o2_out
        self.Q = None

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        stream_pairs = [
            (self.hpc_air_in, self.hpc_air_out),
            (self.lpc_air_in, self.lpc_air_out),
            (self.n2_in, self.n2_out),
            (self.o2_in, self.o2_out),
        ]

        pressure_ratio = 1.0 - self.pressure_drop_percent/100.0
        for inlet, outlet in stream_pairs:
            eqs.append((outlet.m_dot - inlet.m_dot)*m_mult)
            eqs.append((outlet.P - inlet.P*pressure_ratio)*P_mult)

            for i in range(len(inlet.z) - 1):
                eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        eqs.append((self.lpc_air_out.T - self.T_lpc_air_out)*T_mult)
        eqs.append((self.n2_out.T - self.hpc_air_in.T + self.T_diff_n2)*T_mult)
        eqs.append((self.o2_out.T - self.lpc_air_in.T + self.T_diff_o2)*T_mult)

        energy_balance = 0.0
        for inlet, outlet in stream_pairs:
            energy_balance += inlet.m_dot*(outlet.h - inlet.h)
        eqs.append(energy_balance*H_mult)

        self.Q = (
            self.hpc_air_in.m_dot*(self.hpc_air_in.h - self.hpc_air_out.h)
            + self.lpc_air_in.m_dot*(self.lpc_air_in.h - self.lpc_air_out.h)
        )

        return eqs

class Valve:
    """Isenthalpic pressure-reduction valve."""

    def __init__(self, name, P_out, inlet, outlet):
        self.name = name
        self.P_out = P_out
        self.inlet = inlet
        self.outlet = outlet

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

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
        self.name = name
        self.P = P
        self.liquid_in = liquid_in
        self.vapor_in = vapor_in
        self.liquid_out = liquid_out
        self.vapor_out = vapor_out
        self.feeds = feeds

    def flatten_vars(self):
        return []

    def unpack_vars(self, variables, start_idx):
        return start_idx

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

        H_in = 0
        for inlet in inlets:
            H_in += inlet.m_dot*inlet.h
        H_out = L_out.m_dot*L_out.h + V_out.m_dot*V_out.h
        eqs.append((H_in - H_out)*H_mult)

        eqs.append((L_out.T - V_out.T)*T_mult)
        eqs.append((L_out.P - self.P)*P_mult)
        eqs.append((V_out.P - self.P)*P_mult)

        for i in range(len(L_out.species)):
            f_liquid = L_out.phi[i]*L_out.z[i]*L_out.P
            f_vapor = V_out.phi[i]*V_out.z[i]*V_out.P
            eqs.append((f_liquid - f_vapor)*P_mult)

        return eqs
