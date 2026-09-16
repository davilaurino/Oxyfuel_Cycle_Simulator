"""Process and thermodynamic state classes for the ASU model."""

import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import brentq
import utils

# Fixed reference scales used for both solver variables and same-unit residuals. 
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

PHASE_NAMES = {
    CP.iphase_liquid: 'liquid',
    CP.iphase_gas: 'vapor',
    CP.iphase_twophase: 'two-phase',
    CP.iphase_supercritical: 'supercritical',
    CP.iphase_supercritical_liquid: 'supercritical liquid',
    CP.iphase_supercritical_gas: 'supercritical vapor',
    CP.iphase_critical_point: 'critical point',
    CP.iphase_unknown: 'unknown',
}


def new_heos(species):
    """Create a reusable HEOS state for a list of species."""
    fluids = '&'.join(sp.fluid for sp in species)
    return CP.AbstractState('HEOS', fluids)


class State:
    """A process stream backed by a reusable CoolProp HEOS state.

    Process flow and the primary thermodynamic properties use a mass basis.
    Mass fractions and phase-equilibrium data are cached for downstream units."""

    def __init__(self, name, species, m_dot, T, P, z, phase=None):
        self.name = name

        if phase not in (None, 'liquid', 'vapor'):
            raise ValueError("State phase must be None, 'liquid', or 'vapor'")

        self.species = []
        for species_name in species:
                self.species.append(utils.SPS[species_name])

        self.imposed_phase = phase
        self.w = np.zeros(len(self.species))
        if self.imposed_phase in ('liquid', 'vapor'):
            self.phi = np.zeros(len(self.species))
        else:
            self.phi = None

        self.heos = new_heos(self.species)
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
        z[:-1] = variables[
            start_idx + 3:start_idx + 3 + n_composition
        ]
        z[-1] = 1.0 - np.sum(z[:-1])

        self.update(m_dot, T, P, z)

        return start_idx + 3 + n_composition

    def update(self, m_dot, T, P, z):
        """Update the process state and cache its thermodynamic properties."""
        self.m_dot = m_dot
        self.T = T
        self.P = P
        self.z = np.array(z, dtype=float, copy=True)

        self.heos.set_mole_fractions(self.z)
        if self.imposed_phase == 'liquid':
            self.heos.specify_phase(CP.iphase_liquid)
        elif self.imposed_phase == 'vapor':
            self.heos.specify_phase(CP.iphase_gas)
        else:
            self.heos.unspecify_phase()

        self.heos.update(CP.PT_INPUTS, self.P, self.T)

        self.phase = PHASE_NAMES.get(self.heos.phase(), 'unknown')
        self.h = self.heos.hmass()
        self.s = self.heos.smass()
        self.molar_mass = self.heos.molar_mass()

        for i, (z_i, sp) in enumerate(zip(self.z, self.species)):
            self.w[i] = z_i*sp.MW/self.molar_mass

        self.vapor_fraction = self.heos.Q()
        if self.imposed_phase in ('liquid', 'vapor'):
            for i in range(len(self.species)):
                self.phi[i] = self.heos.fugacity_coefficient(i)


class Input:
    """Fixed inlet conditions applied to an outlet process stream."""

    def __init__(self, name, T, P, z, outlet, m_dot=None):
        self.name = name
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


class Compressor:
    """Adiabatic gas compressor with an exact gas isentropic flash."""

    def __init__(self, name, efficiency, pressure_ratio, inlet, outlet):
        self.name = name
        self.efficiency = efficiency
        self.pressure_ratio = pressure_ratio
        self.inlet = inlet
        self.outlet = outlet
        self.W = None

        self.heos = new_heos(inlet.species)

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
        eqs.append((outlet.P - P_out)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        h_iso = self.isentropic_enthalpy(P_out, inlet.s, outlet.z)
        eqs.append(((outlet.h - inlet.h) - (h_iso - inlet.h)/self.efficiency)*h_mult)

        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs


class Turbine:
    """Adiabatic turbine with a phase-aware isentropic reference flash."""

    def __init__(self, name, efficiency, P_out, inlet, outlet):
        self.name = name
        self.efficiency = efficiency
        self.P_out = P_out
        self.inlet = inlet
        self.outlet = outlet
        self.W = None

        self.heos = new_heos(inlet.species)

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
        eqs.append((outlet.P - self.P_out)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        h_iso = self.isentropic_enthalpy(self.P_out, inlet.s, inlet.z)
        eqs.append(((inlet.h - outlet.h) - self.efficiency*(inlet.h - h_iso))*h_mult)

        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs


class LOXPump:
    """Liquid-oxygen pump with an exact liquid isentropic reference state."""

    def __init__(self, name, efficiency, P_out, inlet, outlet):
        self.name = name
        self.efficiency = efficiency
        self.P_out = P_out
        self.inlet = inlet
        self.outlet = outlet
        self.W = None

        if inlet.imposed_phase != 'liquid' or outlet.imposed_phase != 'liquid':
            raise ValueError('LOXPump requires liquid-imposed inlet and outlet states')

        self.heos = new_heos(inlet.species)

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
        eqs.append((outlet.P - self.P_out)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet.z[i] - inlet.z[i])*z_mult)

        h_iso = self.isentropic_enthalpy(self.P_out, inlet.s, inlet.z)
        eqs.append(((outlet.h - inlet.h) - (h_iso - inlet.h)/self.efficiency)*h_mult)

        self.W = inlet.m_dot*(inlet.h - outlet.h)
        return eqs


class Intercooler:
    """Cooler with a specified outlet temperature and pressure drop."""

    def __init__(self, name, T_out, pressure_drop_percent, inlet, outlet):
        self.name = name
        self.T_out = T_out
        self.pressure_drop_percent = pressure_drop_percent
        self.inlet = inlet
        self.outlet = outlet
        self.Q = None

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

        self.Q = inlet.m_dot*(outlet.h - inlet.h)
        return eqs


class ReboilerCondenser:
    """Coupled total condenser and partial reboiler with equal duties."""

    def __init__(self, name, condenser_vapor_in, condenser_liquid_out, reboiler_liquid_in, reboiler_liquid_out, reboiler_vapor_out, T_cold_pinch):
        self.name = name
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

    def __init__(self, name, o2_product, purity_target):
        if not 0.0 < purity_target <= 1.0:
            raise ValueError('purity_target must be between 0 and 1')

        self.name = name
        self.o2_product = o2_product
        self.purity_target = purity_target

    def residuals(self):
        eqs = []
        product = self.o2_product

        for i, species in enumerate(product.species):
            if species.name == 'O2':
                o2_index = i
                break

        eqs.append((product.z[o2_index] - self.purity_target)*z_mult)
        return eqs


class Splitter:
    """Divide one process stream into two streams without separation."""

    def __init__(self, name, inlet, outlet1, outlet2, split_fraction=None):
        self.name = name
        self.inlet = inlet
        self.outlet1 = outlet1
        self.outlet2 = outlet2
        self.split_fraction = split_fraction

    def residuals(self):
        eqs = []
        inlet = self.inlet
        outlet1 = self.outlet1
        outlet2 = self.outlet2

        if self.split_fraction is not None:
            eqs.append((outlet1.m_dot - self.split_fraction*inlet.m_dot)*m_mult)

        eqs.append((inlet.m_dot - outlet1.m_dot - outlet2.m_dot)*m_mult)
        eqs.append((outlet1.T - inlet.T)*T_mult)
        eqs.append((outlet2.T - inlet.T)*T_mult)
        eqs.append((outlet1.P - inlet.P)*P_mult)
        eqs.append((outlet2.P - inlet.P)*P_mult)

        for i in range(len(inlet.z) - 1):
            eqs.append((outlet1.z[i] - inlet.z[i])*z_mult)
            eqs.append((outlet2.z[i] - inlet.z[i])*z_mult)

        return eqs


class Mod_MSHX:
    """Configurable multi-stream heat exchanger.

    ``stream_pairs`` contains ``(inlet, outlet)`` tuples.  Each approach is
    specified as ``(hot_stream, cold_stream, temperature_difference)`` and
    each fixed temperature as ``(stream, temperature)``.

    ``hot_streams`` is optional and is used only to report the heat released
    by selected streams as ``Q``."""

    def __init__(self, name, pressure_drop_percent, stream_pairs, fixed_temperatures=None, approaches=None, hot_streams=None,):
        self.name = name
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
        self.name = name
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
