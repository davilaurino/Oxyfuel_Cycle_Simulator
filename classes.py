import numpy as np
import utils

# Scaling factors for residuals
H_mult = 1e-7
h_mult = 1e-5
s_mult = 1e-1
m_mult = 1e-1
T_mult = 1e-2
P_mult = 1e-4
y_mult = 1e1

class State:
    def __init__(self, Stream, Fluid, m_dot=None, T=None, P=None, v=None, h=None, s=None, y=None, spc=None):
        self.Stream = Stream
        self.Fluid = Fluid
        self.m_dot = m_dot
        self.T = T
        self.P = P
        self.v = v
        self.h = h
        self.s = s
        self.y = y
        if spc is not None:
            self.spc = list(spc)
        else:
            self.spc = list(utils.PRODUCT_SPECIES)

    def flatten_vars(self):
        return [self.m_dot, self.T, self.P, *self.y]
    
    def unpack_vars(self, vars, start_idx):
        n = len(self.y)
        self.m_dot = vars[start_idx]
        self.T = vars[start_idx + 1]
        self.P = vars[start_idx + 2]
        self.y[:] = vars[start_idx + 3:start_idx + 3 + n]
                
        return start_idx + 3 + n

    def to_dict(self):
        self.h = utils.mixture_enthalpy(self.T, self.P, self.y, self.spc)
        self.s = utils.mixture_entropy(self.T, self.P, self.y, self.spc)
        if any((spc is utils.SPS['CO2']) for spc in self.spc):
            self.v = utils.CO2_volume(self.T, self.P)
        else:
            self.v = 0

        d = {
            "Stream": self.Stream,
            "T": round(self.T, 2),
            "P": round(self.P, 1),
            "v (CO2)": round(self.v, 4),
            "m_dot": round(self.m_dot, 2),
        }

        spc_dict = {spc: yi for spc, yi in zip(self.spc, self.y)}
        for spc in utils.SPS.values():
            d[f"y_{spc.name}"] = round(spc_dict.get(spc, 0.0), 4)

        return d

class Input:
    def __init__(self, Name, T, P, y, Outlet, m_dot=None):
        self.Name = Name
        self.T = T
        self.P = P
        self.y = list(y)
        self.Outlet = Outlet
        self.m_dot = m_dot

    def flatten_vars(self):
        return []
    
    def unpack_vars(self, vars, start_idx):
        return start_idx
    
    def residuals(self):
        eqs = []
        S_out = self.Outlet

        if self.m_dot is not None:
            eqs.append((S_out.m_dot - self.m_dot)*m_mult)

        eqs.append((S_out.T - self.T)*T_mult)
        eqs.append((S_out.P - self.P)*P_mult)
        for i in range(len(self.y)):
            eqs.append((S_out.y[i] - self.y[i])*y_mult)

        return eqs

class Compressor:
    def __init__(self, Name, eta, rp_comp, Inlet, Outlet, T_iso=None, W=None):
        self.Name = Name
        self.eta = eta
        self.rp_comp = rp_comp
        self.Inlet = Inlet
        self.Outlet = Outlet
        self.T_iso = T_iso
        self.W = W
    
    def flatten_vars(self):
        return [self.T_iso]
    
    def unpack_vars(self, vars, start_idx):
        self.T_iso = vars[start_idx]
        return start_idx + 1
    
    def residuals(self):
        eqs = []
        S_in = self.Inlet
        S_out = self.Outlet

        P_out = S_in.P*self.rp_comp

        eqs.append((S_out.m_dot - S_in.m_dot)*m_mult)
        for i in range(len(S_in.y)):
            eqs.append((S_out.y[i] - S_in.y[i])*y_mult)

        s_in = utils.mixture_entropy(S_in.T, S_in.P, S_in.y, S_in.spc)
        s_iso = utils.mixture_entropy(self.T_iso, P_out, S_out.y, S_out.spc)
        h_iso = utils.mixture_enthalpy(self.T_iso, P_out, S_out.y, S_out.spc)

        eqs.append((S_out.P - P_out)*P_mult)
        eqs.append((s_iso - s_in)*s_mult)
        eqs.append(((S_out.h - S_in.h) - (h_iso - S_in.h)/self.eta)*h_mult)

        self.W = S_in.m_dot*(S_in.h - S_out.h)

        return eqs

class Pump:
    def __init__(self, Name, eta, P_out, Inlet, Outlet, T_iso=None, W=None):
        self.Name = Name
        self.eta = eta
        self.P_out = P_out
        self.Inlet = Inlet
        self.Outlet = Outlet
        self.T_iso = T_iso
        self.W = W

    def flatten_vars(self):
        return [self.T_iso]

    def unpack_vars(self, vars, start_idx):
        self.T_iso = vars[start_idx]
        return start_idx + 1

    def residuals(self):
        eqs = []
        S_in = self.Inlet
        S_out = self.Outlet

        P_out = self.P_out

        eqs.append((S_out.m_dot - S_in.m_dot)*m_mult)
        for i in range(len(S_in.y)):
            eqs.append((S_out.y[i] - S_in.y[i])*y_mult)

        s_in  = utils.mixture_entropy(S_in.T, S_in.P, S_in.y, S_in.spc)
        s_iso = utils.mixture_entropy(self.T_iso, P_out, S_out.y, S_out.spc)
        h_iso = utils.mixture_enthalpy(self.T_iso, P_out, S_out.y, S_out.spc)

        eqs.append((S_out.P - P_out)*P_mult)
        eqs.append((s_iso - s_in)*s_mult)
        eqs.append(((S_out.h - S_in.h) - (h_iso - S_in.h)/self.eta)*h_mult)

        self.W = S_in.m_dot*(S_in.h - S_out.h)

        return eqs

class Intercooler:
    def __init__(self, Name, T_resf, Perc_Delta_P, Inlet, Outlet, Q=None):
        self.Name = Name
        self.T_resf = T_resf
        self.Perc_Delta_P = Perc_Delta_P
        self.Inlet = Inlet
        self.Outlet = Outlet
        self.Q = Q

    def flatten_vars(self):
        return []
    
    def unpack_vars(self, vars, start_idx):
        return start_idx
    
    def residuals(self):
        eqs = []
        S_in = self.Inlet
        S_out = self.Outlet

        eqs.append((S_in.m_dot - S_out.m_dot)*m_mult)
        for i in range(len(S_in.y)):
            eqs.append((S_out.y[i] - S_in.y[i])*y_mult)

        eqs.append((S_out.P - S_in.P*(1 - self.Perc_Delta_P/100))*P_mult)
        eqs.append((S_out.T - self.T_resf)*T_mult)
        
        self.Q = S_in.m_dot * (S_out.h - S_in.h)

        return eqs

class Mixer:
    def __init__(self, Name, Inlet1, Inlet2, Outlet):
        self.Name = Name
        self.Inlet1 = Inlet1
        self.Inlet2 = Inlet2
        self.Outlet = Outlet

    def flatten_vars(self):
        return []
    
    def unpack_vars(self, vars, start_idx):
        return start_idx
    
    def residuals(self):
        eqs = []
        S_in1 = self.Inlet1
        S_in2 = self.Inlet2
        S_out = self.Outlet

        all_spc = []
        for sp in (S_in1.spc + S_in2.spc):
            if sp not in all_spc:
                all_spc.append(sp)

        P_out = min(S_in1.P, S_in2.P)

        MW_in1, x_in1 = utils.mass_fraction(S_in1.y, S_in1.spc)
        MW_in2, x_in2 = utils.mass_fraction(S_in2.y, S_in2.spc)

        n_in1 = S_in1.m_dot/MW_in1
        n_in2 = S_in2.m_dot/MW_in2
        n_out = n_in1 + n_in2

        n_in1_dict = {spc: n_in1*yi for (spc, yi) in zip(S_in1.spc, S_in1.y)}
        n_in2_dict = {spc: n_in2*yi for (spc, yi) in zip(S_in2.spc, S_in2.y)}

        y_out = []
        for spc in all_spc:
            n_out_spc = n_in1_dict.get(spc, 0) + n_in2_dict.get(spc, 0)
            y_out.append(n_out_spc/n_out)

        H_in1 = S_in1.m_dot*S_in1.h
        H_in2 = S_in2.m_dot*S_in2.h
        H_out = S_out.m_dot*S_out.h

        eqs.append((S_out.m_dot - S_in1.m_dot - S_in2.m_dot)*m_mult)
        eqs.append((H_out - H_in1 - H_in2)*H_mult)
        eqs.append((S_out.P - P_out)*P_mult)
        for i, spc in enumerate(all_spc):
            eqs.append((S_out.y[i] - y_out[i])*y_mult)

        return eqs

class HX_mult:
    def __init__(self, Name, Perc_Delta_P, TC_diff1, T_C3_out, T_H_out, Inlet_C1, Inlet_C2, Inlet_C3, Inlet_H,
                 Outlet_C1, Outlet_C2, Outlet_C3, Outlet_H, Q=None):
        self.Name = Name
        self.Perc_Delta_P = Perc_Delta_P
        self.TC_diff1 = TC_diff1
        self.T_C3_out = T_C3_out
        self.T_H_out = T_H_out
        self.Inlet_C1 = Inlet_C1
        self.Inlet_C2 = Inlet_C2
        self.Inlet_C3 = Inlet_C3
        self.Inlet_H = Inlet_H
        self.Outlet_C1 = Outlet_C1
        self.Outlet_C2 = Outlet_C2
        self.Outlet_C3 = Outlet_C3
        self.Outlet_H = Outlet_H
        self.Q = Q

    def flatten_vars(self):
        return []

    def unpack_vars(self, vars, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        S_C1_in = self.Inlet_C1
        S_C2_in = self.Inlet_C2
        S_C3_in = self.Inlet_C3
        S_H_in  = self.Inlet_H
        S_C1_out = self.Outlet_C1
        S_C2_out = self.Outlet_C2
        S_C3_out = self.Outlet_C3
        S_H_out  = self.Outlet_H

        stream_pairs = [
            (S_C1_in, S_C1_out),
            (S_C2_in, S_C2_out),
            (S_C3_in, S_C3_out),
            (S_H_in,  S_H_out),
        ]

        for (S_in, S_out) in stream_pairs:
            eqs.append((S_out.m_dot - S_in.m_dot)*m_mult)
            eqs.append((S_out.P - S_in.P*(1 - self.Perc_Delta_P/100))*P_mult)
            for i in range(len(S_in.y)):
                eqs.append((S_out.y[i] - S_in.y[i])*y_mult)

        eqs.append((S_C1_out.T - (S_H_in.T - self.TC_diff1))*T_mult)
        eqs.append((S_C3_out.T - self.T_C3_out)*T_mult)
        eqs.append((S_H_out.T  - self.T_H_out)*T_mult)

        # Energy balance — determines S_C2_out.T (oxidant outlet)
        Q_balance = 0
        for (S_in, S_out) in stream_pairs:
            Q_balance += S_in.m_dot*(S_out.h - S_in.h)
        eqs.append(Q_balance*H_mult)

        self.Q = S_H_in.m_dot*(S_H_in.h - S_H_out.h)

        return eqs

class Combustor:
    def __init__(self, Name, fuel_Inlet, Inlet=None, Outlet=None, exc_O2=None):
        self.Name = Name
        self.fuel_Inlet = fuel_Inlet
        self.Inlet = Inlet
        self.Outlet = Outlet
        self.exc_O2 = exc_O2
    
    def flatten_vars(self):
        return []
    
    def unpack_vars(self, vars, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        S_in = self.Inlet
        S_fuel = self.fuel_Inlet
        S_out = self.Outlet

        MW_fuel, n_fuel, n_C, n_H = utils.stoichiometry(S_fuel)
        n_O2_sto = n_C + n_H/4

        # Molar flows in
        M_in, x_in = utils.mass_fraction(S_in.y, S_in.spc)

        n_in = S_in.m_dot/M_in

        n_fuel_dict = {spc: n_fuel*yi for (spc, yi) in zip(S_fuel.spc, S_fuel.y)}
        n_in_dict = {spc: n_in*yi for (spc, yi) in zip(S_in.spc, S_in.y)}

        # Excess O2 equation
        if self.exc_O2 is not None:
            n_O2_exc = (n_O2_sto*self.exc_O2)
            y_in_O2 = n_O2_exc/n_in
            eqs.append((S_in.y[2] - y_in_O2)*y_mult)

        # Molar flows out
        n_out_dict = {}
        for spc in S_out.spc:
            if spc is utils.SPS['CO2']:
                n_out_dict[spc] = n_C + n_in_dict.get(spc, 0) + n_fuel_dict.get(spc, 0)
            elif spc is utils.SPS['H2O']:
                n_out_dict[spc] = n_H/2 + n_in_dict.get(spc, 0)
            elif spc is utils.SPS['O2']:
                n_out_dict[spc] = n_in_dict.get(spc, 0) - n_O2_sto
            else:
                n_out_dict[spc] = n_in_dict.get(spc, 0)

        n_out = 0
        for spc in S_out.spc:
            n_out += n_out_dict[spc]    

        # Molar fractions out
        y_out = []
        for spc in S_out.spc:
            y_out.append(n_out_dict[spc]/n_out)

        m_out = S_in.m_dot + S_fuel.m_dot

        # Conservation residuals
        eqs.append((S_out.m_dot - m_out)*m_mult)
        eqs.append((S_out.P - S_in.P)*P_mult)
        for i in range(len(S_out.y)):
            eqs.append((S_out.y[i] - y_out[i])*y_mult)

        # Energy Balance
        H_reag = S_fuel.m_dot*S_fuel.h + S_in.m_dot*S_in.h
        H_prod = S_out.m_dot*S_out.h
    
        eqs.append((H_reag - H_prod)*H_mult)

        return eqs

class Turbine:
    def __init__(self, Name, eta, P_out, Inlet, Outlet, TIT=None, T_iso=None, W=None):
        self.Name = Name
        self.eta = eta
        self.P_out = P_out
        self.Inlet = Inlet
        self.Outlet = Outlet
        self.TIT = TIT
        self.T_iso = T_iso
        self.W = W
    
    def flatten_vars(self):
        return [self.T_iso]
    
    def unpack_vars(self, vars, start_idx):
        self.T_iso = vars[start_idx]
        return start_idx + 1
    
    def residuals(self):
        eqs = []
        S_in = self.Inlet
        S_out = self.Outlet

        eqs.append((S_out.m_dot - S_in.m_dot)*m_mult)
        for i in range(len(S_in.y)):
            eqs.append((S_out.y[i] - S_in.y[i])*y_mult)

        s_in = utils.mixture_entropy(S_in.T, S_in.P, S_in.y, S_in.spc)
        s_iso = utils.mixture_entropy(self.T_iso, self.P_out, S_out.y, S_out.spc)
        h_iso = utils.mixture_enthalpy(self.T_iso, self.P_out, S_out.y, S_out.spc)

        eqs.append((S_out.P - self.P_out)*P_mult)
        eqs.append((s_iso - s_in)*s_mult)
        eqs.append(((S_in.h - S_out.h) - (S_in.h - h_iso)*self.eta)*h_mult)

        if self.TIT is not None:
            eqs.append((S_in.T - self.TIT)*T_mult)

        self.W = S_in.m_dot*(S_in.h - S_out.h)

        return eqs

class Splitter:
    def __init__(self, Name, Inlet, Outlet1, Outlet2, mass_Outlet1=None):
        self.Name = Name
        self.Inlet = Inlet
        self.Outlet1 = Outlet1
        self.Outlet2 = Outlet2
        self.mass_Outlet1 = mass_Outlet1

    def flatten_vars(self):
        return []
    
    def unpack_vars(self, vars, start_idx):
        return start_idx
    
    def residuals(self):
        eqs = []
        S_in = self.Inlet
        S_out1 = self.Outlet1
        S_out2 = self.Outlet2

        if self.mass_Outlet1 is not None:
            eqs.append((S_out1.m_dot - self.mass_Outlet1)*m_mult)
        
        eqs.append((S_out2.m_dot - (S_in.m_dot - S_out1.m_dot))*m_mult)
        eqs.append((S_out1.T - S_in.T)*T_mult)
        eqs.append((S_out2.T - S_in.T)*T_mult)
        eqs.append((S_out1.P - S_in.P)*P_mult)
        eqs.append((S_out2.P - S_in.P)*P_mult)
        for i in range(len(S_in.y)):
            eqs.append((S_out1.y[i] - S_in.y[i])*y_mult)
            eqs.append((S_out2.y[i] - S_in.y[i])*y_mult)

        return eqs

class Splitter_CO2_3way:
    def __init__(self, Name, mass_CO2_combustion, Inlet, Outlet1, Outlet2, Outlet3, mass_Outlet3=None):
        self.Name = Name
        self.mass_CO2_combustion = mass_CO2_combustion
        self.Inlet = Inlet
        self.Outlet1 = Outlet1
        self.Outlet2 = Outlet2
        self.Outlet3 = Outlet3
        self.mass_Outlet3 = mass_Outlet3

    def flatten_vars(self):
        return []
    
    def unpack_vars(self, vars, start_idx):
        return start_idx
    
    def residuals(self):
        eqs = []
        S_in = self.Inlet
        S_out1 = self.Outlet1
        S_out2 = self.Outlet2
        S_out3 = self.Outlet3

        MW, x = utils.mass_fraction(S_in.y, S_in.spc)
        m_in_dict = {spc: xi*S_in.m_dot for (spc, xi) in zip(S_in.spc, x)}
        m1_dict = {spc: xi*S_out1.m_dot for (spc, xi) in zip(S_in.spc, x)}
        m2_dict = {spc: xi*S_out2.m_dot for (spc, xi) in zip(S_in.spc, x)}
        m3_dict = {spc: xi*S_out3.m_dot for (spc, xi) in zip(S_in.spc, x)}

        m_CO2_in = m_in_dict[utils.SPS['CO2']]
        m1_CO2_out = m1_dict[utils.SPS['CO2']]
        m2_CO2_out = m2_dict[utils.SPS['CO2']]
        m3_CO2_out = m3_dict[utils.SPS['CO2']]

        eqs.append((m2_CO2_out - self.mass_CO2_combustion)*m_mult)
        eqs.append((S_out3.m_dot - (S_in.m_dot - (S_out1.m_dot + S_out2.m_dot))))

        if self.mass_Outlet3 is not None:
            eqs.append((S_out3.m_dot - self.mass_Outlet3)*m_mult)
        
        eqs.append((S_out1.T - S_in.T)*T_mult)
        eqs.append((S_out2.T - S_in.T)*T_mult)
        eqs.append((S_out3.T - S_in.T)*T_mult)
        eqs.append((S_out1.P - S_in.P)*P_mult)
        eqs.append((S_out2.P - S_in.P)*P_mult)
        eqs.append((S_out3.P - S_in.P)*P_mult)
        for i in range(len(S_in.y)):
            eqs.append((S_out1.y[i] - S_in.y[i])*y_mult)
            eqs.append((S_out2.y[i] - S_in.y[i])*y_mult)
            eqs.append((S_out3.y[i] - S_in.y[i])*y_mult)

        return eqs

class Condensator:
    def __init__(self, Name, T_resf, Inlet, Outlet_CO2, Outlet_H2O, Q=None):
        self.Name = Name
        self.T_resf = T_resf
        self.Inlet = Inlet
        self.Outlet_CO2 = Outlet_CO2
        self.Outlet_H2O = Outlet_H2O
        self.Q = Q

    def flatten_vars(self):
        return []
    
    def unpack_vars(self, vars, start_idx):
        return start_idx
    
    def residuals(self):
        eqs = []
        S_in = self.Inlet
        S_CO2_out = self.Outlet_CO2
        S_H2O_out = self.Outlet_H2O

        y_H2O_sat = utils.y_H2O_sat(S_CO2_out.T, S_CO2_out.P)
        (MW_in, x_in) = utils.mass_fraction(S_in.y, S_in.spc)

        n_in = S_in.m_dot/MW_in
        n_in_dict = {spc: yi*n_in for (spc, yi) in zip(S_in.spc, S_in.y)}

        H2O = utils.SPS['H2O']
        O2 = utils.SPS['O2']
        n_non_cond = 0
        for spc in n_in_dict:
            if spc is not H2O:
                n_non_cond += n_in_dict[spc]

        n_H2O_out = y_H2O_sat*n_non_cond/(1 - y_H2O_sat)
        n_out = n_non_cond + n_H2O_out

        y_out = []
        for spc in S_CO2_out.spc:
            if spc is H2O:
                y_out.append(n_H2O_out/n_out)
            else:
                y_out.append(n_in_dict[spc]/n_out)   

        n_H2O_cond = n_in_dict[H2O] - n_H2O_out
        m_cond = n_H2O_cond*H2O.MW
        m_out = S_in.m_dot - m_cond

        eqs.append((S_CO2_out.m_dot - m_out)*m_mult)
        eqs.append((S_H2O_out.m_dot - m_cond)*m_mult)
        eqs.append((S_CO2_out.T - self.T_resf)*T_mult)
        eqs.append((S_H2O_out.T - self.T_resf)*T_mult)
        eqs.append((S_CO2_out.P - S_in.P)*P_mult)
        eqs.append((S_H2O_out.P - S_in.P)*P_mult)

        for i in range(len(S_CO2_out.y)):
            eqs.append((S_CO2_out.y[i] - y_out[i])*y_mult)

        eqs.append((S_H2O_out.y[0] - 1)*y_mult)

        self.Q = S_CO2_out.m_dot*S_CO2_out.h + S_H2O_out.m_dot*S_H2O_out.h - S_in.m_dot*S_in.h

        return eqs
    
class CoolantSplitter:
    def __init__(self, Name, K_cool, T_blade, 
                 Inlet, Outlet_HP, Outlet_IP, HP_turb_inlet, IP_turb_inlet):
        self.Name = Name
        self.K_cool = K_cool
        self.T_blade = T_blade
        self.Inlet = Inlet
        self.Outlet_HP = Outlet_HP
        self.Outlet_IP = Outlet_IP
        self.HP_turb_inlet = HP_turb_inlet
        self.IP_turb_inlet = IP_turb_inlet

    def flatten_vars(self):
        return []

    def unpack_vars(self, vars, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        S_in    = self.Inlet
        S_HP    = self.Outlet_HP
        S_IP    = self.Outlet_IP
        S_hot_HP = self.HP_turb_inlet
        S_hot_IP = self.IP_turb_inlet

        T_cool = S_in.T
        m_HP = self.K_cool*S_hot_HP.m_dot*(S_hot_HP.T - self.T_blade)/(self.T_blade - T_cool)
        m_IP = self.K_cool*S_hot_IP.m_dot*(S_hot_IP.T - self.T_blade)/(self.T_blade - T_cool)

        eqs.append((S_in.m_dot - (m_HP + m_IP))*m_mult)
        eqs.append((S_HP.m_dot - m_HP)*m_mult)
        eqs.append((S_IP.m_dot - m_IP)*m_mult)
        eqs.append((S_HP.T - S_in.T)*T_mult)
        eqs.append((S_IP.T - S_in.T)*T_mult)
        eqs.append((S_HP.P - S_in.P)*P_mult)
        eqs.append((S_IP.P - S_in.P)*P_mult)
        for i in range(len(S_in.y)):
            eqs.append((S_HP.y[i] - S_in.y[i])*y_mult)
            eqs.append((S_IP.y[i] - S_in.y[i])*y_mult)

        return eqs

class MHX_ASU:
    def __init__(self, Name, Perc_Delta_P, T_diff_O2, T_diff_N2, Inlet_O2, Inlet_N2, Inlet_Air,
                 Outlet_O2, Outlet_N2, Outlet_Air, Q=None):
        self.Name = Name
        self.Perc_Delta_P = Perc_Delta_P
        self.T_diff_O2 = T_diff_O2
        self.T_diff_N2 = T_diff_N2
        self.Inlet_O2 = Inlet_O2
        self.Inlet_N2 = Inlet_N2
        self.Inlet_Air = Inlet_Air
        self.Outlet_O2 = Outlet_O2
        self.Outlet_N2 = Outlet_N2
        self.Outlet_Air = Outlet_Air
        self.Q = Q

    def flatten_vars(self):
        return []

    def unpack_vars(self, vars, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        S_O2_in = self.Inlet_O2
        S_N2_in = self.Inlet_N2
        S_H_in  = self.Inlet_Air
        S_O2_out = self.Outlet_O2
        S_N2_out = self.Outlet_N2
        S_H_out  = self.Outlet_Air

        stream_pairs = [
            (S_O2_in, S_O2_out),
            (S_N2_in, S_N2_out),
            (S_H_in,  S_H_out),
        ]

        for (S_in, S_out) in stream_pairs:
            eqs.append((S_out.m_dot - S_in.m_dot)*m_mult)
            eqs.append((S_out.P - S_in.P*(1 - self.Perc_Delta_P/100))*P_mult)
            for i in range(len(S_in.y)):
                eqs.append((S_out.y[i] - S_in.y[i])*y_mult)

        # Pinch temp for O2
        eqs.append((S_O2_out.T - (S_H_in.T - self.T_diff_O2))*T_mult)

        # Pinch temp for N2
        eqs.append((S_N2_out.T - (S_H_in.T - self.T_diff_N2))*T_mult)

        # Energy balance — determines S_air_out.T
        Q_balance = 0
        for (S_in, S_out) in stream_pairs:
            Q_balance += S_in.m_dot*(S_out.h - S_in.h)
        eqs.append(Q_balance*H_mult)

        self.Q = S_H_in.m_dot*(S_H_in.h - S_H_out.h)

        return eqs

class ColdBox_ASU:
    def __init__(self, Name, Air_Inlet, O2_Outlet, N2_Outlet,
                 purity_O2, y_O2_waste, T_O2_out, P_O2_out, P_N2_out, T_N2_out, eta):
        self.Name = Name
        self.Air_Inlet = Air_Inlet
        self.O2_Outlet = O2_Outlet
        self.N2_Outlet = N2_Outlet
        self.purity_O2 = purity_O2
        self.y_O2_waste = y_O2_waste
        self.T_O2_out = T_O2_out
        self.P_O2_out = P_O2_out
        self.P_N2_out = P_N2_out
        self.T_N2_out = T_N2_out
        self.eta = eta
        self.W = 0.0
        self.Q = 0.0

    def flatten_vars(self):
        return []

    def unpack_vars(self, vars, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        S_air = self.Air_Inlet
        S_O2  = self.O2_Outlet
        S_N2  = self.N2_Outlet

        p = self.purity_O2
        y_w = self.y_O2_waste

        MW_air, _ = utils.mass_fraction(S_air.y, S_air.spc)
        MW_O2_out = p*utils.MW_O2 + (1.0 - p)*utils.MW_N2
        MW_N2_out = y_w*utils.MW_O2 + (1.0 - y_w)*utils.MW_N2

        n_air    = S_air.m_dot / MW_air
        y_O2_air = S_air.y[0]
        n_O2_out = n_air*(y_O2_air - y_w)/(p - y_w)
        n_N2_out = n_air - n_O2_out

        m_O2_out = n_O2_out*MW_O2_out
        m_N2_out = n_N2_out*MW_N2_out

        # O2 outlet
        eqs.append((S_O2.m_dot - m_O2_out)*m_mult)
        eqs.append((S_O2.y[0] - p)*y_mult)
        eqs.append((S_O2.y[1] - (1.0 - p))*y_mult)
        eqs.append((S_O2.T - self.T_O2_out)*T_mult)
        eqs.append((S_O2.P - self.P_O2_out)*P_mult)

        # N2 outlet
        eqs.append((S_N2.m_dot - m_N2_out)*m_mult)
        eqs.append((S_N2.y[0] - y_w)*y_mult)
        eqs.append((S_N2.y[1] - (1.0 - y_w))*y_mult)
        eqs.append((S_N2.T - self.T_N2_out)*T_mult)
        eqs.append((S_N2.P - self.P_N2_out)*P_mult)

        _, x_O2 = utils.mass_fraction(S_O2.y, S_O2.spc)
        m_dot_O2_pure = S_O2.m_dot*x_O2[0]

        s_air = utils.mixture_entropy(S_air.T, S_air.P, S_air.y, S_air.spc)
        s_O2  = utils.mixture_entropy(S_O2.T,  S_O2.P,  S_O2.y,  S_O2.spc)
        s_N2  = utils.mixture_entropy(S_N2.T,  S_N2.P,  S_N2.y,  S_N2.spc)

        B_air = utils.flow_exergy(S_air.m_dot, S_air.h, s_air)
        B_O2  = utils.flow_exergy(S_O2.m_dot,  S_O2.h,  s_O2)
        B_N2  = utils.flow_exergy(S_N2.m_dot,  S_N2.h,  s_N2)

        H_air = S_air.m_dot*S_air.h
        H_O2 = S_O2.m_dot*S_O2.h
        H_N2 = S_N2.m_dot*S_N2.h
        deltaH = H_air - H_O2 - H_N2

        W_exergy = B_air - B_O2 - B_N2
        self.W = W_exergy * (1 - 1/self.eta)
        self.Q = self.W - deltaH

        return eqs

class GrayBox_ASU:
    def __init__(self, Name, Air_Inlet, O2_Outlet, N2_Outlet,
                 purity_O2, y_O2_waste, eta_sep, T_O2_out, P_O2_out,
                 P_N2_out=1.05e5, T_N2_out=295.0):
        self.Name          = Name
        self.Air_Inlet     = Air_Inlet
        self.O2_Outlet     = O2_Outlet
        self.N2_Outlet     = N2_Outlet
        self.purity_O2     = purity_O2     # O2 mole fraction in O2 product
        self.y_O2_waste    = y_O2_waste    # O2 mole fraction in N2 waste stream
        self.eta_sep       = eta_sep       # MAC + cold box 2nd-law eff. (Gibbs demixing ref.)
        self.T_O2_out      = T_O2_out      # O2 delivery temperature (K)
        self.P_O2_out      = P_O2_out      # O2 delivery pressure (Pa)
        self.P_N2_out      = P_N2_out      # N2 exhaust pressure (Pa)
        self.T_N2_out      = T_N2_out      # N2 exhaust temperature (K)
        self.W            = 0.0          # total ASU work (W), set in residuals()
        self.Q            = 0.0          # heat rejected to environment (W), set in residuals()

    def flatten_vars(self):
        return []

    def unpack_vars(self, vars, start_idx):
        return start_idx

    def residuals(self):
        eqs = []
        S_air = self.Air_Inlet
        S_O2 = self.O2_Outlet
        S_N2 = self.N2_Outlet

        p = self.purity_O2
        y_w = self.y_O2_waste

        # Molar masses of outlet streams
        MW_air, _ = utils.mass_fraction(S_air.y, S_air.spc)
        MW_O2_out = p*utils.MW_O2 + (1.0 - p)*utils.MW_N2
        MW_N2_out = y_w*utils.MW_O2 + (1.0 - y_w)*utils.MW_N2

        # Molar flow rates (mol/s) from molar balance + O2 species balance
        #   n_air = n_O2_out + n_N2_out
        #   n_air * y_O2_air = n_O2_out * p + n_N2_out * y_w
        n_air    = S_air.m_dot/MW_air
        y_O2_air = S_air.y[0]
        n_O2_out = n_air*(y_O2_air - y_w)/(p - y_w)
        n_N2_out = n_air - n_O2_out

        m_O2_out = n_O2_out*MW_O2_out
        m_N2_out = n_N2_out*MW_N2_out

        # --- O2 outlet ---
        eqs.append((S_O2.m_dot - m_O2_out)*m_mult) # mass flow
        eqs.append((S_O2.y[0] - p)*y_mult) # O2 purity
        eqs.append((S_O2.y[1] - (1.0 - p))*y_mult) # y sums to 1
        eqs.append((S_O2.T - self.T_O2_out)*T_mult) # delivery T
        eqs.append((S_O2.P - self.P_O2_out)*P_mult) # delivery P

        # --- N2 outlet ---
        eqs.append((S_N2.m_dot - m_N2_out)*m_mult) # mass flow
        eqs.append((S_N2.y[0] - y_w)*y_mult) # O2 in waste
        eqs.append((S_N2.y[1] - (1.0 - y_w))*y_mult) # y sums to 1
        eqs.append((S_N2.T - self.T_N2_out)*T_mult) # exhaust T
        eqs.append((S_N2.P - self.P_N2_out)*P_mult) # exhaust P

        # --- Separation work: Gibbs minimum (outlet exergy - inlet exergy) / eta_sep ---
        s_air = utils.mixture_entropy(S_air.T, S_air.P, S_air.y, S_air.spc)
        s_O2  = utils.mixture_entropy(S_O2.T,  S_O2.P,  S_O2.y,  S_O2.spc)
        s_N2  = utils.mixture_entropy(S_N2.T,  S_N2.P,  S_N2.y,  S_N2.spc)

        B_air = utils.flow_exergy(S_air.m_dot, S_air.h, s_air)
        B_O2  = utils.flow_exergy(S_O2.m_dot,  S_O2.h,  s_O2)
        B_N2  = utils.flow_exergy(S_N2.m_dot,  S_N2.h,  s_N2)

        W_min_Gibbs = B_air - (B_O2 + B_N2)
        self.W = W_min_Gibbs/self.eta_sep

        # First law for ASU: Q_env = (H_O2 + H_N2 - H_air) + W  (negative = rejected to env)
        self.Q = S_O2.m_dot*S_O2.h + S_N2.m_dot*S_N2.h - S_air.m_dot*S_air.h + self.W

        return eqs