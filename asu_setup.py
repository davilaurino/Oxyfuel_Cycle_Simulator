import utils
from classes import MHX_ASU, State, Input, ColdBox_ASU, Compressor, Intercooler, Pump

T_atm = 298.15
P_atm = 101325
y_atm = [0.21, 0.79]

T_N2_cold = 90  # K (N2 cold temperature after cold box)
T_N2_out = 295     # K  (N2 exhaust temperature)
P_N2_out  = 1.35e5  # Pa (N2 exhaust pressure)

T_lox = 85      # K  (LOX temperature)
P_lox = 1.25e5   # Pa (LOX pressure — slightly above atmospheric for pumping)

T_O2_out  = 295     # K  (O2 delivery temperature)
P_O2_out  = 8e6     # Pa (O2 delivery pressure — internal compression inside cold box)

P_MAC        = 6e5   # Pa (MAC outlet pressure)
eta_air_comp = 0.88
n_air_comp   = 3
pr_air_comp  = (P_MAC/P_atm)**(1/n_air_comp)

eta_lox_pump = 0.88

T_resf = 300   # K
PERC_DELTAP_IC = 0.2   # %
PERC_DELTAP_MHX = 1.0  # %

T_pinch_O2 = 2.0  # K, pinch for O2 in MHX
T_pinch_N2 = 5.0  # K, pinch for N2 in MHX

asu_multiplier = 1

def setup_asu(O2_purity, y_O2_waste, eta_coldbox=None):

    if eta_coldbox is None:
        eta_coldbox = utils.asu_eta_poor(O2_purity, asu_multiplier)

    S_asu = []
    C_asu = {}

    # Air Compressor 1 - Inlet (atmospheric air)
    S0_ASU = State('Air Comp 1 - Inlet', 'Oxygen + Nitrogen', spc=utils.AIR_SPECIES)
    S0_ASU.m_dot = 100
    S0_ASU.T = T_atm
    S0_ASU.P = P_atm
    S0_ASU.y = list(y_atm)
    S_asu.append(S0_ASU)

    Air_Input = Input('Air Inlet', T_atm, P_atm, y_atm, S0_ASU)
    C_asu[Air_Input.Name] = Air_Input

    P1 = P_atm * pr_air_comp
    P2 = P_atm * pr_air_comp**2

    # Air Intercooler 1 - Inlet
    S1_ASU = State('Air Ic 1 - Inlet', 'Oxygen + Nitrogen', spc=utils.AIR_SPECIES)
    S1_ASU.m_dot = 100
    S1_ASU.T = 350.0
    S1_ASU.P = P1
    S1_ASU.y = list(y_atm)
    S_asu.append(S1_ASU)

    Air_comp1 = Compressor('Air Compressor 1', eta_air_comp, pr_air_comp, S0_ASU, S1_ASU)
    Air_comp1.T_iso = 350.0
    C_asu[Air_comp1.Name] = Air_comp1

    # Air Compressor 2 - Inlet
    S2_ASU = State('Air Comp 2 - Inlet', 'Oxygen + Nitrogen', spc=utils.AIR_SPECIES)
    S2_ASU.m_dot = 100
    S2_ASU.T = T_resf
    S2_ASU.P = P1
    S2_ASU.y = list(y_atm)
    S_asu.append(S2_ASU)

    Air_Ic1 = Intercooler('Air Intercooler 1', T_resf, PERC_DELTAP_IC, S1_ASU, S2_ASU)
    C_asu[Air_Ic1.Name] = Air_Ic1

    # Air Intercooler 2 - Inlet
    S3_ASU = State('Air Ic 2 - Inlet', 'Oxygen + Nitrogen', spc=utils.AIR_SPECIES)
    S3_ASU.m_dot = 100
    S3_ASU.T = 350.0
    S3_ASU.P = P2
    S3_ASU.y = list(y_atm)
    S_asu.append(S3_ASU)

    Air_comp2 = Compressor('Air Compressor 2', eta_air_comp, pr_air_comp, S2_ASU, S3_ASU)
    Air_comp2.T_iso = 350.0
    C_asu[Air_comp2.Name] = Air_comp2

    # Air Compressor 3 - Inlet
    S4_ASU = State('Air Comp 3 - Inlet', 'Oxygen + Nitrogen', spc=utils.AIR_SPECIES)
    S4_ASU.m_dot = 100
    S4_ASU.T = T_resf
    S4_ASU.P = P2
    S4_ASU.y = list(y_atm)
    S_asu.append(S4_ASU)

    Air_Ic2 = Intercooler('Air Intercooler 2', T_resf, PERC_DELTAP_IC, S3_ASU, S4_ASU)
    C_asu[Air_Ic2.Name] = Air_Ic2

    # Air Intercooler 3 - Inlet
    S5_ASU = State('Air Ic 3 - Inlet', 'Oxygen + Nitrogen', spc=utils.AIR_SPECIES)
    S5_ASU.m_dot = 100
    S5_ASU.T = 350.0
    S5_ASU.P = P_MAC
    S5_ASU.y = list(y_atm)
    S_asu.append(S5_ASU)

    Air_comp3 = Compressor('Air Compressor 3', eta_air_comp, pr_air_comp, S4_ASU, S5_ASU)
    Air_comp3.T_iso = 350.0
    C_asu[Air_comp3.Name] = Air_comp3

    # Compressed Air
    S6_ASU = State('Compressed Air', 'Oxygen + Nitrogen', spc=utils.AIR_SPECIES)
    S6_ASU.m_dot = 100
    S6_ASU.T = 300.0
    S6_ASU.P = P_MAC
    S6_ASU.y = list(y_atm)
    S_asu.append(S6_ASU)

    Air_Ic3 = Intercooler('Air Intercooler 3', T_resf, PERC_DELTAP_IC, S5_ASU, S6_ASU)
    C_asu[Air_Ic3.Name] = Air_Ic3

    # Cold Air
    S7_ASU = State('Cold Air', 'Oxygen + Nitrogen', spc=utils.AIR_SPECIES)
    S7_ASU.m_dot = 100
    S7_ASU.T = 120
    S7_ASU.P = P_MAC
    S7_ASU.y = list(y_atm)
    S_asu.append(S7_ASU)

    # O2 Cold
    S8_ASU = State('ASU O2 Cold', 'Oxygen', spc=utils.AIR_SPECIES)
    S8_ASU.m_dot = 28.53
    S8_ASU.T = T_O2_out
    S8_ASU.P = P_O2_out
    S8_ASU.y = [0.99, 0.01]
    S_asu.append(S8_ASU)

    # N2 Cold
    S9_ASU = State('ASU N2 Cold', 'Nitrogen', spc=utils.AIR_SPECIES)
    S9_ASU.m_dot = 79.0
    S9_ASU.T = T_N2_out
    S9_ASU.P = P_N2_out
    S9_ASU.y = [0.99, 0.01]
    S_asu.append(S9_ASU)

    # Pessurized cold O2
    S10_ASU = State('ASU O2 Pressurized', 'Oxygen', spc=utils.AIR_SPECIES)
    S10_ASU.m_dot = 28.53
    S10_ASU.T = T_lox
    S10_ASU.P = P_lox
    S10_ASU.y = [0.99, 0.01]
    S_asu.append(S10_ASU)

    LOX_pump = Pump('LOX Pump', eta_lox_pump, P_O2_out, S8_ASU, S10_ASU)
    LOX_pump.T_iso = 100
    C_asu[LOX_pump.Name] = LOX_pump

    ColdBox = ColdBox_ASU('ASU Cold Box', S7_ASU, S8_ASU, S9_ASU,
                           O2_purity, y_O2_waste, T_lox, P_lox, P_N2_out, T_N2_cold, eta_coldbox)
    C_asu[ColdBox.Name] = ColdBox

    # O2 Delivery
    S11_ASU = State('O2 Delivery', 'Oxygen', spc=utils.AIR_SPECIES)
    S11_ASU.m_dot = 28.53
    S11_ASU.T = T_O2_out
    S11_ASU.P = P_O2_out
    S11_ASU.y = [0.99, 0.01]

    # N2 Vent
    S12_ASU = State('N2 Vent', 'Nitrogen', spc=utils.AIR_SPECIES)
    S12_ASU.m_dot = 79.0
    S12_ASU.T = T_N2_out
    S12_ASU.P = P_N2_out
    S12_ASU.y = [0.99, 0.01]
    S_asu.append(S12_ASU)

    MHX = MHX_ASU('ASU MHX', PERC_DELTAP_MHX, T_pinch_O2, T_pinch_N2,
                  S10_ASU, S9_ASU, S6_ASU, S11_ASU, S12_ASU, S7_ASU)
    C_asu[MHX.Name] = MHX

    return {
        'O2_out':     S11_ASU,
        'states':     S_asu,
        'components': C_asu,
    }