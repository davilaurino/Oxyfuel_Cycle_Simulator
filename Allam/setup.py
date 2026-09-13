import utils as utils
from classes import State, Input, Compressor, Pump, Intercooler, Combustor, Mixer, Turbine
from classes import Splitter, HX_mult, Condensator, Splitter_CO2_3way, CoolantSplitter

P_in  = 3e6  # Pa
P_out = 8e6  # Pa
P_max = 4e7  # Pa (max cycle pressure)

eta_pump = 0.88

eta_compressor = 0.87
pr_comp_tot    = P_out/P_in
n_comp         = 4
pr_comp        = pr_comp_tot**(1/n_comp)

eta_turb    = 0.92
rp_turb_tot = P_max/P_in
n_turb      = 3
rp_turb     = rp_turb_tot**(1/n_turb)
T_resf      = 300  # K

P_out_turb1 = P_max/rp_turb
P_out_turb2 = P_max/(rp_turb**2)
P_out_turb3 = P_in

PERC_DELTA_P_Recup = 0.5  # %
Perc_Delta_P_IC    = 0.1  # %

T_pinch_extra = 5
T_flue_out = 400  # K — flue gas recuperator exit temperature (above H2O dew point)

T_fuel   = 298.15  # K
P_fuel   = 7e5  # Pa (7 bar, per Rogalev 2021)
y_ng     = [0.94, 0.03, 0.02, 0.01]
MW_fuel, x_fuel = utils.mass_fraction(y_ng, utils.FUEL_SPECIES)

eta_comp_fuel = 0.85
pr_comp_fuel  = P_max/P_fuel

LHV_ng = sum(xi * utils.LHV.get(spc.name, 0) for spc, xi in zip(utils.FUEL_SPECIES, x_fuel))

K_cool  = 0.06  # El-Masri cooling coefficient
T_blade = 1120  # K — blade metal temperature limit

T_cool = 473  # K — coolant temperature

def build_cycle(exc_O2=1.01, r_CO2_O2=10, TIT=1500, O2_purity=0.99,
                m_dot_fuel=7.4, T_O2=295, P_O2=P_max):
    
    S = []
    Component = {}

    # State 0 (Fuel Inlet)
    S0 = State('Fuel Compressor 1 - Inlet', 'Natural Gas', spc=utils.FUEL_SPECIES)
    S0.m_dot = m_dot_fuel
    S0.T = T_fuel
    S0.P = P_fuel
    S0.y = list(y_ng)
    S.append(S0)

    _, _, m_O2_req = utils.fuel_requirements(S0)
    m_CO2_comb = m_O2_req * r_CO2_O2

    Fuel_in = Input('Fuel Inlet', T_fuel, P_fuel, y_ng, S[0])
    Fuel_in.m_dot = m_dot_fuel
    Component[Fuel_in.Name] = Fuel_in

    # State 1 (Compressed Fuel - Combustor Inlet)
    S1 = State('Compressed Fuel - Combustor Inlet', 'Natural Gas', spc=utils.FUEL_SPECIES)
    S1.m_dot = m_dot_fuel
    S1.T = 500
    S1.P = P_max
    S1.y = list(y_ng)
    S.append(S1)

    Fuel_comp = Compressor('Fuel Comp1', eta_comp_fuel, pr_comp_fuel, S[0], S[1])
    Fuel_comp.T_iso = 618  # K
    Component[Fuel_comp.Name] = Fuel_comp

    # State 2 (Direct oxygen input)
    # Its flow rate remains a solver variable fixed by the combustor O2 demand.
    y_O2 = [O2_purity, 1 - O2_purity]
    S2 = State('O2 Input', 'Oxygen', spc=utils.AIR_SPECIES)
    S2.m_dot = 28.8  # kg/s, initial guess only
    S2.T = T_O2
    S2.P = P_O2
    S2.y = list(y_O2)
    S.append(S2)

    O2_in = Input('O2 Input', T_O2, P_O2, y_O2, S[2])
    Component[O2_in.Name] = O2_in

    # State 3 (Compressor 1 Inlet)
    S3 = State('Compressor 1 - Inlet', 'CarbonDioxide')
    S3.m_dot = 580.85  # kg/s
    S3.T = 299.15  # K
    S3.P = 2955150.0  # Pa
    S3.y = [0.9989, 0.0011, 0, 0]
    S.append(S3)

    # State 4 (Intercooler 1 Inlet)
    S4 = State('Intercooler 1 - Inlet', 'CarbonDioxide')
    S4.m_dot = 580.85  # kg/s
    S4.T = 319.28  # K
    S4.P = S3.P*pr_comp  # Pa
    S4.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S4)

    Comp1 = Compressor('Compressor 1', eta_compressor, pr_comp, S[3], S[4])
    Comp1.T_iso = 318  # K
    Component[Comp1.Name] = Comp1

    # State 5 (Compressor 2 Inlet)
    S5 = State('Compressor 2 - Inlet', 'CarbonDioxide')
    S5.m_dot = 580.85  # kg/s
    S5.T = 299.15  # K
    S5.P = S4.P*(1 - Perc_Delta_P_IC/100)  # Pa
    S5.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S5)

    Ic1 = Intercooler('Intercooler 1', T_resf, Perc_Delta_P_IC, S[4], S[5])
    Component[Ic1.Name] = Ic1

    # State 6 (Intercooler 2 Inlet)
    S6 = State('Intercooler 2 - Inlet', 'CarbonDioxide')
    S6.m_dot = 580.85  # kg/s
    S6.T = 319.28  # K
    S6.P = S5.P*pr_comp  # Pa
    S6.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S6)

    Comp2 = Compressor('Compressor 2', eta_compressor, pr_comp, S[5], S[6])
    Comp2.T_iso = 318  # K
    Component[Comp2.Name] = Comp2

    # State 7 (Compressor 3 Inlet)
    S7 = State('Compressor 3 - Inlet', 'CarbonDioxide')
    S7.m_dot = 580.85  # kg/s
    S7.T = 299.15  # K
    S7.P = S6.P*(1 - Perc_Delta_P_IC/100)  # Pa
    S7.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S7)

    Ic2 = Intercooler('Intercooler 2', T_resf, Perc_Delta_P_IC, S[6], S[7])
    Component[Ic2.Name] = Ic2

    # State 8 (Intercooler 3 Inlet)
    S8 = State('Intercooler 3 - Inlet', 'CarbonDioxide')
    S8.m_dot = 580.85  # kg/s
    S8.T = 319.02  # K
    S8.P = S7.P*pr_comp  # Pa
    S8.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S8)

    Comp3 = Compressor('Compressor 3', eta_compressor, pr_comp, S[7], S[8])
    Comp3.T_iso = 318  # K
    Component[Comp3.Name] = Comp3

    # State 9 (Compressor 4 Inlet)
    S9 = State('Compressor 4 - Inlet', 'CarbonDioxide')
    S9.m_dot = 580.85  # kg/s
    S9.T = 299.15  # K
    S9.P = S8.P*(1 - Perc_Delta_P_IC/100)  # Pa
    S9.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S9)

    Ic3 = Intercooler('Intercooler 3', T_resf, Perc_Delta_P_IC, S[8], S[9])
    Component[Ic3.Name] = Ic3

    # State 10 (Intercooler 4 Inlet)
    S10 = State('Intercooler 4 - Inlet', 'CarbonDioxide')
    S10.m_dot = 580.85  # kg/s
    S10.T = 317.48  # K
    S10.P = S9.P*pr_comp  # Pa
    S10.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S10)

    Comp4 = Compressor('Compressor 4', eta_compressor, pr_comp, S[9], S[10])
    Comp4.T_iso = 316  # K
    Component[Comp4.Name] = Comp4

    # State 11 (Compressed RSCCO2)
    S11 = State('Compressed RSCCO2', 'CarbonDioxide')
    S11.m_dot = 580.85  # kg/s
    S11.T = 299.15  # K
    S11.P = S10.P*(1 - Perc_Delta_P_IC/100)  # Pa
    S11.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S11)

    Ic4 = Intercooler('Intercooler 4', T_resf, Perc_Delta_P_IC, S[10], S[11])
    Component[Ic4.Name] = Ic4

    # State 12 (RSCCO2 - Combustion)
    S12 = State('RSCCO2 - Combustion', 'CarbonDioxide')
    S12.m_dot = 271.04  # kg/s
    S12.T = 299.15  # K
    S12.P = S11.P  # Pa
    S12.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S12)

    # State 13 (RSCCO2 - Max. Pressure - Combustion)
    S13 = State('RSCCO2 - Max. Pressure - Combustion', 'CarbonDioxide')
    S13.m_dot = 271.04  # kg/s
    S13.T = 329  # K
    S13.P = P_max
    S13.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S13)

    Pump1 = Pump('CO2-Combustion Pump', eta_pump, P_max, S[12], S[13])
    Pump1.T_iso = 327  # K
    Component[Pump1.Name] = Pump1

    # State 14 (RSCCO2 + O2 - Max. Pressure - Combustion)
    S14 = State('RSCCO2 + O2 - Max. Pressure - Combustion', 'CarbonDioxide + Oxygen')
    S14.m_dot = 299.58  # kg/s
    S14.T = 326.5  # K
    S14.P = S2.P  # Pa
    S14.y = [0.8726, 0.0010, 0.1264, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S14)

    MixerCO2_O2 = Mixer('CO2-O2 Mixer', S[13], S[2], S[14])
    Component[MixerCO2_O2.Name] = MixerCO2_O2

    # State 15 (RSCCO2 + O2 - Oxidant)
    S15 = State('RSCCO2 + O2 - Oxidant', 'CarbonDioxide + Oxygen')
    S15.m_dot = 299.58  # kg/s
    S15.T = 867.54  # K
    S15.P = S14.P*(1 - PERC_DELTA_P_Recup/100)  # Pa
    S15.y = [0.8726, 0.0010, 0.1264, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S15)

    # State 16 (RSCCO2 - Extra + TC)
    S16 = State('RSCCO2 - Extra + TC', 'CarbonDioxide')
    S16.m_dot = 289.66  # kg/s
    S16.T = 299.15  # K
    S16.P = 7786259.9  # Pa
    S16.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S16)

    # State 17 (RSCCO2 - Max. Pressure - Extra + TC)
    S17 = State('RSCCO2 - Max. Pressure - Extra + TC', 'CarbonDioxide')
    S17.m_dot = 289.66  # kg/s
    S17.T = 329  # K
    S17.P = P_max  # Pa
    S17.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S17)

    Pump2 = Pump('CO2-Extra+TC Pump', eta_pump, P_max, S[16], S[17])
    Pump2.T_iso = 327  # K
    Component[Pump2.Name] = Pump2

    # State 18 (RSCCO2 - Extra)
    S18 = State('RSCCO2 - Extra', 'CarbonDioxide')
    S18.m_dot = 239.66  # kg/s
    S18.T = S17.T  # K
    S18.P = S17.P  # Pa
    S18.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S18)

    # State 19 (RSCCO2 - Extra - Heated)
    S19 = State('RSCCO2 - Extra - Heated', 'CarbonDioxide')
    S19.m_dot = 239.66  # kg/s
    S19.T = 972.54  # K
    S19.P = S18.P*(1 - PERC_DELTA_P_Recup/100)  # Pa
    S19.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S19)

    # State 20 (RSCCO2 - TC)
    S20 = State('RSCCO2 - TC', 'CarbonDioxide')
    S20.m_dot = 50  # kg/s
    S20.T = S17.T  # K
    S20.P = S17.P  # Pa
    S20.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S20)

    Split2 = Splitter('Split2', S[17], S[20], S[18])
    Component['Split2'] = Split2

    # State 21 (RSCCO2 - TC - Heated)
    S21 = State('RSCCO2 - TC - Heated', 'CarbonDioxide')
    S21.m_dot = 50  # kg/s
    S21.T = 517.11  # K
    S21.P = S20.P*(1 - PERC_DELTA_P_Recup/100)  # Pa
    S21.y = [0.9989, 0.0011, 0, 0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S21)

    # State 22 (Combustion Products)
    S22 = State('Combustion Products', 'CarbonDioxide + Water')
    S22.m_dot = 306.98  # kg/s
    S22.T = 1707.5  # K
    S22.P = S15.P  # Pa
    S22.y = [0.8821, 0.1179, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S22)

    Combustor_main = Combustor('Combustor', S[1], S[15], S[22])
    Combustor_main.exc_O2 = exc_O2
    Component[Combustor_main.Name] = Combustor_main

    # State 23 (HP Turbine - Inlet)
    S23 = State('HP Turbine - Inlet', 'CarbonDioxide + Water')
    S23.m_dot = 546.63  # kg/s
    S23.T = 1400.0  # K
    S23.P = S22.P  # Pa
    S23.y = [0.9313, 0.0687, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S23)

    Extra_Mixer = Mixer('Extra CO2 Dilution Mixer', S[19], S[22], S[23])
    Component[Extra_Mixer.Name] = Extra_Mixer

    # State 24 (HP Turbine - Outlet)
    S24 = State('HP Turbine - Outlet', 'CarbonDioxide + Water')
    S24.m_dot = 546.63  # kg/s
    S24.T = 1268.46  # K
    S24.P = 13924766.5  # Pa
    S24.y = [0.9313, 0.0687, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S24)

    Turb1 = Turbine('HP Turbine', eta_turb, P_out_turb1, S[23], S[24])
    Turb1.TIT = TIT
    Turb1.T_iso = 1240  # K
    Component[Turb1.Name] = Turb1

    # State 25 (Coolant flow - HP Turbine)
    S25 = State('Coolant flow - HP Turbine', 'CarbonDioxide + Water')
    S25.m_dot = 25  # kg/s
    S25.T = 517.11  # K
    S25.P = S21.P  # Pa
    S25.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S25)

    # State 26 (IP Turbine - Inlet)
    S26 = State('IP Turbine - Inlet', 'CarbonDioxide + Water')
    S26.m_dot = 571.63  # kg/s
    S26.T = 1236.84  # K
    S26.P = 13924766.5  # Pa
    S26.y = [0.9341, 0.0659, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S26)

    HP_cool_Mixer = Mixer('Mixer - HP Turbine Coolant', S[24], S[25], S[26])
    Component[HP_cool_Mixer.Name] = HP_cool_Mixer

    # State 27 (IP Turbine - Outlet)
    S27 = State('IP Turbine - Outlet', 'CarbonDioxide + Water')
    S27.m_dot = 571.63  # kg/s
    S27.T = 1113.2  # K
    S27.P = 6463304.1  # Pa
    S27.y = [0.9341, 0.0659, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S27)

    Turb2 = Turbine('IP Turbine', eta_turb, P_out_turb2, S[26], S[27])
    Turb2.T_iso = 1150  # K
    Component[Turb2.Name] = Turb2

    # State 28 (Coolant flow - IP Turbine)
    S28 = State('Coolant flow - IP Turbine', 'CarbonDioxide + Water')
    S28.m_dot = 25  # kg/s
    S28.T = 517.11  # K
    S28.P = S21.P  # Pa
    S28.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S28)

    TC_Split = CoolantSplitter('Splitter - Turbine Cooling Flow', K_cool, T_blade, 
                               S[21], S[25], S[28], S[23], S[26])
    Component[TC_Split.Name] = TC_Split

    # State 29 (LP Turbine - Inlet)
    S29 = State('LP Turbine - Inlet', 'CarbonDioxide + Water')
    S29.m_dot = 596.63  # kg/s
    S29.T = 1088.41  # K
    S29.P = 6463304.1  # Pa
    S29.y = [0.9367, 0.0633, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S29)

    IP_cool_Mixer = Mixer('Mixer - IP Turbine Coolant', S[27], S[28], S[29])
    Component[IP_cool_Mixer.Name] = IP_cool_Mixer

    # State 30 (LP Turbine - Outlet)
    S30 = State('LP Turbine - Outlet', 'CarbonDioxide + Water')
    S30.m_dot = 596.63  # kg/s
    S30.T = 977.54  # K
    S30.P = 3000000.0  # Pa
    S30.y = [0.9367, 0.0633, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S30)

    Turb3 = Turbine('LP Turbine', eta_turb, P_out_turb3, S[29], S[30])
    Turb3.T_iso = 1000  # K
    Component[Turb3.Name] = Turb3

    # State 31 (Flue gas - Recuperator Outlet)
    S31 = State('Flue gas 3 - Recuperator Outlet', 'CarbonDioxide + Water')
    S31.m_dot = 596.63  # kg/s
    S31.T = 353.05  # K
    S31.P = S30.P*(1 - PERC_DELTA_P_Recup/100)  # Pa
    S31.y = [0.9367, 0.0633, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S31)

    MSHX = HX_mult('Multi-Stream Heat Exchanger', PERC_DELTA_P_Recup, T_pinch_extra, T_cool, T_flue_out,
                   S[18], S[14], S[20], S[30], S[19], S[15], S[21], S[31])
    Component['MSHX'] = MSHX

    # State 32 (H2O Out)
    S32 = State('H2O Out', 'Water', spc=[utils.SPS['H2O']])
    S32.m_dot = 15.78  # kg/s
    S32.T = 299.15  # K
    S32.P = S31.P  # Pa
    S32.y = [1.0]
    S.append(S32)

    Cond = Condensator('Condensator', T_resf, S[31], S[3], S[32])
    Component[Cond.Name] = Cond

    # State 33 (CO2 Out)
    S33 = State('CO2 Out', 'CarbonDioxide')
    S33.m_dot = 24.49  # kg/s
    S33.T = 299.15  # K
    S33.P = S11.P  # Pa
    S33.y = [0.9989, 0.0011, 0.0, 0.0]  # y_CO2, y_H2O, y_O2, y_N2
    S.append(S33)

    Split_CO2 = Splitter_CO2_3way('CO2 3-way', m_CO2_comb, S[11], S[33], S[12], S[16])
    Component[Split_CO2.Name] = Split_CO2

    X0 = []
    for state in S:
        X0 += state.flatten_vars()
    for comp_name in Component:
        X0 += Component[comp_name].flatten_vars()

    return S, Component, X0, LHV_ng
