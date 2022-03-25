import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
import random

# ----------------------------------------------------------------------------------------------------------------

def matS_h(datstot_h):
    # Entropy Heating
    HintStot_h = datstot_h[0, 1:]
    TempStot_h = datstot_h[1:, 0]
    mS_h = RectBivariateSpline(TempStot_h, HintStot_h, datstot_h[1:, 1:], kx=1, ky=1)
    return mS_h

def matCp_h(cpdat_h):
    # Heat capacity heating
    HintCp_h = cpdat_h[0, 1:]
    TempCp_h = cpdat_h[1:, 0]
    mCp_h = RectBivariateSpline(TempCp_h, HintCp_h, cpdat_h[1:, 1:], ky=1, kx=1)
    return mCp_h


# ---------------- Plotting material properties -------------------

''' This section plots material properties of a layered AMR bed. It is also possible to plot thermodynamic cycles of 
particular nodes in the bed. To do so, uncomment the proper section below and select the output file of the case to plot
above.'''

# Inputs
id_first_mat = 98
id_last_mat = 107
materials = []
mag_field_to_plot = 1.4
for i in range(id_first_mat, id_last_mat + 1):
    materials.append('M' + str(i))
# materials = ['M6', 'M7', 'M8',  'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15']
# materials = ['M0', 'M2']
temperature_range = [275, 320]  # [K] Used for plots
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'olive', 'orangered', 'indigo', 'crimson', 'grey', 'royalblue', 'khaki',
          'darkorange', 'slategray', 'deeppink', 'teal', 'peru', 'darkviolet']

# colors = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])  # This works inside the loop

cp_legends = []

# Calculations
index = 0
for mat in materials:
    # print(mat)
    plt.figure(1)  # Heat capacity vs Temperature
    cp = np.loadtxt(mat+'/'+mat+'_cp_h.txt')
    cp_c = np.loadtxt(mat+'/'+mat+'_cp_c.txt')
    T_cp = cp[1:, 0]
    C_p_0T = cp[1:, 1]
    i_mag_field = list(cp[0, :]).index(mag_field_to_plot)
    C_p_09T = cp[1:, i_mag_field]
    Cp_hi_field = cp[1:, 15]
    plt.plot(T_cp, C_p_0T, color=colors[index], linestyle='dotted')  # Heating zero field
    plt.plot(T_cp, C_p_09T, color=colors[index], linestyle='solid')  # Heating high field

    plt.plot(T_cp, cp_c[1:, 1], color=colors[index], linestyle='dotted')  # Cooling zero field
    plt.plot(T_cp, cp_c[1:, i_mag_field], color=colors[index], linestyle='solid')  # Cooling high field

    cp_legends.append('{} - {} T'.format(mat, mag_field_to_plot))
    # plt.plot(T_cp, Cp_hi_field, color=colors[index], linestyle='dashed')

    plt.figure(2)  # Magnetic entropy change vs Temperature
    S = np.loadtxt(mat+'/'+mat+'_S_h.txt')
    T_s = S[1:, 0]
    dS_m = S[1:, 1] - S[1:, i_mag_field]  # Negative magnetic entropy change from 0 T to 0.9 T
    dS_m_hi_field = S[1:, 1] - S[1:, 15]
    plt.plot(T_s, dS_m, color=colors[index], linestyle='solid')
    plt.plot(T_s, dS_m_hi_field, color=colors[index], linestyle='dashed')

    plt.figure(3)  # Adiabatic temperature change vs Temperature
    s_0 = interp1d(S[1:, 0], S[1:, 1], kind='cubic')
    T_1 = interp1d(S[1:, i_mag_field], S[1:, 0], kind='cubic')
    T_hi_field = interp1d(S[1:, 15], S[1:, 0], kind='cubic')
    dT_ad = []
    dT_hi_field = []
    # T_set = np.linspace(temperature_range[0]+index, temperature_range[1]-20+index, 300)  # TODO: generalize these limits
    T_set = np.linspace(S[1, 0], S[-30, 0], 300)
    for Temperature in T_set:
        dT_ad.append(T_1(s_0(Temperature))-Temperature)
        dT_hi_field.append(T_hi_field(s_0(Temperature))-Temperature)
    plt.plot(T_set, dT_ad, color=colors[index], linestyle='solid')
    # plt.plot(T_set, dT_hi_field, color=colors[index], linestyle='dashed')

    plt.figure(4)  # ST diagram
    plt.plot(T_s, S[1:, 1], color=colors[index])
    plt.plot(T_s, S[1:, i_mag_field], color=colors[index])

    plt.figure(5)  # MT curves
    Mag = np.loadtxt(mat+'/'+mat+'_Mag_h.txt')
    T_mag = Mag[1:, 0]

    # Mag_0_1T = Mag[1:, 2]
    i_mag_field = list(Mag[0, :]).index(mag_field_to_plot)
    Mag_hi_field = Mag[1:, i_mag_field]
    # plt.plot(T_mag, Mag_0_1T, color=colors[index])
    plt.plot(T_mag, Mag_hi_field, color=colors[index], linestyle='solid')
    plt.plot(T_mag, Mag[1:, 15], color=colors[index], linestyle='dashed')

    plt.figure(6)  # dMdT
    i = 0
    dMdT_lo_field = []
    dMdT_hi_field = []
    for temp in Mag[2:-1, 0]:
        dMdT_lo_field.append((Mag[3+i, 10]-Mag[1+i, 10]) / (Mag[3+i, 0] - Mag[1+i, 0]))
        dMdT_hi_field.append((Mag[3+i, 15]-Mag[1+i, 15]) / (Mag[3+i, 0] - Mag[1+i, 0]))
        i = i+1
    plt.plot(Mag[2:-1, 0], dMdT_lo_field, color=colors[index], linestyle='solid')
    plt.plot(Mag[2:-1, 0], dMdT_hi_field, color=colors[index], linestyle='dashed')

    # Plotting thermodynamic cycles experienced by the center node of each layer on a ST diagram

    # plt.figure(4)
    # s_if = matS_h(S)
    # entropy = np.zeros((time_steps + 1, len(materials)))
    # for n in range(time_steps + 1):
    #     # entropy[n, index] = s_if(solidTemp[n, node[index]], ap_field[n, node[index]])[0, 0]
    #     entropy[n, index] = s_if(solidTemp[n, node[index]], int_mag_field[n, int(node[index]/2)])[0, 0]
    # plt.plot(solidTemp[:, node[index]], entropy[:, index], color=colors[index], marker='+')

    # Plotting cycles on a Cp vs T diagram (experienced by the center node of each layer)

    # C = np.loadtxt(mat+'/'+mat+'_cp_h.txt')
    # c_if = matCp_h(C)
    # heatcap = np.zeros((time_steps + 1, len(materials)))
    # for n in range(time_steps + 1):
    #     # heatcap[n, index] = c_if(solidTemp[n, node[index]], ap_field[n, node[index]])[0, 0]
    #     heatcap[n, index] = c_if(solidTemp[n, node[index]], int_mag_field[n, int(node[index]/2)])[0, 0]
    # plt.figure(1)
    # plt.plot(solidTemp[:, node[index]], heatcap[:, index], color=colors[index], marker='+')

    index = index + 1


# Outputs
# TODO include also magnetization curves? at 0 T and 0.9 T??
plt.figure(1)
plt.xlabel('T [K]')
plt.ylabel('$C_{p}$ [J kg$^{-1}$ K$^{-1}$]')
# plt.xlim(temperature_range[0], temperature_range[1])
plt.grid(which='major', axis='both')
# plt.legend(['Gd - 0 T', 'Gd - 0.9 T', 'Gd - 1.4 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0.9 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 1.4 T'])
plt.legend(cp_legends, loc='upper left')

plt.figure(2)
plt.xlabel('T [K]')
plt.ylabel('$-\u0394s_{m}$ [J kg$^{-1}$ K$^{-1}$]')
# plt.xlim(temperature_range[0], temperature_range[1])
plt.grid(which='major', axis='both')
plt.legend(['Gd - 0.9 T', 'Gd - 1.4 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0.9 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 1.4 T'])

plt.figure(3)
plt.xlabel('T [K]')
plt.ylabel('$\u0394T_{ad}$ [K]')
# plt.xlim(temperature_range[0], temperature_range[1])
plt.grid(which='major', axis='both')
# plt.legend(['Gd - 0.9 T', 'Gd - 1.4 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0.9 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 1.4 T'])

plt.figure(4)
plt.xlabel('T [K]')
plt.ylabel('s [J kg$^{-1}$ K$^{-1}$]')
plt.xlim(temperature_range[0], temperature_range[1])
# plt.xlim(295,317)
plt.ylim(63, 115)
plt.grid(which='major', axis='both')

plt.figure(5)
plt.xlabel('T [K]')
plt.ylabel('M [A m$^2$ kg$^{-1}$]')
# plt.xlim(temperature_range[0], temperature_range[1])
# plt.xlim(295,317)
# plt.ylim(65, 110)
plt.grid(which='major', axis='both')
plt.legend(['Gd - 0.9 T', 'Gd - 1.4 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0.9 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 1.4 T'])


# plt.show()


# 2) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ploting on TS diagrams %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import importlib

# directory = "../../output/FAME_20layer_infl_Thot_flow"
directory = "output/FAME_MnFePSi/FAME_Dsp300um_B1400mT_ff_vfl4"  # "output/FAME_20layer_infl_Thot_flow2"
# inputs_file_name = 'FAME_20layer_infl_Thot_flow'  # File were the values of the input variables were defined.
inputs_file_name = "FAME_Dsp300um_B1400mT_ff_vfl4"  # "Run_parallel"
case_to_plot = 2637
# inputs = importlib.import_module(directory.replace('/', '.').replace('.', '', 6)+'.'+inputs_file_name)
inputs = importlib.import_module(directory.replace('/', '.')+'.'+inputs_file_name)



# 2.1) Simulation parameters

Thot        = 310    # [K]
Tcold       = 283   # [K]
applied_field = 1.4
# dThot       = 4      # [K] Maximum temperature difference between the fluid leaving hot side and hot reservoir
# dTcold      = 4      # [K] Maximum temperature difference between the fluid leaving cold side and cold reservoir
Reng_Length = 0.060  # [mm]
nodes       = 600   #  1800  # [-]
time_steps  = 200 # 600     # [-]
# node = [48, 143, 237, 332, 427, 521, 616, 711, 805, 900, 995, 1089, 1184, 1279, 1373, 1468, 1563, 1657, 1752] # 19 layers 1800 columns in temperature matrices
# node = [16, 48, 79, 111, 142, 174, 205, 237, 268, 300, 332, 363, 395, 426, 458, 489, 521, 552, 584]  # for 19 layers

node_min = [1, 61, 121, 181, 241, 301, 361, 421, 481, 541]#[1, 44, 87, 130, 173, 216, 259, 301, 344, 387, 430, 473, 516, 559]  #[1, 151, 301, 451]  # for 12 layers 600 columns in temperature matrix
node_mid = [30, 90, 150, 210, 270, 330, 390, 450, 510, 570] #[22, 65, 108, 150, 193, 236, 279, 322, 365, 408, 450, 493, 536, 579]  #[75,	225, 375, 525]
node_max = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600] #[43, 86, 129, 172, 215, 258, 300, 343, 386, 429, 472, 515, 558, 600]  #[150, 300, 450, 600]

# node = [30, 90, 150, 210, 270, 330, 390, 450, 510, 570]
# node = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 375, 405, 435, 465, 495, 525, 555, 585]

# 2.2) %%%%%%%%%%%%%%%%%%% Input and output file names %%%%%%%%%%%%%%%%%%

# text_file_input = "../../output/FAME_10layer_Th312K_Tc298K_infl_ff_flow2/146.0FAME_10layer_Th312K_Tc298K_infl_ff_flow.sh-146.txt"
# text_file_input = "../../output/FAME_Dsp300um_B900mT_ff_vfl2/2177.0FAME_Dsp300um_B900mT_ff_vfl2_reused.txt"
text_file_input = "../../output/FAME_MnFePSi/FAME_Dsp300um_B1400mT_ff_vfl4/2637.0FAME_Dsp300um_B1400mT_ff_vfl4_reused.txt"

# 2.3) %%%%%% Getting temperature data if the file only contains data corresponding to the temp. of solid or fluid %%%%%

# # Note: uncomment if section 3) is going to be used
#
# # fluid_temp_data_file_name = '../../Ends/9KGroup3FluidTemp.txt'
# # solid_temp_data_file_name = '../../Ends/9KGroup3SolidTemp.txt'
# # fluidTemp = np.loadtxt(fluid_temp_data_file_name)
# # solidTemp = np.loadtxt(solid_temp_data_file_name)
#
# # 2.4) %%%%%% Getting temperature data if the input file contains more than just the temp of solid or fluid %%%%%%

myfile = open(text_file_input, "rt")
contents = myfile.read()
myfile.close()

fluidTemp = np.ones((time_steps+1, nodes+1))
solidTemp = np.ones((time_steps+1, nodes+1))
int_mag_field = np.ones((time_steps+1, 300+1))  # np.ones((time_steps+1, nodes+1))

for j in range(time_steps+1):
    # fluid T matrix must start in row 3
    fluidTemp[j] = [float(i) for i in ((contents.split('\n'))[3+j].split())]
    # Solid T matrix must start in row time_steps+6
    solidTemp[j] = [float(i) for i in ((contents.split('\n'))[time_steps+5+j].split())]
    int_mag_field[j] = [float(i) for i in ((contents.split('\n'))[(time_steps+1)*2+8+j].split())]

# 2.5) Converting normalized temperatures to temperatures in [K]

fluidTemp = fluidTemp*(Thot-Tcold)+Tcold
solidTemp = solidTemp*(Thot-Tcold)+Tcold


index = 0
for mat in materials:

    # Plotting thermodynamic cycles experienced by the center node of each layer on a ST diagram
    S = np.loadtxt(mat+'/'+mat+'_S_h.txt')

    plt.figure(4)
    s_if = matS_h(S)
    entropy_min = np.zeros((time_steps + 1, len(materials)))
    entropy_mid = np.zeros((time_steps + 1, len(materials)))
    entropy_max = np.zeros((time_steps + 1, len(materials)))
    for n in range(time_steps + 1):
        # entropy[n, index] = s_if(solidTemp[n, node[index]], ap_field[n, node[index]])[0, 0]
        entropy_min[n, index] = s_if(solidTemp[n, node_min[index]], int_mag_field[n, int(node_min[index]/2)])[0, 0]
        entropy_mid[n, index] = s_if(solidTemp[n, node_mid[index]], int_mag_field[n, int(node_mid[index]/2)])[0, 0]
        entropy_max[n, index] = s_if(solidTemp[n, node_max[index]], int_mag_field[n, int(node_max[index]/2)])[0, 0]
    plt.plot(solidTemp[:, node_min[index]], entropy_min[:, index], color=colors[index], marker='+')
    plt.plot(solidTemp[:, node_mid[index]], entropy_mid[:, index], color=colors[index], marker='o')
    plt.plot(solidTemp[:, node_max[index]], entropy_max[:, index], color=colors[index], marker='+')

# Calulating area of loops in a TS diagram

int_discription = np.zeros(nodes+1, dtype=np.int)
species_descriptor = []
xloc = np.zeros(nodes+1)
# Set the rest of the nodes to id with geoDis(cription)

for i in range(N+1): # sets 0->N
    xloc[i] = (DX * i + DX / 2)  #modify i so 0->N
    if (xloc[i] >= x_discription[nn + 1]):
        nn = nn + 1
    int_discription[i] = nn
    species_descriptor.append(species_discription[nn])


    # Plotting cycles on a Cp vs T diagram (experienced by the center node of each layer)

    # C = np.loadtxt(mat+'/'+mat+'_cp_h.txt')
    # c_if = matCp_h(C)
    # heatcap = np.zeros((time_steps + 1, len(materials)))
    # for n in range(time_steps + 1):
    #     # heatcap[n, index] = c_if(solidTemp[n, node[index]], ap_field[n, node[index]])[0, 0]
    #     heatcap[n, index] = c_if(solidTemp[n, node[index]], int_mag_field[n, int(node[index]/2)])[0, 0]
    # plt.figure(1)
    # plt.plot(solidTemp[:, node[index]], heatcap[:, index], color=colors[index], marker='+')

    index = index + 1

plt.show()

# ------------------------------------------------------------------------

# #
# # from sourcefiles.device.FAME_app_field import app_field
# # ap_field = app_field(time_steps, nodes, applied_field)
# #
# # ff = 2
# # dispV         = 3 * 16.667e-6
# # acc_period    = 5
# # max_flow_per  = 45
# # full_magn_ang = 30
# # unbal_rat     = 1
# # from sourcefiles.device.FAME_V_flow import vol_flow_rate
# # volum_flow_profile = vol_flow_rate(time_steps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)



# 3) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Instantaneous utilization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# nn = 0
# DX = Reng_Length / (nodes-1)
# int_discription = np.zeros(nodes+1, dtype=np.int)
# species_descriptor = []
# xloc = np.zeros(nodes+1)
# species_discription = ['reg-M6', 'reg-M7', 'reg-M8', 'reg-M9', 'reg-M10', 'reg-M11', 'reg-M12', 'reg-M13', 'reg-M14', 'reg-M15']
# x_discription = [0, 0.006, 0.012, 0.018, 0.024, 0.030, 0.036, 0.042, 0.048, 0.054, 0.060]
# # Set the rest of the nodes to id with geoDis(cription)
#
# for i in range(nodes+1): # sets 0->N
#     xloc[i] = (DX * i + DX / 2)  #modify i so 0->
#
#     if xloc[i] >= x_discription[nn + 1]+DX and i != nodes:
#         nn = nn + 1
#     print(i, xloc[i], x_discription[nn]+DX, nn)
#     int_discription[i] = nn
#     species_descriptor.append(materials[nn])
#
# inst_cp = np.zeros(time_steps+1)
# inst_uti = np.zeros(time_steps+1)
#
# c_int_funct = []
# for mat in materials:
#     C = np.loadtxt(mat+'/'+mat+'_cp_h.txt')
#     c_int_funct.append(matCp_h(C))
#
#
# for n in range(time_steps + 1):
#
#     # This part is for the calculation of the instantaneous utilization
#     mCp_MCM = 0
#     for x in range(1, nodes):
#         # if x < 5 or x > nodes - 5:
#         #     print(x)
#         c_if = c_int_funct[int_discription[x]]
#         mCp_MCM = mCp_MCM + 6100 * (0.045 * 0.013 * 0.060 * 0.64) / (nodes-2) * c_if(solidTemp[n, x], ap_field[n, x])[0, 0]
#     inst_cp[n] = mCp_MCM
#     inst_uti[n] = 1000 * volum_flow_profile[n] * 4200 * (50 / 180) * ff / mCp_MCM
#
# plt.figure(5)
# plt.plot(np.linspace(0, time_steps, time_steps+1), inst_uti)
# plt.grid(which='major', axis='both')
# plt.xlabel('Time steps []')
# plt.ylabel("Instantaneous utilization of bed [J/K]")
#
# plt.figure(6)
# plt.plot(np.linspace(0, time_steps, time_steps+1), inst_cp)
# plt.grid(which='major', axis='both')
# plt.xlabel('Time steps []')
# plt.ylabel("Instantaneous thermal mass of bed [J/K]")
#
# plt.show()


