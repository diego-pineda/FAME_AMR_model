import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
import random
import pandas as pd
import PyQt5
import addcopyfighandler

mpl.use('Qt5Agg')
# ----------------------------------------------------------------------------------------------------------------


def matS_h(datstot_h):  # TODO this should be called matS without the _h because this is just an interpolation function
    # Entropy Heating
    HintStot_h = datstot_h[0, 1:]
    TempStot_h = datstot_h[1:, 0]
    mS_h = RectBivariateSpline(TempStot_h, HintStot_h, datstot_h[1:, 1:], kx=1, ky=1)
    return mS_h


def matCp_h(cpdat_h):  # TODO this should be called matCp without the _h because this is just an interpolation function
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
path_dir = './sourcefiles/new_mat/'
id_first_mat = 1000  #328 #166
id_last_mat = 1000 #357 #179
increment = 1
materials = []
mag_field_to_plot = 0.9
for i in range(id_first_mat, id_last_mat + 1, increment):  # [268, 283, 297]:  #
    materials.append('M' + str(i))
# materials = ['M6', 'M7', 'M8',  'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15']
# materials = ['M0', 'M2']
# materials = ['M601', 'M333', 'M334',  'M335', 'M336', 'M337', 'M338', 'M339', 'M340', 'M341', 'M342', 'M343', 'M344', 'M345', 'M346', 'M347', 'M348', 'M349', 'M350', 'M351', 'M352', 'M602']
temperature_range = [260, 310]  # [K] Used for plots
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'olive', 'orangered', 'indigo', 'crimson', 'grey', 'royalblue', 'khaki',
          'darkorange', 'slategray', 'deeppink', 'teal', 'peru', 'darkviolet', 'hotpink', 'darkorange','rosybrown', 'sienna', 'steelblue',
          'navy', 'khaki', 'gold', 'royalblue', 'tan',
          'b', 'r', 'g', 'c', 'm', 'y', 'k', 'olive', 'orangered', 'indigo', 'crimson', 'grey', 'royalblue', 'khaki',
          'darkorange', 'slategray', 'deeppink', 'teal', 'peru', 'darkviolet', 'hotpink', 'darkorange','rosybrown', 'sienna', 'steelblue',
          'navy', 'khaki', 'gold', 'royalblue', 'tan']

# colors = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])  # This works inside the loop

cp_legends = []
s_legends = []

# Calculations
index = 0
for mat in materials:
    # print(mat)
    plt.figure(1)  # Heat capacity vs Temperature
    try:
        cp_h_dataframe = pd.read_csv(mat+'/'+mat+'_cp_h.csv', sep=';', header=None)
        cp_h = pd.DataFrame(cp_h_dataframe).to_numpy()
        cp_c_dataframe = pd.read_csv(mat+'/'+mat+'_cp_c.csv', sep=';', header=None)
        cp_c = pd.DataFrame(cp_c_dataframe).to_numpy()
    except FileNotFoundError:
        cp_h_dataframe = pd.read_csv(mat+'/'+mat+'_cp_h.txt', sep='\t', lineterminator='\n', header=None)
        cp_h = pd.DataFrame(cp_h_dataframe).to_numpy()
        cp_c_dataframe = pd.read_csv(mat+'/'+mat+'_cp_c.txt', sep='\t', lineterminator='\n', header=None)
        cp_c = pd.DataFrame(cp_c_dataframe).to_numpy()

    # cp_h = np.loadtxt(mat+'/'+mat+'_cp_h.txt')
    # cp_c = np.loadtxt(mat+'/'+mat+'_cp_c.txt')
    T_cp_h = cp_h[1:, 0]
    T_cp_c = cp_c[1:, 0]
    # C_p_0T = cp_h[1:, 1]
    print(cp_c[0, :])
    # if index == 0 or index == 21:
    #     i_mag_field = list(cp_h[0, :]).index(1)
    # else:
    # i_mag_field = list(cp_h[0, :]).index(mag_field_to_plot)
    i_mag_field = list(cp_c[0, :]).index(mag_field_to_plot)

    # C_p_09T = cp_h[1:, i_mag_field]
    # Cp_hi_field = cp_h[1:, i_mag_field]
    # plt.plot(T_cp_h, cp_h[1:, 1], color=colors[index], linestyle='dotted')  # Heating zero field
    # plt.plot(T_cp_c, cp_c[1:, 1], color=colors[index], linestyle='dashed')  # Cooling zero field
    plt.plot(T_cp_h, cp_h[1:, i_mag_field], color=colors[index], linestyle='solid')  # Heating high field
    # plt.plot(T_cp_c, cp_c[1:, i_mag_field], color=colors[index], linestyle='dashdot')  # Cooling high field
    # plt.show()

    cp_legends.append('{} - {} T'.format(mat, 0))
    # cp_legends.append('{} - {} T cooling'.format(mat, 0))
    cp_legends.append('{} - {} T'.format(mat, mag_field_to_plot))
    # cp_legends.append('{} - {} T cooling'.format(mat, mag_field_to_plot))

    # plt.plot(T_cp, Cp_hi_field, color=colors[index], linestyle='dashed')




    plt.figure(2)  # Magnetic entropy change vs Temperature

    try:
        s_h_df = pd.read_csv(mat+'/'+mat+'_S_h.csv', sep=';', header=None)
        s_h = pd.DataFrame(s_h_df).to_numpy()
        s_c_df = pd.read_csv(mat+'/'+mat+'_S_c.csv', sep=';', header=None)
        s_c = pd.DataFrame(s_c_df).to_numpy()
    except FileNotFoundError:
        s_h_df = pd.read_csv(mat+'/'+mat+'_S_h.txt', sep='\t', lineterminator='\n', header=None)
        s_h = pd.DataFrame(s_h_df).to_numpy()
        s_c_df = pd.read_csv(mat+'/'+mat+'_S_c.txt', sep='\t', lineterminator='\n', header=None)
        s_c = pd.DataFrame(s_c_df).to_numpy()

    # s_h = np.loadtxt(mat+'/'+mat+'_S_h.txt')
    # s_c = np.loadtxt(mat+'/'+mat+'_S_c.txt')


    T_s_h = s_h[1:, 0]
    # T_s_c = s_c[1:, 0]
    dS_m_h = s_h[1:, 1] - s_h[1:, i_mag_field]  # Negative magnetic entropy change from 0 T to 0.9 T
    # dS_m_c = s_c[1:, 1] - s_c[1:, i_mag_field]
    # dS_m_hi_field = s_h[1:, 1] - s_h[1:, 15]
    plt.plot(T_s_h, dS_m_h, color=colors[index], linestyle='solid')
    # plt.plot(T_s_c, dS_m_c, color=colors[index], linestyle='dashed')



    plt.figure(4)  # ST diagram
    plt.plot(T_s_h, s_h[1:, 1], color=colors[index], linestyle='solid')  # Heating low field
    # plt.plot(T_s_c, s_c[1:, 1], color=colors[index], linestyle='dashed')  # Cooling low field
    plt.plot(T_s_h, s_h[1:, i_mag_field], color=colors[index], linestyle='solid')  # Heating high field
    # plt.plot(T_s_c, s_c[1:, i_mag_field], color=colors[index], linestyle='dashdot')  # Cooling high field

    s_legends.append('{} - {} T'.format(mat, 0))
    # s_legends.append('{} - {} T cooling'.format(mat, 0))
    s_legends.append('{} - {} T'.format(mat, mag_field_to_plot))
    # s_legends.append('{} - {} T cooling'.format(mat, mag_field_to_plot))

    # plt.show()

    plt.figure(3)  # Adiabatic temperature change vs Temperature
    s_0_h = interp1d(s_h[1:, 0], s_h[1:, 1], kind='cubic')
    T_1_h = interp1d(s_h[1:, i_mag_field], s_h[1:, 0], kind='cubic')
    s_0_c = interp1d(s_c[1:, 0], s_c[1:, 1], kind='cubic')
    T_1_c = interp1d(s_c[1:, i_mag_field], s_c[1:, 0], kind='cubic')
    # T_hi_field = interp1d(s_h[1:, 15], s_h[1:, 0], kind='cubic')
    dT_ad_h = []
    dT_ad_c = []
    # dT_hi_field = []
    # T_set = np.linspace(temperature_range[0]+index, temperature_range[1]-20+index, 300)  # TODO: generalize these limits
    T_set = np.linspace(s_h[1, 0], s_h[-30, 0], 300)
    for Temperature in T_set:
        dT_ad_h.append(T_1_h(s_0_h(Temperature))-Temperature)
        # dT_ad_c.append(T_1_c(s_0_c(Temperature))-Temperature)
        # dT_hi_field.append(T_hi_field(s_0_h(Temperature))-Temperature)
    plt.plot(T_set, dT_ad_h, color=colors[index], linestyle='solid')
    # plt.plot(T_set, dT_ad_c, color=colors[index], linestyle='dashed')
    # plt.plot(T_set, dT_hi_field, color=colors[index], linestyle='dashed')

    plt.figure(5)  # MT curves

    try:
        Mag_h_df = pd.read_csv(mat+'/'+mat+'_Mag_h.csv', sep=';', header=None)
        Mag = pd.DataFrame(Mag_h_df).to_numpy()

    except FileNotFoundError:
        Mag_h_df = pd.read_csv(mat+'/'+mat+'_Mag_h.txt', sep='\t', lineterminator='\n', header=None)
        Mag = pd.DataFrame(Mag_h_df).to_numpy()

    # Mag = np.loadtxt(mat+'/'+mat+'_Mag_h.txt')
    T_mag = Mag[1:, 0]

    # Mag_0_1T = Mag[1:, 2]
    M_curves_to_plot = [0.6, 1, 1.4, 1.8]
    for m in M_curves_to_plot:
        i_mag_field = list(Mag[0, :]).index(m)
        Mag_hi_field = Mag[1:, i_mag_field]
        plt.plot(T_mag, Mag_hi_field, linestyle='solid')

    # i_mag_field = list(Mag[0, :]).index(mag_field_to_plot)
    # print(i_mag_field)
    # Mag_hi_field = Mag[1:, i_mag_field]
    # plt.plot(T_mag, Mag_0_1T, color=colors[index])
    # plt.plot(T_mag, Mag_hi_field, color=colors[index], linestyle='solid')
    # plt.plot(T_mag, Mag[1:, 15], color=colors[index], linestyle='dashed')

    plt.figure(6)  # dMdT
    i = 0
    dMdT_lo_field = []
    dMdT_hi_field = []
    for temp in Mag[2:-1, 0]:
        # dMdT_lo_field.append((Mag[3+i, 10]-Mag[1+i, 10]) / (Mag[3+i, 0] - Mag[1+i, 0]))
        dMdT_hi_field.append((Mag[3+i, i_mag_field]-Mag[1+i, i_mag_field]) / (Mag[3+i, 0] - Mag[1+i, 0]))
        i = i+1
    # plt.plot(Mag[2:-1, 0], dMdT_lo_field, color=colors[index], linestyle='solid')
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
plt.figure(1)  # Cp vs T
plt.xlabel('$T_s$ [K]', fontsize=12)
plt.ylabel('$C_{p}$ [J kg$^{-1}$ K$^{-1}$]', fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
# plt.xlim(temperature_range[0], temperature_range[1])
# plt.grid(which='major', axis='both')
# plt.legend(['Gd - 0 T', 'Gd - 0.9 T', 'Gd - 1.4 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0.9 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 1.4 T'])
# plt.legend(cp_legends, loc='upper left')

plt.figure(2)  # Magnetic entropy change vs Temperature
plt.xlabel('T [K]')
plt.ylabel('$-\u0394s_{m}$ [J kg$^{-1}$ K$^{-1}$]')
# plt.xlim(temperature_range[0], temperature_range[1])
# plt.grid(which='major', axis='both')
# plt.legend(["HH path", "CC path"])
# plt.legend(['Gd - 0.9 T', 'Gd - 1.4 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0.9 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 1.4 T'])

plt.figure(3)  # Adiabatic temperature change vs Temperature
plt.xlabel(r'$T_s$ [K]')
plt.ylabel('$\u0394T_{ad}$ [K]')
# plt.xlim(280, 315)
# plt.ylim(0, 3)
# plt.hlines(2, 260, 350, colors='k')
# plt.vlines([293.608, 303.4515], 0, 3, colors=['k', 'k'])
# plt.text(282, 2.7, 'a)', fontsize=15)
# plt.xlim(temperature_range[0], temperature_range[1])
# plt.grid(which='major', axis='both')
# plt.legend(['HH path', 'CC path'])
# plt.legend(['Gd - 0.9 T', 'Gd - 1.4 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0.9 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 1.4 T'])

plt.figure(4)  # S vs T
plt.xlabel(r'$T_s$ [K]')
plt.ylabel(r'$s$ [J kg$^{-1}$ K$^{-1}$]')
plt.xlim(280, 315)
plt.ylim(60, 120)
plt.vlines([293.608, 303.4515], 60, 120, colors=['k', 'k'])
plt.hlines([75.0682, 104.551], 280, 315, colors=['k', 'k'])
plt.text(282, 115, 'b)', fontsize=15)
# plt.xlim(temperature_range[0], temperature_range[1])
# plt.xlim(270, 320)
# plt.ylim(30, 130)
# plt.grid(which='major', axis='both')
# plt.legend(s_legends, loc='upper left')
# plt.legend(['0 T', '1.4 T'])

plt.figure(5)
plt.xlabel('T [K]')
plt.ylabel('M [A m$^2$ kg$^{-1}$]')
# plt.xlim(temperature_range[0], temperature_range[1])
# plt.xlim(295,317)
# plt.ylim(65, 110)
# plt.grid(which='major', axis='both')
# plt.legend(['Gd - 0.9 T', 'Gd - 1.4 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 0.9 T', 'Mn$_{1.18}$Fe$_{0.73}$P$_{0.48}$Si$_{0.52}$ - 1.4 T'])
plt.legend(['0.6 T', '1.0 T', '1.4 T', '1.8 T'])

plt.show()


# 2) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ploting on TS diagrams %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import importlib

# directory = "../../output/FAME_20layer_infl_Thot_flow"
directory = "output/FAME_MnFePSi/internal_voids_DDMC/results"  #   "output/FAME_20layer_infl_Thot_flow2"
# inputs_file_name = 'FAME_20layer_infl_Thot_flow'  # File were the values of the input variables were defined.
inputs_file_name = "FAME_PSB_Dsp300um_B1400mT_L60mm_ff_vfl_Lvoid"  #  "Run_parallel"
case = 6
# inputs = importlib.import_module(directory.replace('/', '.').replace('.', '', 6)+'.'+inputs_file_name)
inputs = importlib.import_module(directory.replace('/', '.')+'.'+inputs_file_name)

variable_1_name = inputs.vble1name  # This must be either Thot or any other variable used in X_resolution
variable_1_units = inputs.vble1units
variable_1_values = inputs.vble1values
variable_1_resolution = inputs.vble1resolution
variable_2_name = inputs.vble2name # This must be the variable changed inside the if conditions in the inputs file
variable_2_units = inputs.vble2units
variable_2_values = inputs.vble2values  # [units] Variable name. Note: values used for variable 2 in the cases simulated
variable_2_resolution = inputs.vble2resolution
variable_3_name = inputs.vble3name # This must be the variable changed inside the if conditions in the inputs file
variable_3_units = inputs.vble3units
variable_3_values = inputs.vble3values  # [units] Variable name. Note: values used for variable 2 in the cases simulated
variable_3_resolution = inputs.vble3resolution

cases = inputs.numGroups
hot_resolution = inputs.hotResolution
span_resolution = inputs.TspanResolution

Thot_values = inputs.Thotarr
Tspan_values = inputs.Tspanarr

materials_per_config = inputs.layer_description
layer_interp_pos_per_config = inputs.layer_positions  # TODO: generalize! This is particular for this batch of cases


casegroup = int(np.floor(case / (span_resolution * hot_resolution)))

a = int(np.floor((casegroup - variable_1_resolution * int(np.floor(casegroup / variable_1_resolution))) / 1))
b = int(np.floor((casegroup - variable_1_resolution * variable_2_resolution * int(np.floor(casegroup / (variable_1_resolution * variable_2_resolution)))) / variable_1_resolution))
c = int(np.floor((casegroup - variable_1_resolution * variable_2_resolution * variable_3_resolution * int(np.floor(casegroup / (variable_1_resolution * variable_2_resolution * variable_3_resolution)))) / (variable_1_resolution * variable_2_resolution)))
y = int(np.floor(case / span_resolution) % hot_resolution)
x = case % span_resolution

# 2.1) Simulation parameters

Thot        = Thot_values[y]  # 310    # [K]
Tcold       = Thot - Tspan_values[x]  # 283   # [K]
applied_field = 1.4
# dThot       = 4      # [K] Maximum temperature difference between the fluid leaving hot side and hot reservoir
# dTcold      = 4      # [K] Maximum temperature difference between the fluid leaving cold side and cold reservoir
Reg_Length = 0.060  # [mm]
nodes       = 1000   #  1800  # [-]
time_steps  = 400  # 600     # [-]
# node = [48, 143, 237, 332, 427, 521, 616, 711, 805, 900, 995, 1089, 1184, 1279, 1373, 1468, 1563, 1657, 1752] # 19 layers 1800 columns in temperature matrices
# node = [16, 48, 79, 111, 142, 174, 205, 237, 268, 300, 332, 363, 395, 426, 458, 489, 521, 552, 584]  # for 19 layers

# Materials per layer

materials_this_case = materials_per_config[c]  # TODO: generalize! This is particular for this batch of cases
materials = list(s.strip('reg-') for s in materials_this_case)
layer_interp_pos_this_case = layer_interp_pos_per_config[c]
print(materials_this_case)

# Assigning a material to each node
nn = 0
int_discription = np.zeros(nodes+1, dtype=int)
species_descriptor = []
node_min = [1]
node_max = []
xloc = np.zeros(nodes+1)
# Set the rest of the nodes to id with geoDis(cription)
DX = Reg_Length / (nodes - 1)
for i in range(nodes+1):  # sets 0->N-2 which is from 1st node with material to last node with material, so no ghosts
    xloc[i] = (DX * i + DX / 2)  # modify i so 0->N
    if xloc[i] >= layer_interp_pos_this_case[nn + 1] + DX and i < nodes:
        nn = nn + 1
        node_max.append(i-1)
        node_min.append(i)
    int_discription[i] = nn
    species_descriptor.append(materials[nn])
node_max.append(nodes)
node_mid = [int(i) for i in (np.array(node_min) + np.array(node_max))/2]
print(species_descriptor)

# node_min = [1, 51,101,151,201,251,301,351,401,451,501,551]
# node_mid = [25,75,125,175,225,275,325,375,425,475,525,575]
# node_max = [50,100,150,200,250,300,350,400,450,500,550,600]

# node_min = [1,  21, 41, 61,  81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501, 521, 541, 561, 581]  # [1, 61, 121, 181, 241, 301, 361, 421, 481, 541]  #[1, 151, 301, 451]  # for 12 layers 600 columns in temperature matrix
# node_mid = [10, 30, 50, 70,  90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450, 470, 490, 510, 530, 550, 570, 590]  # [30, 90, 150, 210, 270, 330, 390, 450, 510, 570]  #[75,	225, 375, 525]
# node_max = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600]  # [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]  #[150, 300, 450, 600]

# node_min = [1, 51,101,151,201,251,301,351,401,451,501,551]
# node_mid = [25,75,125,175,225,275,325,375,425,475,525,575]
# node_max = [50,100,150,200,250,300,350,400,450,500,550,600]

# node = [30, 90, 150, 210, 270, 330, 390, 450, 510, 570]
# node = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 375, 405, 435, 465, 495, 525, 555, 585]

# 2.2) %%%%%%%%%%%%%%%%%%% Input and output file names %%%%%%%%%%%%%%%%%%

# text_file_input = "../../output/FAME_10layer_Th312K_Tc298K_infl_ff_flow2/146.0FAME_10layer_Th312K_Tc298K_infl_ff_flow.sh-146.txt"
# text_file_input = "../../output/FAME_Dsp300um_B900mT_ff_vfl2/2177.0FAME_Dsp300um_B900mT_ff_vfl2_reused.txt"
# text_file_input = "../../output/FAME_MnFePSi/FAME_Dsp300um_B1400mT_ff_vfl4/2637.0FAME_Dsp300um_B1400mT_ff_vfl4_reused.txt"  # TODO: never use the complementary words again, e.g. "reused"
text_file_input = '../../' + directory + '/' + str(case) + '.0' + inputs_file_name #+ '.txt'  # TODO
text_file_input = text_file_input + '.txt'  # TODO: generalize. This is only for cases reused from previous simulations
#
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
int_mag_field = np.ones((time_steps+1, int(nodes)+1))  # np.ones((time_steps+1, nodes+1))  # TODO

for j in range(time_steps+1):
    # fluid T matrix must start in row 3
    fluidTemp[j] = [float(i) for i in ((contents.split('\n'))[3+j].split())]
    # Solid T matrix must start in row time_steps+6
    solidTemp[j] = [float(i) for i in ((contents.split('\n'))[time_steps+5+j].split())]
    int_mag_field[j] = [float(i) for i in ((contents.split('\n'))[(time_steps+1)*2+8+j].split())]

# 2.5) Converting normalized temperatures to temperatures in [K]

fluidTemp = fluidTemp*(Thot-Tcold)+Tcold
solidTemp = solidTemp*(Thot-Tcold)+Tcold

# ----------------------- Calculation of the area of loops in TS diagram ------------------------ (section in between)

s_int_functions = []
for mat in materials:
    S = np.loadtxt(mat+'/'+mat+'_S_h.txt')
    # plt.figure(4)
    s_int_functions.append(matS_h(S))

W_mag_AMR = 0
s_t0 = np.zeros(nodes+1)
s_t126 = np.zeros(nodes+1)
entropy_node0 = np.zeros(time_steps)
entropy_mid_node = np.zeros(time_steps)
entropy_last_node = np.zeros(time_steps)
for node in range(0, nodes, 4):
    w_mag_node = 0
    for n in range(time_steps):
        s_current = s_int_functions[int_discription[node+1]](solidTemp[n, node + 1], int_mag_field[n, int((node+1)/2)])[0, 0]
        plt.figure(7)
        if node == 0:
            color_node = 'red'
            # solidTemp_node0[n] = solidTemp[n, node + 1]
            entropy_node0[n] = s_current
        elif node == 1000:
            entropy_mid_node[n] = s_current
            color_node = 'red'
        elif node == nodes-1:
            entropy_last_node[n] = s_current
            color_node = 'red'
        else:
            color_node = 'powderblue'
        plt.plot(solidTemp[n, node + 1], s_current, marker='o', color=color_node)
        s_next = s_int_functions[int_discription[node+1]](solidTemp[n+1, node + 1], int_mag_field[n+1, int((node+1)/2)])[0, 0]
        Area_loop = 0.5 * (solidTemp[n, node+1] + solidTemp[n+1, node+1]) * (s_next - s_current)
        # print('ds = ', (s_next - s_current))
        w_mag_node = w_mag_node + Area_loop
        if n==0:
            s_t0[node] = s_current
        if n==126:
            s_t126[node] = s_current
    W_mag_node = w_mag_node * 6100 * (0.045*0.013*DX*(1-0.36))
    W_mag_AMR = W_mag_AMR + W_mag_node

plt.figure(7)
plt.plot(solidTemp[0:-1, 1], entropy_node0, marker='o', color='red')
plt.plot(solidTemp[0:-1, 1000], entropy_mid_node, marker='o', color='red')
plt.plot(solidTemp[0:-1, -1], entropy_last_node, marker='o', color='red')

# Plot entropy curves for node 0
s_h = np.loadtxt(species_descriptor[1]+'/'+species_descriptor[1]+'_S_h.txt')
T_s_h = s_h[1:, 0]
plt.figure(7)  # ST diagram
plt.plot(T_s_h, s_h[1:, 1], color='k', linestyle='solid')  # Heating low field
# plt.plot(T_s_c, s_c[1:, 1], color=colors[index], linestyle='dashed')  # Cooling low field
plt.plot(T_s_h, s_h[1:, 15], color='k', linestyle='solid')  # Heating high field

# Plot entropy curves for node 300
s_h = np.loadtxt(species_descriptor[1000]+'/'+species_descriptor[1000]+'_S_h.txt')
T_s_h = s_h[1:, 0]
plt.figure(7)  # ST diagram
plt.plot(T_s_h, s_h[1:, 1], color='k', linestyle='solid')  # Heating low field
# plt.plot(T_s_c, s_c[1:, 1], color=colors[index], linestyle='dashed')  # Cooling low field
plt.plot(T_s_h, s_h[1:, 15], color='k', linestyle='solid')  # Heating high field

# Plot entropy curves for node 600
s_h = np.loadtxt(species_descriptor[2000]+'/'+species_descriptor[2000]+'_S_h.txt')
T_s_h = s_h[1:, 0]
plt.figure(7)  # ST diagram
plt.plot(T_s_h, s_h[1:, 1], color='k', linestyle='solid')  # Heating low field
# plt.plot(T_s_c, s_c[1:, 1], color=colors[index], linestyle='dashed')  # Cooling low field
plt.plot(T_s_h, s_h[1:, 15], color='k', linestyle='solid')  # Heating high field

plt.axhline(y=105, color='k', linestyle='solid')  # xmin=250, xmax=350,
plt.axhline(y=75, color='k', linestyle='solid')  # xmin=250, xmax=350,
# plt.xlim((300, 310))
# plt.show()

plt.figure(4)
plt.scatter(solidTemp[0, :], s_t0)
plt.scatter(solidTemp[126, :], s_t126)

ff = variable_2_values[b]
P_mag_AMR = W_mag_AMR * ff
print('Total magnetic power is: {} [W]'.format(P_mag_AMR))

# ----- Plotting thermodynamic cycles experienced by the center node of each layer on a ST diagram

index = 0
for mat in materials:

    S = np.loadtxt(mat+'/'+mat+'_S_h.txt')

    # ----------- 4-4-22 trial --------------
    S_c = np.loadtxt(mat+'/'+mat+'_S_c.txt')
    S_h = np.loadtxt(mat+'/'+mat+'_S_h.txt')
    # -------------------

    plt.figure(4)
    s_if = matS_h(S)

    # ----------- 4-4-22 trial --------------
    s_if_c = matS_h(S_c)
    s_if_h = matS_h(S_h)
    # -----------------------

    entropy_min = np.zeros((time_steps + 1, len(materials)))
    entropy_mid = np.zeros((time_steps + 1, len(materials)))
    entropy_max = np.zeros((time_steps + 1, len(materials)))
    for n in range(time_steps + 1):
        # entropy[n, index] = s_if(solidTemp[n, node[index]], ap_field[n, node[index]])[0, 0]
        entropy_min[n, index] = s_if(solidTemp[n, node_min[index]], int_mag_field[n, int(node_min[index])])[0, 0]
        entropy_mid[n, index] = s_if(solidTemp[n, node_mid[index]], int_mag_field[n, int(node_mid[index])])[0, 0]
        entropy_max[n, index] = s_if(solidTemp[n, node_max[index]], int_mag_field[n, int(node_max[index])])[0, 0]

        # ----------- 4-4-22 trial --------------

        # entropy_min[n, index] = 0.5 * s_if_c(solidTemp[n, node_min[index]], int_mag_field[n, int(node_min[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_min[index]], int_mag_field[n, int(node_min[index])])[0, 0]
        # entropy_mid[n, index] = 0.5 * s_if_c(solidTemp[n, node_mid[index]], int_mag_field[n, int(node_mid[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_mid[index]], int_mag_field[n, int(node_mid[index])])[0, 0]
        # entropy_max[n, index] = 0.5 * s_if_c(solidTemp[n, node_max[index]], int_mag_field[n, int(node_max[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_max[index]], int_mag_field[n, int(node_max[index])])[0, 0]

        # -----------
    if mat in ['M268', 'M283', 'M297']:
        # plt.plot(solidTemp[:, node_min[index]], entropy_min[:, index], color=colors[index], marker='v', markersize=2, markerfacecolor='white')
        plt.plot(solidTemp[:, node_mid[index]], entropy_mid[:, index], color=colors[index], marker='o', markersize=2, markerfacecolor='white')  # TODO. Activate this to plot cycle of center node
        # plt.plot(solidTemp[:, node_max[index]], entropy_max[:, index], color=colors[index], marker='s', markersize=2, markerfacecolor='white')

#
#
#
#     # ------------ Plotting cycles on a Cp vs T diagram (experienced by the center node of each layer) -------------
#
#     # C = np.loadtxt(mat+'/'+mat+'_cp_h.txt')
#     # c_if = matCp_h(C)
#     #
#     # heatcap_min = np.zeros((time_steps + 1, len(materials)))
#     # heatcap_mid = np.zeros((time_steps + 1, len(materials)))
#     # heatcap_max = np.zeros((time_steps + 1, len(materials)))
#     #
#     # # ----------- 4-4-22 trial --------------
#     # C_c = np.loadtxt(mat+'/'+mat+'_cp_c.txt')
#     # C_h = np.loadtxt(mat+'/'+mat+'_cp_h.txt')
#     # c_if_c = matCp_h(C_c)
#     # c_if_h = matCp_h(C_h)
#     # heatcap_hys_min = np.zeros((time_steps + 1, len(materials)))
#     # heatcap_hys_mid = np.zeros((time_steps + 1, len(materials)))
#     # heatcap_hys_max = np.zeros((time_steps + 1, len(materials)))
#     # dT = 0.5
#     # # -------------------
#     # for n in range(time_steps + 1):
#     #     # heatcap[n, index] = c_if(solidTemp[n, node[index]], ap_field[n, node[index]])[0, 0]
#     #     heatcap_min[n, index] = c_if(solidTemp[n, node_min[index]], int_mag_field[n, int(node_min[index]/2)])[0, 0]
#     #     heatcap_mid[n, index] = c_if(solidTemp[n, node_mid[index]], int_mag_field[n, int(node_mid[index]/2)])[0, 0]
#     #     heatcap_max[n, index] = c_if(solidTemp[n, node_max[index]], int_mag_field[n, int(node_max[index]/2)])[0, 0]
#     #
#     #     # ----------- 4-4-22 trial --------------
#     #     # ds = (0.5 * s_if_c(solidTemp[n, node_min[index]] + dT, int_mag_field[n, int(node_min[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_min[index]] + dT, int_mag_field[n, int(node_min[index])])[0, 0]) - (0.5 * s_if_c(solidTemp[n, node_min[index]] - dT, int_mag_field[n, int(node_min[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_min[index]] - dT, int_mag_field[n, int(node_min[index])])[0, 0])
#     #     # heatcap_hys_min[n, index] = solidTemp[n, node_min[index]] * ds / (2 * dT)
#     #     # ds = (0.5 * s_if_c(solidTemp[n, node_mid[index]] + dT, int_mag_field[n, int(node_mid[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_mid[index]] + dT, int_mag_field[n, int(node_mid[index])])[0, 0]) - (0.5 * s_if_c(solidTemp[n, node_mid[index]] - dT, int_mag_field[n, int(node_mid[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_mid[index]] - dT, int_mag_field[n, int(node_mid[index])])[0, 0])
#     #     # heatcap_hys_mid[n, index] = solidTemp[n, node_mid[index]] * ds / (2 * dT)
#     #     # ds = (0.5 * s_if_c(solidTemp[n, node_max[index]] + dT, int_mag_field[n, int(node_max[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_max[index]] + dT, int_mag_field[n, int(node_max[index])])[0, 0]) - (0.5 * s_if_c(solidTemp[n, node_max[index]] - dT, int_mag_field[n, int(node_max[index])])[0, 0] + 0.5 * s_if_h(solidTemp[n, node_max[index]] - dT, int_mag_field[n, int(node_max[index])])[0, 0])
#     #     # heatcap_hys_max[n, index] = solidTemp[n, node_max[index]] * ds / (2 * dT)
#     #     # -------------------
#     #
#     # plt.figure(1)
#     # plt.plot(solidTemp[:, node_min[index]], heatcap_min[:, index], color=colors[index], marker='+')
#     # plt.plot(solidTemp[:, node_mid[index]], heatcap_mid[:, index], color=colors[index], marker='+')  # TODO uncomment for plots without hysteresis
#     # plt.plot(solidTemp[:, node_max[index]], heatcap_max[:, index], color=colors[index], marker='+')
#     #
#     # # ----------- 4-4-22 trial --------------
#     # # plt.plot(solidTemp[:, node_min[index]], heatcap_hys_min[:, index], color=colors[index], marker='+')
#     # # plt.plot(solidTemp[:, node_mid[index]], heatcap_hys_mid[:, index], color=colors[index], marker='o')
#     # # plt.plot(solidTemp[:, node_max[index]], heatcap_hys_max[:, index], color=colors[index], marker='s')
#     # # -------------------
#
#     # ----------------------------------- end section Cp loops --------------------------------------------
#
    index = index + 1
plt.figure(4)
plt.plot([270, 320, 320, 270], [75, 75, 105, 105], '-k')
# plt.plot([293.59, 293.59, 303.65, 303.65], [0, 150, 150, 0], '-k')  # [277.39, 277.39, 287.45, 287.45]
# plt.figure(3)
# plt.plot([293.59, 293.59, 303.65, 303.65], [0, 3, 3, 0], '-k')
plt.show()
#
# # ------------------------------------------------------------------------
#
# # #
# # # from sourcefiles.device.FAME_app_field import app_field
# # # ap_field = app_field(time_steps, nodes, applied_field)
# # #
# # # ff = 2
# # # dispV         = 3 * 16.667e-6
# # # acc_period    = 5
# # # max_flow_per  = 45
# # # full_magn_ang = 30
# # # unbal_rat     = 1
# # # from sourcefiles.device.FAME_V_flow import vol_flow_rate
# # # volum_flow_profile = vol_flow_rate(time_steps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)
#
#
#
# # 3) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Instantaneous utilization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# # nn = 0
# # DX = Reng_Length / (nodes-1)
# # int_discription = np.zeros(nodes+1, dtype=np.int)
# # species_descriptor = []
# # xloc = np.zeros(nodes+1)
# # species_discription = ['reg-M6', 'reg-M7', 'reg-M8', 'reg-M9', 'reg-M10', 'reg-M11', 'reg-M12', 'reg-M13', 'reg-M14', 'reg-M15']
# # x_discription = [0, 0.006, 0.012, 0.018, 0.024, 0.030, 0.036, 0.042, 0.048, 0.054, 0.060]
# # # Set the rest of the nodes to id with geoDis(cription)
# #
# # for i in range(nodes+1): # sets 0->N
# #     xloc[i] = (DX * i + DX / 2)  #modify i so 0->
# #
# #     if xloc[i] >= x_discription[nn + 1]+DX and i != nodes:
# #         nn = nn + 1
# #     print(i, xloc[i], x_discription[nn]+DX, nn)
# #     int_discription[i] = nn
# #     species_descriptor.append(materials[nn])
# #
# # inst_cp = np.zeros(time_steps+1)
# # inst_uti = np.zeros(time_steps+1)
# #
# # c_int_funct = []
# # for mat in materials:
# #     C = np.loadtxt(mat+'/'+mat+'_cp_h.txt')
# #     c_int_funct.append(matCp_h(C))
# #
# #
# # for n in range(time_steps + 1):
# #
# #     # This part is for the calculation of the instantaneous utilization
# #     mCp_MCM = 0
# #     for x in range(1, nodes):
# #         # if x < 5 or x > nodes - 5:
# #         #     print(x)
# #         c_if = c_int_funct[int_discription[x]]
# #         mCp_MCM = mCp_MCM + 6100 * (0.045 * 0.013 * 0.060 * 0.64) / (nodes-2) * c_if(solidTemp[n, x], ap_field[n, x])[0, 0]
# #     inst_cp[n] = mCp_MCM
# #     inst_uti[n] = 1000 * volum_flow_profile[n] * 4200 * (50 / 180) * ff / mCp_MCM
# #
# # plt.figure(5)
# # plt.plot(np.linspace(0, time_steps, time_steps+1), inst_uti)
# # plt.grid(which='major', axis='both')
# # plt.xlabel('Time steps []')
# # plt.ylabel("Instantaneous utilization of bed [J/K]")
# #
# # plt.figure(6)
# # plt.plot(np.linspace(0, time_steps, time_steps+1), inst_cp)
# # plt.grid(which='major', axis='both')
# # plt.xlabel('Time steps []')
# # plt.ylabel("Instantaneous thermal mass of bed [J/K]")
# #
# # plt.show()
#
#
