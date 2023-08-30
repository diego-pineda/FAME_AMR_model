import numpy as np
import importlib
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d

''' 
README
Run this script on the python console
'''

def matS_h(datstot_h):  # TODO this should be called matS without the _h because this is just an interpolation function
    # Entropy Heating
    HintStot_h = datstot_h[0, 1:]
    TempStot_h = datstot_h[1:, 0]
    mS_h = RectBivariateSpline(TempStot_h, HintStot_h, datstot_h[1:, 1:], kx=1, ky=1)
    return mS_h

colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'olive', 'orangered', 'indigo', 'crimson', 'grey', 'royalblue', 'khaki',
          'darkorange', 'slategray', 'deeppink', 'teal', 'peru', 'darkviolet', 'hotpink', 'darkorange','rosybrown', 'sienna', 'steelblue',
          'navy', 'khaki', 'gold', 'royalblue', 'tan',
          'b', 'r', 'g', 'c', 'm', 'y', 'k', 'olive', 'orangered', 'indigo', 'crimson', 'grey', 'royalblue', 'khaki',
          'darkorange', 'slategray', 'deeppink', 'teal', 'peru', 'darkviolet', 'hotpink', 'darkorange','rosybrown', 'sienna', 'steelblue',
          'navy', 'khaki', 'gold', 'royalblue', 'tan']

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

directory = "output/FAME_MnFePSi/internal_voids_DDMC/results"  # Directory where the results are stored
inputs_file_name = "FAME_PSB_Dsp300um_B1400mT_L60mm_ff_vfl_Lvoid"  # File were the values of the input variables were defined. This file is supposed to be located in the directory defined above
case = 6  # Number of the case for which the thermodynamic cycles will be plotted
Reg_Length = 0.060  # [mm] This could be obtained from the inputs file in most cases, but it is preferred to leave it as a manual input to make more general

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text_file_input = './' + directory + '/' + str(case) + '.0' + inputs_file_name + '.txt'
inputs = importlib.import_module(directory.replace('/', '.')+'.'+inputs_file_name)

variable_1_resolution = inputs.vble1resolution
variable_2_resolution = inputs.vble2resolution
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

# Simulation parameters

Thot        = Thot_values[y]  # 310    # [K]
Tcold       = Thot - Tspan_values[x]  # 283   # [K]
nodes       = int(inputs.nodes / inputs.node_reduct_factor)
time_steps  = int(inputs.timesteps / inputs.timestep_reduct_factor)
mag_field_to_plot = inputs.max_app_field

# Materials per layer

materials_this_case = materials_per_config[c]  # TODO: generalize! This is particular for this batch of cases
species = list(s.strip('reg-') for s in materials_this_case)
materials = list(s.strip('reg-') for s in materials_this_case if s.startswith('reg'))
layer_interp_pos_this_case = layer_interp_pos_per_config[c]
# print(materials_this_case)

# Assigning a material to each node
nn = 0
int_discription = np.zeros(nodes+1, dtype=int)
species_descriptor = []

if species[nn].startswith('M'):
    node_min = [1]  # first nodes of each layer of MCM
else:
    node_min = []

node_max = []  # Last nodes of each layer of MCM
xloc = np.zeros(nodes+1)
DX = Reg_Length / (nodes - 1)

for i in range(nodes+1):  # sets 0->N-2 which is from 1st node with material to last node with material, so no ghosts
    xloc[i] = (DX * i + DX / 2)  # modify i so 0->N
    if xloc[i] >= layer_interp_pos_this_case[nn + 1] + DX and i < nodes:
        nn = nn + 1
        if species[nn].startswith('M'):
            node_min.append(i+1)  # The second node of the layer is actually taken as this is not the same species description used inside the loop
            if species[nn-1].startswith('M'):
                node_max.append(i-2)
        if species[nn].startswith('void'):
            node_max.append(i-2)
    int_discription[i] = nn
    species_descriptor.append(species[nn])
if species_descriptor[-1].startswith('M'):
    node_max.append(nodes)
node_mid = [int(i) for i in (np.array(node_min) + np.array(node_max))/2]
# print(species_descriptor)

# Getting temperature data if the input file contains more than just the temp of solid or fluid

myfile = open(text_file_input, "rt")
contents = myfile.read()
myfile.close()

fluidTemp = np.ones((time_steps+1, nodes+1))
solidTemp = np.ones((time_steps+1, nodes+1))
int_mag_field = np.ones((time_steps+1, int(nodes)+1))

for j in range(time_steps+1):
    # fluid T matrix must start in row 3
    fluidTemp[j] = [float(i) for i in ((contents.split('\n'))[3+j].split())]
    # Solid T matrix must start in row time_steps+6
    solidTemp[j] = [float(i) for i in ((contents.split('\n'))[time_steps+5+j].split())]
    int_mag_field[j] = [float(i) for i in ((contents.split('\n'))[(time_steps+1)*2+8+j].split())]

# Converting normalized temperatures to temperatures in [K]

fluidTemp = fluidTemp*(Thot-Tcold)+Tcold
solidTemp = solidTemp*(Thot-Tcold)+Tcold

index = 0
for mat in materials:

    S = np.loadtxt('./sourcefiles/new_mat/' + mat+'/' + mat+'_S_h.txt')
    i_mag_field = list(S[0, :]).index(mag_field_to_plot)

    plt.figure(5)
    plt.plot(S[1:, 0], S[1:, 1], color=colors[index], linestyle='solid')  # Heating low field
    # plt.plot(T_s_c, s_c[1:, 1], color=colors[index], linestyle='dashed')  # Cooling low field
    plt.plot(S[1:, 0], S[1:, i_mag_field], color=colors[index], linestyle='solid')  # Heating high field
    # plt.plot(T_s_c, s_c[1:, i_mag_field], color=colors[index], linestyle='dashdot')  # Cooling high field

    s_if = matS_h(S)

    entropy_min = np.zeros((time_steps + 1, len(materials)))
    entropy_mid = np.zeros((time_steps + 1, len(materials)))
    entropy_max = np.zeros((time_steps + 1, len(materials)))
    for n in range(time_steps + 1):
        # entropy[n, index] = s_if(solidTemp[n, node[index]], ap_field[n, node[index]])[0, 0]
        entropy_min[n, index] = s_if(solidTemp[n, node_min[index]], int_mag_field[n, int(node_min[index])])[0, 0]
        entropy_mid[n, index] = s_if(solidTemp[n, node_mid[index]], int_mag_field[n, int(node_mid[index])])[0, 0]
        entropy_max[n, index] = s_if(solidTemp[n, node_max[index]], int_mag_field[n, int(node_max[index])])[0, 0]

    plt.plot(solidTemp[:, node_min[index]], entropy_min[:, index], color=colors[index], marker='v', markersize=2, markerfacecolor='white')
    plt.plot(solidTemp[:, node_mid[index]], entropy_mid[:, index], color=colors[index], marker='o', markersize=2, markerfacecolor='white')
    plt.plot(solidTemp[:, node_max[index]], entropy_max[:, index], color=colors[index], marker='s', markersize=2, markerfacecolor='white')

    index = index + 1

plt.xlabel(r'$T_s$ [K]')
plt.ylabel(r'$s$ [J kg$^{-1}$ K$^{-1}$]')
# plt.vlines([293.608, 303.4515], 60, 120, colors=['k', 'k'])
# plt.hlines([75.0682, 104.551], 280, 315, colors=['k', 'k'])
plt.xlim(275, 315)
plt.ylim(70, 100)
plt.show()
