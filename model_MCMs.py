import numpy as np
import os
from tools.write_data_to_file import FileSaveMatrix

'''README: this script is useful for creating a series of materials starting from the properties of one material. The 
properties of the new materials are the same as in the original material but shifted to different transition 
temperatures'''

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Model material

S_model_mat_h = np.loadtxt('sourcefiles/new_mat/M2/M2_S_h.txt')
S_model_mat_c = np.loadtxt('sourcefiles/new_mat/M2/M2_S_c.txt')

Mag_model_mat_h = np.loadtxt('sourcefiles/new_mat/M2/M2_Mag_h.txt')
Mag_model_mat_c = np.loadtxt('sourcefiles/new_mat/M2/M2_Mag_c.txt')

Cp_model_mat_h = np.loadtxt('sourcefiles/new_mat/M2/M2_cp_h.txt')
Cp_model_mat_c = np.loadtxt('sourcefiles/new_mat/M2/M2_cp_c.txt')

# Information required for creating the new data sets with shifted Tc

dT_span = 27  # Difference in Tc of hottest and coldest layers
Tc_cold = 281  # Transition temperature of the coldest layer
T_offset = 16.2  # [K] Temperature offset of coldest layer w.r.t the original Tc of the model material
num_mat = 30  # Number of layers in the AMR
ini = 268  # Material numbering starts at
L = 60  # [m] Total length of AMR

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Define the shift in Tc of each layer with respect to the coldest one

'''Note: the relation of Tc as a function of the position of the layer in the AMR, funct => Tc = f(x), must be defined. 
This function can be linear or any other shape. In what follows a sigmoid function is presented.'''

dX = L/num_mat  # [mm] Length per layer

# Sigmoid function
lin = 5  # This parameter only has to do with the curvature of the sigmoid function
A = dT_span / ((1/(1+np.exp(-lin))) - (1/(1+np.exp(lin))))
B = 0.5*(L-dX)
funct = '{}/(1+np.exp(-({}/{})*(x-{}))) + {} - {}/(1+np.exp({}))'.format(A, lin, B, B+dX/2, Tc_cold, A, lin)

# Calculation of shift in Tc of each layer
x = np.linspace(dX/2, L-dX/2, num_mat)
y = eval(funct)
T_shift = y - Tc_cold

# The peak of the in-field Cp heating curve is taken as a reference Tc for the definition of the new materials

Cp_temp_vect_h = np.array(list(Cp_model_mat_h[1:, 0] - T_offset))
Cp_temp_vect_c = np.array(list(Cp_model_mat_c[1:, 0] - T_offset))

Mag_temp_vect_h = np.array(list(Mag_model_mat_h[1:, 0] - T_offset))
Mag_temp_vect_c = np.array(list(Mag_model_mat_c[1:, 0] - T_offset))

S_temp_vect_h = np.array(list(S_model_mat_h[1:, 0] - T_offset))
S_temp_vect_c = np.array(list(S_model_mat_c[1:, 0] - T_offset))

# Note: data in numpy array is first converted to a list and back to a numpy array so that the new variables are
# completely different entities. Otherwise, they would be linked, and when one changes so does the other.
hysteresis = "o"
while hysteresis != 'y' and hysteresis != 'n':
    hysteresis = input("Is thermal hysteresis to be included? y/n: ")
if hysteresis == 'n':
    heat_or_cool = 'something'
    while heat_or_cool != 'heating' and heat_or_cool != 'cooling':
        heat_or_cool = input("Do you want to reproduce heating or cooling curves? heating/cooling: ")

for i in range(num_mat):

    if not os.path.exists('sourcefiles/new_mat/M'+str(i + ini)):
        os.mkdir('sourcefiles/new_mat/M'+str(i + ini))
    Cp_model_mat_h[1:, 0] = Cp_temp_vect_h + T_shift[i]
    Cp_model_mat_c[1:, 0] = Cp_temp_vect_c + T_shift[i]
    Mag_model_mat_h[1:, 0] = Mag_temp_vect_h + T_shift[i]
    Mag_model_mat_c[1:, 0] = Mag_temp_vect_c + T_shift[i]
    S_model_mat_h[1:, 0] = S_temp_vect_h + T_shift[i]
    S_model_mat_c[1:, 0] = S_temp_vect_c + T_shift[i]

    Mx_cp_c  = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_cp_c.txt'
    Mx_cp_h  = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_cp_h.txt'
    Mx_Mag_c = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_Mag_c.txt'
    Mx_Mag_h = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_Mag_h.txt'
    Mx_S_c   = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_S_c.txt'
    Mx_S_h   = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_S_h.txt'

    if not os.path.exists(Mx_cp_c):
        if hysteresis == 'y':
            FileSaveMatrix(Mx_cp_c, Cp_model_mat_c)
        elif hysteresis == 'n' and heat_or_cool == 'heating':
            FileSaveMatrix(Mx_cp_c, Cp_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'cooling':
            FileSaveMatrix(Mx_cp_c, Cp_model_mat_c)
    else:
        print('File:' + Mx_cp_c + ', already exists. File not saved.')

    if not os.path.exists(Mx_cp_h):
        # FileSaveMatrix(Mx_cp_h, Cp_model_mat_h)
        if hysteresis == 'y':
            FileSaveMatrix(Mx_cp_h, Cp_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'heating':
            FileSaveMatrix(Mx_cp_h, Cp_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'cooling':
            FileSaveMatrix(Mx_cp_h, Cp_model_mat_c)
    else:
        print('File:' + Mx_cp_h + ', already exists. File not saved.')

    if not os.path.exists(Mx_Mag_c):
        # FileSaveMatrix(Mx_Mag_c, Mag_model_mat_c)
        if hysteresis == 'y':
            FileSaveMatrix(Mx_Mag_c, Mag_model_mat_c)
        elif hysteresis == 'n' and heat_or_cool == 'heating':
            FileSaveMatrix(Mx_Mag_c, Mag_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'cooling':
            FileSaveMatrix(Mx_Mag_c, Mag_model_mat_c)
    else:
        print('File:' + Mx_Mag_c + ', already exists. File not saved.')

    if not os.path.exists(Mx_Mag_h):
        # FileSaveMatrix(Mx_Mag_h, Mag_model_mat_h)
        if hysteresis == 'y':
            FileSaveMatrix(Mx_Mag_h, Mag_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'heating':
            FileSaveMatrix(Mx_Mag_h, Mag_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'cooling':
            FileSaveMatrix(Mx_Mag_h, Mag_model_mat_c)
    else:
        print('File:' + Mx_Mag_h + ', already exists. File not saved.')

    if not os.path.exists(Mx_S_c):
        # FileSaveMatrix(Mx_S_c, S_model_mat_c)
        if hysteresis == 'y':
            FileSaveMatrix(Mx_S_c, S_model_mat_c)
        elif hysteresis == 'n' and heat_or_cool == 'heating':
            FileSaveMatrix(Mx_S_c, S_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'cooling':
            FileSaveMatrix(Mx_S_c, S_model_mat_c)
    else:
        print('File:' + Mx_S_c + ', already exists. File not saved.')

    if not os.path.exists(Mx_S_h):
        # FileSaveMatrix(Mx_S_h, S_model_mat_h)
        if hysteresis == 'y':
            FileSaveMatrix(Mx_S_h, S_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'heating':
            FileSaveMatrix(Mx_S_h, S_model_mat_h)
        elif hysteresis == 'n' and heat_or_cool == 'cooling':
            FileSaveMatrix(Mx_S_h, S_model_mat_c)
    else:
        print('File:' + Mx_S_h + ', already exists. File not saved.')
