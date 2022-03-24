import numpy as np
import os

'''README: this script is useful for creating a series of materials starting from the properties of one material. The 
properties of the new materials are the same as in the original material but shifted to different transition 
temperatures'''


def FileSaveMatrix(filename, content):
    with open(filename, "a") as f:
        for line in content:
            f.write(" ".join("{:9.6f}\t".format(x) for x in line))
            f.write("\n")


# Model material
# TODO: add cooling properties so that materials with hysteresis can be also created by using this file
S_model_mat = np.loadtxt('sourcefiles/mat/ssi2/M2_S_Si_05_AT_G_Q20_h.txt')
Mag_model_mat = np.loadtxt('sourcefiles/mat/ssi2/M2_Mag_Si_05_AT_G_Q20_h.txt')
Cp_model_mat = np.loadtxt('sourcefiles/mat/ssi2/M2_cp_Si_05_AT_G_Q20_h.txt')

T_offset = 16.8  # [K] Temperature offset of coldest layer w.r.t the original Tc of the model material
num_mat = 4
T_shift = 9  # [K] Separation in Tc between adjacent layers
ini = 122  # Material numbering starts at


Cp_temp_vect = np.array(list(Cp_model_mat[1:, 0] - T_offset))
Mag_temp_vect = np.array(list(Mag_model_mat[1:, 0] - T_offset))
S_temp_vect = np.array(list(S_model_mat[1:, 0] - T_offset))

# Note: data in numpy array is first converted to a list and back to a numpy array so that the new variables are
# completely different entities. Otherwise, they would be linked, and when one changes so does the other.


for i in range(num_mat):

    if not os.path.exists('sourcefiles/new_mat/M'+str(i + ini)):
        os.mkdir('sourcefiles/new_mat/M'+str(i + ini))
    Cp_model_mat[1:, 0] = Cp_temp_vect + T_shift * (i)
    Mag_model_mat[1:, 0] = Mag_temp_vect + T_shift * (i)
    S_model_mat[1:, 0] = S_temp_vect + T_shift * (i)

    Mx_cp_c  = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_cp_c.txt'
    Mx_cp_h  = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_cp_h.txt'
    Mx_Mag_c = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_Mag_c.txt'
    Mx_Mag_h = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_Mag_h.txt'
    Mx_S_c   = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_S_c.txt'
    Mx_S_h   = 'sourcefiles/new_mat/M'+str(i + ini)+'/M'+str(i + ini)+'_S_h.txt'

    if not os.path.exists(Mx_cp_c):
        FileSaveMatrix(Mx_cp_c, Cp_model_mat)
    else:
        print('File:' + Mx_cp_c + ', already exists. File not saved.')

    if not os.path.exists(Mx_cp_h):
        FileSaveMatrix(Mx_cp_h, Cp_model_mat)
    else:
        print('File:' + Mx_cp_h + ', already exists. File not saved.')

    if not os.path.exists(Mx_Mag_c):
        FileSaveMatrix(Mx_Mag_c, Mag_model_mat)
    else:
        print('File:' + Mx_Mag_c + ', already exists. File not saved.')

    if not os.path.exists(Mx_Mag_h):
        FileSaveMatrix(Mx_Mag_h, Mag_model_mat)
    else:
        print('File:' + Mx_Mag_h + ', already exists. File not saved.')

    if not os.path.exists(Mx_S_c):
        FileSaveMatrix(Mx_S_c, S_model_mat)
    else:
        print('File:' + Mx_S_c + ', already exists. File not saved.')

    if not os.path.exists(Mx_S_h):
        FileSaveMatrix(Mx_S_h, S_model_mat)
    else:
        print('File:' + Mx_S_h + ', already exists. File not saved.')


