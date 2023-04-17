import numpy as np
from scipy.interpolate import CubicSpline
import pickle

'''README
This script creates a pickle file containing 5 elements: y, s, stepTolInt, iyCycle, and isCycle based on the results of
a previously completed simulation. 

If the number of nodes and time steps of the new simulation do not coincide with those of the previous simulation, the
missing data will be created by interpolation.

The pickle file is saved with the proper name for running a new simulation based on your entries. So be careful with the
consistency of name that you assign here to the pickle file and the name you assign to the inputs file of the new 
simulation.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WARNING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
The stepTolInt used here should be equal to the absolute value of the exponent of the variable cycle_toler (that was 
used in the simulation whose results are going to be used as initial condition for the new simulation) plus 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

# Inputs regarding the old simulation
old_directory = "./output/FAME_MnFePSi/FAME_Dsp300um_B_1400mT_layering/sensitivity_num_params/ff500mHz"
old_case = 19
old_inputs_file_name = 'Linear_500mHz_N_nt_vflow'
old_nodes = 6000
old_time_steps = 800
old_nodes_reduct_fact = 3
old_time_steps_reduct_factor = 1

# Inputs regarding the new simulation
nodes = 6000
time_steps = 800
stepTolInt = 6
new_case_number = 19
new_inputs_file_name = 'Linear_500mHz_N_nt_vflow_after_calc_correction'
PickleFileName = "./pickleddata/{0:}-{1:d}".format(new_inputs_file_name, int(new_case_number))
# TODO: not so sure that I should do it like this. Maybe it is wise to set this to zero so than it runs a few cycles. In case this does not really applies remove the warning above.

# Getting the temperature data from the text file
data_file = old_directory + '/' + str(old_case) + '.0' + old_inputs_file_name + '.txt'
myfile = open(data_file, "rt")
contents = myfile.read()
myfile.close()

Tf_old = np.ones((int(old_time_steps/old_time_steps_reduct_factor)+1, int(old_nodes/old_nodes_reduct_fact)+1))
Ts_old = np.ones((int(old_time_steps/old_time_steps_reduct_factor)+1, int(old_nodes/old_nodes_reduct_fact)+1))

for j in range(int(old_time_steps/old_time_steps_reduct_factor)+1):
    Tf_old[j] = [float(i) for i in ((contents.split('\n'))[3+j].split())]
    Ts_old[j] = [float(i) for i in ((contents.split('\n'))[1 * time_steps+5 +j].split())]

# Interpolation of the data to obtain the data for points in the new grid
Tf = np.ones((int(old_time_steps/old_time_steps_reduct_factor)+1, nodes+1))
Ts = np.ones((int(old_time_steps/old_time_steps_reduct_factor)+1, nodes+1))
Tf_new = np.ones((time_steps+1, nodes+1))
Ts_new = np.ones((time_steps+1, nodes+1))

# Using CubicSpline interpolation
# if nodes == old_nodes and time_steps == old_time_steps and old_time_steps_reduct_factor == 1 and old_nodes_reduct_fact == 1:
#     Tf = Tf_old
#     Ts = Ts_old
# else:
#     for j in range(int(old_time_steps/old_time_steps_reduct_factor)+1):
#         spl_f = CubicSpline(np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), Tf_old[j])
#         spl_s = CubicSpline(np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), Ts_old[j])
#         Tf[j] = spl_f(np.linspace(0, 1, nodes+1))
#         Ts[j] = spl_s(np.linspace(0, 1, nodes+1))
#     for i in range(nodes+1):
#         spl_f = CubicSpline(np.linspace(0, 1, int(old_time_steps/old_time_steps_reduct_factor)+1), Tf[:, i])
#         spl_s = CubicSpline(np.linspace(0, 1, int(old_time_steps/old_time_steps_reduct_factor)+1), Ts[:, i])
#         Tf_new[:, i] = spl_f(np.linspace(0, 1, time_steps+1))
#         Ts_new[:, i] = spl_s(np.linspace(0, 1, time_steps+1))

# Using linear piecewise interpolation
if nodes == old_nodes and time_steps == old_time_steps and old_time_steps_reduct_factor == 1 and old_nodes_reduct_fact == 1:
    Tf = Tf_old
    Ts = Ts_old
else:
    for j in range(int(old_time_steps/old_time_steps_reduct_factor)+1):
        Tf[j] = np.interp(np.linspace(0, 1, nodes+1), np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), Tf_old[j])
        Ts[j] = np.interp(np.linspace(0, 1, nodes+1), np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), Ts_old[j])
    for i in range(nodes+1):
        Tf_new[:, i] = np.interp(np.linspace(0, 1, time_steps+1), np.linspace(0, 1, int(old_time_steps/old_time_steps_reduct_factor)+1), Tf[:, i])
        Ts_new[:, i] = np.interp(np.linspace(0, 1, time_steps+1), np.linspace(0, 1, int(old_time_steps/old_time_steps_reduct_factor)+1), Ts[:, i])

aaa = (Tf_new, Ts_new, stepTolInt, Tf_new, Ts_new)
fileObject = open(PickleFileName, 'wb')  # open the file for writing
pickle.dump(aaa, fileObject)  # this writes the object aaa to the file named 'PickleFileName'
fileObject.close()  # here we close the fileObject