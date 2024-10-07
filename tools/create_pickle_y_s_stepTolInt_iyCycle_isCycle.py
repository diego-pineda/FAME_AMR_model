import numpy as np
from scipy.interpolate import CubicSpline
import pickle
import importlib

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

Note: for running in Python console just select all, right click, and run in Python console. For running with the run 
button make sure there are two dots at the beginning of the variables old_directory and 
'''

# ------------ Inputs regarding the old simulation ------------

old_directory = "output/FAME_MnFePSi/FAME_Dsp300um_B_1400mT_layering/sensitivity_num_params/ff4500mHz"
old_cases = [int(i) for i in np.linspace(0, 99, 100)]
old_inputs_file_name = 'Linear_4500mHz_N_nt_vflow'
# old_nodes = 1500
# old_time_steps = 100
# old_nodes_reduct_fact = 1
old_time_steps_reduct_factor = 1

inputs = importlib.import_module(old_directory.replace('/', '.')+'.'+old_inputs_file_name)

variable_1_values = inputs.vble1values
variable_1_resolution = inputs.vble1resolution
variable_2_values = inputs.vble2values  # [units] Variable name. Note: values used for variable 2 in the cases simulated
variable_2_resolution = inputs.vble2resolution
variable_3_values = inputs.vble3values  # [units] Variable name. Note: values used for variable 2 in the cases simulated
variable_3_resolution = inputs.vble3resolution
node_reduct_factors = inputs.node_reduct_factors
# timestep_reduc_factors = inputs.timestep_reduc_factors
hot_resolution = inputs.hotResolution
span_resolution = inputs.TspanResolution
# Thotarr = inputs.Thotarr
# Tspanarr = inputs.Tspanarr

# ------------- Inputs regarding the new simulation ---------------
# nodes = 1500
# time_steps = 100
stepTolInt = 6
new_case_numbers = [int(i) for i in np.linspace(0, 99, 100)]
new_inputs_file_name = 'Linear_4500mHz_Thot310K_Tspan27K_gain2_N_nt_vflow'

# TODO: when this was set to zero it run many cycles: 64 in Delftblue and 99 locally (why the difference).
#  Now testing what happens if this is set to 6. In case this does not really applies remove the warning above.
i = 0
for old_case in old_cases:

    casegroup = int(np.floor(old_case / (span_resolution * hot_resolution)))

    a = int(np.floor((casegroup - variable_1_resolution * int(np.floor(casegroup / variable_1_resolution))) / 1))
    b = int(np.floor((casegroup - variable_1_resolution * variable_2_resolution * int(np.floor(casegroup / (variable_1_resolution * variable_2_resolution)))) / variable_1_resolution))
    c = int(np.floor((casegroup - variable_1_resolution * variable_2_resolution * variable_3_resolution * int(np.floor(casegroup / (variable_1_resolution * variable_2_resolution * variable_3_resolution)))) / (variable_1_resolution * variable_2_resolution)))
    # y = int(np.floor(old_case / span_resolution) % hot_resolution)
    # x = old_case % span_resolution

    old_nodes = int(variable_1_values[a])
    old_nodes_reduct_fact = int(node_reduct_factors[a])
    old_time_steps = int(variable_2_values[b])
    old_time_steps_reduct_factor = 1  # int(timestep_reduc_factors[b])

    nodes = int(variable_1_values[a])
    time_steps = int(variable_2_values[b])

    PickleFileName = "../pickleddata/{0:}-{1:d}".format(new_inputs_file_name, new_case_numbers[i])
    # Getting the temperature data from the text file
    data_file = '../' + old_directory + '/' + str(old_case) + '.0' + old_inputs_file_name + '.txt'
    myfile = open(data_file, "rt")
    contents = myfile.read()
    myfile.close()
    print(old_case, new_case_numbers[i])
    # New attempt: grab only last time step temperatures of existing results and use that to populate new matrices

    y1_old = [float(i) for i in ((contents.split('\n'))[3+int(old_time_steps/old_time_steps_reduct_factor)].split())]
    s1_old = [float(i) for i in ((contents.split('\n'))[1 * int(old_time_steps/old_time_steps_reduct_factor) + 5 + int(old_time_steps/old_time_steps_reduct_factor)].split())]

    y1_new = np.interp(np.linspace(0, 1, nodes+1), np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), y1_old)
    s1_new = np.interp(np.linspace(0, 1, nodes+1), np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), s1_old)

    y = np.ones((time_steps+1, nodes+1)) * y1_new
    s = np.ones((time_steps+1, nodes+1)) * s1_new
    iyCycle = np.copy(y)
    isCycle = np.copy(s)

    aaa = (y, s, stepTolInt, iyCycle, isCycle)
    fileObject = open(PickleFileName, 'wb')  # open the file for writing
    pickle.dump(aaa, fileObject)  # this writes the object aaa to the file named 'PickleFileName'
    fileObject.close()  # here we close the fileObject

    i = i+1

    # --------------- First attempt which populates last and previous cycles with same temperature data ----------------

    # Tf_old = np.ones((int(old_time_steps/old_time_steps_reduct_factor)+1, int(old_nodes/old_nodes_reduct_fact)+1))
    # Ts_old = np.ones((int(old_time_steps/old_time_steps_reduct_factor)+1, int(old_nodes/old_nodes_reduct_fact)+1))
    #
    # for j in range(int(old_time_steps/old_time_steps_reduct_factor)+1):
    #     Tf_old[j] = [float(i) for i in ((contents.split('\n'))[3+j].split())]
    #     Ts_old[j] = [float(i) for i in ((contents.split('\n'))[1 * time_steps+5 +j].split())]
    #
    # # Interpolation of the data to obtain the data for points in the new grid
    # Tf = np.ones((int(old_time_steps/old_time_steps_reduct_factor)+1, nodes+1))
    # Ts = np.ones((int(old_time_steps/old_time_steps_reduct_factor)+1, nodes+1))
    # Tf_new = np.ones((time_steps+1, nodes+1))
    # Ts_new = np.ones((time_steps+1, nodes+1))
    #
    # # Using CubicSpline interpolation
    # # if nodes == old_nodes and time_steps == old_time_steps and old_time_steps_reduct_factor == 1 and old_nodes_reduct_fact == 1:
    # #     Tf = Tf_old
    # #     Ts = Ts_old
    # # else:
    # #     for j in range(int(old_time_steps/old_time_steps_reduct_factor)+1):
    # #         spl_f = CubicSpline(np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), Tf_old[j])
    # #         spl_s = CubicSpline(np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), Ts_old[j])
    # #         Tf[j] = spl_f(np.linspace(0, 1, nodes+1))
    # #         Ts[j] = spl_s(np.linspace(0, 1, nodes+1))
    # #     for i in range(nodes+1):
    # #         spl_f = CubicSpline(np.linspace(0, 1, int(old_time_steps/old_time_steps_reduct_factor)+1), Tf[:, i])
    # #         spl_s = CubicSpline(np.linspace(0, 1, int(old_time_steps/old_time_steps_reduct_factor)+1), Ts[:, i])
    # #         Tf_new[:, i] = spl_f(np.linspace(0, 1, time_steps+1))
    # #         Ts_new[:, i] = spl_s(np.linspace(0, 1, time_steps+1))
    #
    # # Using linear piecewise interpolation
    # if nodes == old_nodes and time_steps == old_time_steps and old_time_steps_reduct_factor == 1 and old_nodes_reduct_fact == 1:
    #     Tf = Tf_old
    #     Ts = Ts_old
    # else:
    #     for j in range(int(old_time_steps/old_time_steps_reduct_factor)+1):
    #         Tf[j] = np.interp(np.linspace(0, 1, nodes+1), np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), Tf_old[j])
    #         Ts[j] = np.interp(np.linspace(0, 1, nodes+1), np.linspace(0, 1, int(old_nodes/old_nodes_reduct_fact)+1), Ts_old[j])
    #     for i in range(nodes+1):
    #         Tf_new[:, i] = np.interp(np.linspace(0, 1, time_steps+1), np.linspace(0, 1, int(old_time_steps/old_time_steps_reduct_factor)+1), Tf[:, i])
    #         Ts_new[:, i] = np.interp(np.linspace(0, 1, time_steps+1), np.linspace(0, 1, int(old_time_steps/old_time_steps_reduct_factor)+1), Ts[:, i])
    #
    # aaa = (Tf_new, Ts_new, stepTolInt, Tf_new, Ts_new)
    # fileObject = open(PickleFileName, 'wb')  # open the file for writing
    # pickle.dump(aaa, fileObject)  # this writes the object aaa to the file named 'PickleFileName'
    # fileObject.close()  # here we close the fileObject