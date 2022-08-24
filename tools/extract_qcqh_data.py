import numpy as np
import os
import sys
import importlib

# directory = "../../output/FAME_20layer_infl_Thot_flow"
directory = "output/FAME_GD/FAME_Dsp300um_B1400mT_Gd_ff_vfl" #"output/FAME_MnFePSi/FAME_Dsp300um_B1400mT_ff_vfl4" # "output/FAME_GD/FAME_Dsp300um_B1400mT_num_layers_Tspan_Gdlike"  # "output/FAME_20layer_nodes_sens/nodes_sens4"  # "output/FAME_20layer_cteTcold/cteTcold2"  # "output/FAME_20layer_timesteps_sens/timesteps_sens3"
# directory = "output/FAME_20layer_Dsp"
# inputs_file_name = 'FAME_20layer_infl_Thot_flow'  # File were the values of the input variables were defined.
inputs_file_name = "FAME_Dsp300um_B1400mT_Gd_ff_vfl" #"FAME_Dsp300um_B1400mT_num_layers_Tspan_Gdlike"  # "FAME_20layer_nodes_sens4"  # "Run_parallel" #"FAME_20layer_cteTcold2"  #"FAME_20layer_timesteps_sens3" #"FAME_20layer_cteTcold"  #

# inputs = importlib.import_module(directory.replace('/', '.').replace('.', '', 6)+'.'+inputs_file_name)
# inputs = importlib.import_module(directory.replace('/', '.')+'.'+inputs_file_name)


def extract_qc_qh_data(directory, inputs_file_name):

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

    Thot = inputs.Thotarr
    Tspan = inputs.Tspanarr

    legends = []
    legends2 = []

    maxcase = inputs.maxcase

    Qc = np.ones((variable_3_resolution, variable_2_resolution, variable_1_resolution, hot_resolution, span_resolution))
    Qh = np.ones((variable_3_resolution, variable_2_resolution, variable_1_resolution, hot_resolution, span_resolution))

    results = np.ones((maxcase, 3))  # This is for making a table of | case | Qc | Qh |
    i = 0
    for files in os.listdir(directory):  # Goes over all files in the directory

        if '.txt' in files:
            if 'index' in files:
                continue
            # case = int(files.split('-')[1].split('.')[0])
            case = int(files.split('.')[0])
            results[i, 0] = case
            casegroup = int(np.floor(case / (span_resolution * hot_resolution)))

            a = int(np.floor((casegroup - variable_1_resolution * int(np.floor(casegroup / variable_1_resolution))) / 1))
            b = int(np.floor((casegroup - variable_1_resolution * variable_2_resolution * int(np.floor(casegroup / (variable_1_resolution * variable_2_resolution)))) / variable_1_resolution))
            c = int(np.floor((casegroup - variable_1_resolution * variable_2_resolution * variable_3_resolution * int(np.floor(casegroup / (variable_1_resolution * variable_2_resolution * variable_3_resolution)))) / (variable_1_resolution * variable_2_resolution)))
            y = int(np.floor(case / span_resolution) % hot_resolution)
            x = case % span_resolution
            # print(case, z, x, y)
            myfile = open(directory + '/' + files, "rt")
            contents = myfile.read()
            myfile.close()
            Qc[c, b, a, y, x] = float(((contents.split('\n'))[1].split(','))[2])
            Qh[c, b, a, y, x] = float(((contents.split('\n'))[1].split(','))[1])
            results[i, 1] = float(((contents.split('\n'))[1].split(','))[2])
            results[i, 2] = float(((contents.split('\n'))[1].split(','))[1])
            i = i + 1

    results = np.delete(results, slice(i, maxcase, 1), 0)
    print(results[np.argsort(results[:, 0])])

    np.save(directory+'/'+inputs_file_name+"_Qc.npy", Qc, allow_pickle=True, fix_imports=True)
    np.save(directory+'/'+inputs_file_name+"_Qh.npy", Qh, allow_pickle=True, fix_imports=True)

    # np.save("output/FAME_Dsp300um_B1400mT_ff_vfl/FAME_Dsp300um_B1400mT_ff_vfl_Qc.npy", Qc, allow_pickle=True, fix_imports=True)
    # np.save("output/FAME_Dsp300um_B1400mT_ff_vfl/FAME_Dsp300um_B1400mT_ff_vfl_Qh.npy", Qh, allow_pickle=True, fix_imports=True)
    # np.save("output/FAME_Dsp300um_B1400mT_ff_vfl/FAME_Dsp300um_B1400mT_ff_vfl_Qc.npy", Qc, allow_pickle=True, fix_imports=True)
