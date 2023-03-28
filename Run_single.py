from FAME_DP_V1 import runActive
from tools.write_data_to_file import FileSave, FileSaveMatrix, FileSaveVector
from tools.reduce_matrix import reduce_matrix
import numpy as np
import importlib
import pickle

# ---------------------------------- Calculation of just one case --------------------------------------

caseNumber    = 1

# Numerical parameters
nodes         = 100
node_reduct_factor = 1
timesteps     = 50
timestep_reduct_factor = 1
time_limit    = 8400  # [min] Time limit for the simulation in minutes
cycle_toler   = 1e-3  # Maximum cycle tolerance: criterion for ending the iterative calculation process
maxStepIter   = 2000  # Maximum time step iterations the simulation is allowed to take
maxCycleIter  = 2000  # Maximum cycle iterations the simulation is allowed to take
cen_loc       = 0
gain          = 0

# Simulation temperatures
Thot          = 308
Tcold         = 281
Tambset       = 288

# Frequency of AMR cycle
ff            = 2  # [Hz] frequency of AMR cycle

# Flow profile

# FAME cooler
dispV         = 2 * 16.667e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
acc_period    = 5
max_flow_per  = 45
full_magn_ang = 30
unbal_rat     = 1
from sourcefiles.device.FAME_V_flow import vol_flow_rate
volum_flow_profile = vol_flow_rate(timesteps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)

# POLO cooler
# dispV = 6.85e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
# from sourcefiles.device.polo_V_flow import polo_vol_flow
# volum_flow_profile = polo_vol_flow(timesteps, dispV, ff)

# Magnetic field profile

# FAME Cooler
from sourcefiles.device import FAME_app_field
mag_field = 1.4  # [T]
app_field = FAME_app_field.app_field(timesteps, nodes, mag_field)

# POLO cooler
# from sourcefiles.device.polo_mag_field import polo_app_field
# app_field = polo_app_field(timesteps, nodes, 0.1)

# Geometry of regenerator
'''Geometric parameters that differ from the configuration from case to case can be adjusted here. This way it is not
necessary to create a new configuration file each time a single parameter need to be changed. Take the following lines
starting with R8 as an example.'''

cName   = "PB"  # Name of file where the geometric configuration of the regenerator is defined
jName   = "Test_Gd_Qleak_disable"  # DP: use underlines to connect words because this is used as file name
num_reg = 1

dsp = 300e-6  # Note: this is particular for the packed bed and packed screen bed configuration
#msc = 2000  # Note: this is particular for the packed screen bed configuration

configuration = importlib.import_module('configurations.' + cName)

# Overwriting variables in the configuration file for this particular simulation

configuration.species_discription = ['reg-M0']
configuration.x_discription = [0, 0.0327418]
configuration.W_reg = 0.0824634
configuration.L_reg1 = 0.0327418
configuration.reduct_coeff = dict(M0=1, M122=1, M123=1, M124=1, M125=1, M166=0.77, M167=0.77, M168=0.77, M169=0.77,
                       M170=0.77, M171=0.77, M172=0.77, M173=0.77, M174=0.77, M175=0.77, M176=0.77, M177=0.77, M178=0.77, M179=0.77)
configuration.Dsp = dsp
#configuration.Msc = msc
#configuration.er = 1 - np.pi * dsp * msc**2 * np.sqrt(dsp**2 + msc**-2) / 4  # [] Porosity of the packed screen bed

# Switches for activating and deactivating terms in governing equations
CF   = 1
CS   = 1
CL   = 0
CVD  = 1
CMCE = 1

# Flow and Heat transfer models
htc_model_name = 'Macias_Machin_1991'  # Name of the file containing the function of the model for htc
leaks_model_name = 'flow_btw_plates'  # Name of the file containing the function of the model for heat leaks
pdrop_model_name = 'pb_ergun_1952'

results = runActive(caseNumber, Thot, Tcold, cen_loc, Tambset, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, cName,
                    jName, time_limit, cycle_toler, maxStepIter, maxCycleIter, volum_flow_profile, app_field,
                    htc_model_name, leaks_model_name, pdrop_model_name, num_reg, gain)

#  runActive():  returns
# Thot          0  |  sHalfBlow   12 |  sMaxCBlow   24 |  S_ht_cold     36 |
# Tcold         1  |  sEndBlow    13 |  sMaxHBlow   25 |  S_ht_fs       37 |
# qc            2  |  y           14 |  qh          26 |  S_vd          38 |
# qccor         3  |  s           15 |  cycleCount  27 |  S_condu_stat  39 |
# (t1-t0)/60    4  |  pt          16 |  int_field   28 |  S_condu_disp  40 |
# pave          5  |  np.max(pt)  17 |  htc_fs      29 |  S_ht_amb      41 |
# eff_HB_CE     6  |  Uti         18 |  fluid_dens  30 |  P_pump_AMR    42 |
# eff_CB_HE     7  |  freq        19 |  mass_flow   31 |  P_mag_AMR     43 |
# tFce          8  |  t           20 |  dP/dx       32 |  Q_leak        44 |
# tFhe          9  |  xloc        21 |  k_stat      33 |  Power_cold    45 |
# yHalfBlow     10 |  yMaxCBlow   22 |  k_disp      34 |  Power_hot     46 |
# yEndBlow      11 |  yMaxHBlow   23 |  S_ht_hot    35 |


fileName = "Gd_test_Qleak_disable.txt"
fileNameSave = './output/' + fileName
PickleFileName = "./pickleddata/{0:}-{1:d}".format(jName, int(caseNumber))

if len(results) > 10:

    FileSave(fileNameSave, "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n".format('Tspan [K]', 'Qh [W]', 'Qc [W]', 'Cycles [-]', 'Run time [min]', 'Max. Pressure drop [Pa]', 'Thot [K]', 'Tcold [K]', 'S_ht_hot [W/K]', 'S_ht_cold [W/K]', 'S_ht_fs [W/K]', 'S_vd [W/K]', 'S_condu_stat [W/K]', 'S_condu_disp [W/K]', 'S_ht_amb [W/K]', 'Pump_power_input [W]', 'Mag_power_input [W]', 'Q_leak [W]', 'Qc2 [W]', 'Qh2 [W]'))
    FileSave(fileNameSave, "{},{:7.6f},{:7.6f},{},{:7.6f},{:7.6f},{},{},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f} \n".format(results[0]-results[1], results[26], results[2], results[27], results[4], results[17], Thot, Tcold, results[35], results[36], results[37], results[38], results[39], results[40], results[41], results[42], results[43], results[44], results[45], results[46]))
    FileSave(fileNameSave, "Fluid temperatures\n")
    FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[14], node_reduct_factor, timestep_reduct_factor))
    FileSave(fileNameSave, "Solid temperatures\n")
    FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[15], node_reduct_factor, timestep_reduct_factor))
    FileSave(fileNameSave, "Pressure drop accross the regenerator for the entire cycle\n")
    FileSaveVector(fileNameSave, results[16])
    FileSave(fileNameSave, "\nInternal Magnetic Field\n")
    FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[28], node_reduct_factor, timestep_reduct_factor))
    FileSave(fileNameSave, "\nHeat transfer coefficient between solid and fluid in the packed bed\n")
    FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[29], node_reduct_factor, timestep_reduct_factor))
    FileSave(fileNameSave, "\nMass flow rate\n")
    FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[31], node_reduct_factor, timestep_reduct_factor))
    FileSave(fileNameSave, "\nPressure drop per unit length\n")
    FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[32], node_reduct_factor, timestep_reduct_factor))
    FileSave(fileNameSave, "\nEffective thermal conductivity of solid\n")
    FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[33], node_reduct_factor, timestep_reduct_factor))
    FileSave(fileNameSave, "\nEffective thermal conductivity of fluid\n")
    FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[34], node_reduct_factor, timestep_reduct_factor))

else:  # Save Pickle data

    aaa = (results[0], results[1], results[2], results[3], results[4])
    fileObject = open(PickleFileName, 'wb')  # open the file for writing
    pickle.dump(aaa, fileObject)  # this writes the object aaa to the file named 'PickleFileName'
    fileObject.close()  # here we close the fileObject