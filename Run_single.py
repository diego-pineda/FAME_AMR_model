from FAME_DP_V1 import runActive
import numpy as np

# ---------------------------------- Calculation of just one case --------------------------------------

#runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName,time_limit,cycle_toler,maxStepIter,maxCycleIter)

caseNumber    = 1

# Numerical parameters
nodes         = 1800
timesteps     = 400
time_limit    = 600  # [min] Time limit for the simulation in minutes
cycle_toler   = 1e-4  # Maximum cycle tolerance: criterion for ending the iterative calculation process
maxStepIter   = 2000  # Maximum time step iterations the simulation is allowed to take
maxCycleIter  = 1000  # Maximum cycle iterations the simulation is allowed to take
cen_loc       = 0

# Simulation temperatures
Thot          = 307
Tcold         = 280
Tambset       = 288

# Frequency of AMR cycle
ff            = 1  # [Hz] frequency of AMR cycle

# Flow profile

# FAME cooler
dispV         = 2.5 * 16.667e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
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
from configurations import R8
R8.species_discription = ['reg-M166', 'reg-M167', 'reg-M168', 'reg-M169', 'reg-M170', 'reg-M171', 'reg-M172', 'reg-M173', 'reg-M174', 'reg-M175', 'reg-M176', 'reg-M177', 'reg-M178', 'reg-M179']
R8.x_discription = [0, 0.00428571, 0.00857143, 0.01285714, 0.01714286, 0.02142857, 0.02571429, 0.03, 0.03428571, 0.03857143, 0.04285714, 0.04714286, 0.05142857, 0.05571429, 0.06]
R8.reduct_coeff = dict(M166=0.77, M167=0.77, M168=0.77, M169=0.77,
                       M170=0.77, M171=0.77, M172=0.77, M173=0.77, M174=0.77, M175=0.77, M176=0.77, M177=0.77, M178=0.77, M179=0.77)
R8.mK = 6
R8.mRho = 6100
cName   = "R8"  # Name of file where the geometric configuration of the regenerator is defined
jName   = "Test_change_cp_calc"  # DP: use underlines to connect words because this is used as file name
num_reg = 1

# Switches for activating and deactivating terms in governing equations
CF   = 1
CS   = 1
CL   = 1
CVD  = 1
CMCE = 1

# Heat transfer models
htc_model_name = 'Macias_Machin_1991'  # Name of the file containing the function of the model for htc
leaks_model_name = 'flow_btw_plates'  # Name of the file containing the function of the model for heat leaks


results = runActive(caseNumber, Thot, Tcold, cen_loc, Tambset, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, cName,
                    jName, time_limit, cycle_toler, maxStepIter, maxCycleIter, volum_flow_profile, app_field,
                    htc_model_name, leaks_model_name, num_reg)


# ------------------ Some useful functions for storing data --------------------
def FileSave(filename, content):
    with open(filename, "a") as myfile:
        myfile.write(content)


def FileSaveMatrix(filename, content):
    with open(filename, "a") as f:
        for line in content:
            f.write(" ".join("{:9.6f}\t".format(x) for x in line))
            f.write("\n")


def FileSaveVector(filename, content):
    with open(filename, "a") as f:
        f.write(" ".join("{:9.6f}\t".format(x) for x in content))
        f.write("\n")

# ------------------- Function for reducing the size of matrices --------------------

def reduce_matrix(nodes, timesteps, matrix, fraction_nodes, fraction_timesteps):
    new_matrix = np.ones((int(timesteps / fraction_timesteps)+1, int(nodes / fraction_nodes)+1))
    ii = 0
    for i in range(timesteps+1):
        jj = 0
        if i % fraction_timesteps == 0:
            for j in range(nodes+1):
                if j % fraction_nodes == 0:
                    new_matrix[ii, jj] = matrix[i, j]
                    jj = jj + 1
            ii = ii + 1
    return new_matrix


#  runActive():  returns
# Thot          0   eff_HB_CE   6   sHalfBlow   12  Uti         18  sMaxCBlow   24  fluid_dens  30
# Tcold         1   eff_CB_HE   7   sEndBlow    13  freq        19  sMaxHBlow   25  mass_flow   31
# qc            2   tFce        8   y           14  t           20  qh          26
# qccor         3   tFhe        8   s           15  xloc        21  cycleCount  27
# (t1-t0)/60    4   yHalfBlow   10  pt          16  yMaxCBlow   22  int_field   28
# pave          5   yEndBlow    11  np.max(pt)  17  yMaxHBlow   23  htc_fs      29


fileName = "Test_change_cp_calc.txt"
fileNameSave = './output/' + fileName
#FileSave(fileNameSave,"{},{},{},{},{},{},{} \n".format(results[0], results[1], results[2], results[3], results[4], results[5],results[26]))
FileSave(fileNameSave, "{},{},{},{},{},{} \n".format('Tspan [K]', 'Qh [W]', 'Qc [W]', 'Cycles [-]', 'run time [min]', 'Max. Pressure drop [Pa]'))
FileSave(fileNameSave, "{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f} \n".format(results[0]-results[1], results[26], results[2], results[27], results[4], results[17]))
FileSave(fileNameSave, "Fluid temperatures\n")
FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[14], 3, 2))
#FileSave(fileNameSave, "\n")
FileSave(fileNameSave, "Solid temperatures\n")
FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[15], 3, 2))
#FileSave(fileNameSave, "\n")
FileSave(fileNameSave, "Pressure drop accross the regenerator for the entire cycle\n")
FileSaveVector(fileNameSave, results[16])
FileSave(fileNameSave, "\nInternal Magnetic Field\n")
FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[28], 3, 2))
FileSave(fileNameSave, "\nHeat transfer coefficient between solid and fluid in the packed bed\n")
FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[29], 3, 2))
FileSave(fileNameSave, "\nMass flow rate\n")
FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[31], 3, 2))

#FileSave(fileNameSave, "\n")


# fileNameSliceTemp = './Blow/{:3.0f}-{:3.0f}-BlowSlice'.format(Thot, Tcold) + fileName
# FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]', 'Uti [-]', 'freq [Hz]', 'run time [min]'))
# FileSave(fileNameSliceTemp,"{},{},{:4.2f},{},{:4.2f} \n".format(results[0], results[1], results[18], results[19], results[4]))
# BlowSliceTemperatures = np.stack((results[21], results[10], results[11], results[12], results[13], results[22], results[23], results[24], results[25]), axis=-1)
# FileSaveMatrix(fileNameSliceTemp, BlowSliceTemperatures)

# fluidtemperature = './' + "Fluid_Temperature2.txt"
# fluidtemperatures = results[14]
# FileSaveMatrix(fluidtemperature,fluidtemperatures)
#
# solidtemperature = './' + "Solid_Temperature2.txt"
# solidtemperatures = results[15]
# FileSaveMatrix(solidtemperature,solidtemperatures)