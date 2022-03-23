from FAME_DP_V1 import runActive
import numpy as np

# ---------------------------------- Calculation of just one case --------------------------------------

#runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName,time_limit,cycle_toler,maxStepIter,maxCycleIter)

caseNumber    = 1

# Numerical parameters
nodes         = 300
timesteps     = 400
time_limit    = 600  # [min] Time limit for the simulation in minutes
cycle_toler   = 1e-1  # Maximum cycle tolerance: criterion for ending the iterative calculation process
maxStepIter   = 2000  # Maximum time step iterations the simulation is allowed to take
maxCycleIter  = 1000  # Maximum cycle iterations the simulation is allowed to take
cen_loc       = 0

# Simulation temperatures
Thot          = 312
Tcold         = 298
Tambset       = 300

# Frequency of AMR cycle
ff            = 1  # [Hz] frequency of AMR cycle

# Flow profile

# FAME cooler
dispV         = 1.1 * 16.667e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
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
R8.species_discription = ['reg-M6', 'reg-M7', 'reg-M8', 'reg-M9', 'reg-M10', 'reg-M11', 'reg-M12', 'reg-M13', 'reg-M14', 'reg-M15']
R8.x_discription = [0, 0.006, 0.012, 0.018, 0.024, 0.030, 0.036, 0.042, 0.048, 0.054, 0.060]
R8.reduct_coeff = dict(M0=1, M1=0.55, M2=0.77, M6=1, M7=1, M8=1, M9=1, M10=1, M11=1, M12=1, M13=1, M14=1, M15=1)
R8.mK = 6
R8.mRho = 6100
cName   = "R8"  # Name of file where the geometric configuration of the regenerator is defined
jName   = "Test_write_new_matrices"  # DP: use underlines to connect words because this is used as file name
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


# Some useful functions for storing data.
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

#  runActive():  returns
# Thot          0   eff_HB_CE   6   sHalfBlow   12  Uti         18  sMaxCBlow   24  fluid_dens  30
# Tcold         1   eff_CB_HE   7   sEndBlow    13  freq        19  sMaxHBlow   25  mass_flow   31
# qc            2   tFce        8   y           14  t           20  qh          26
# qccor         3   tFhe        8   s           15  xloc        21  cycleCount  27
# (t1-t0)/60    4   yHalfBlow   10  pt          16  yMaxCBlow   22  int_field   28
# pave          5   yEndBlow    11  np.max(pt)  17  yMaxHBlow   23  htc_fs      29


fileName = "Test_write_new_matrices.txt"
fileNameSave = './output/' + fileName
#FileSave(fileNameSave,"{},{},{},{},{},{},{} \n".format(results[0], results[1], results[2], results[3], results[4], results[5],results[26]))
FileSave(fileNameSave, "{},{},{},{},{},{} \n".format('Tspan [K]', 'Qh [W]', 'Qc [W]', 'Cycles [-]', 'run time [min]', 'Max. Pressure drop [Pa]'))
FileSave(fileNameSave, "{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f} \n".format(results[0]-results[1], results[26], results[2], results[27], results[4], results[17]))
FileSave(fileNameSave, "Fluid temperatures\n")
FileSaveMatrix(fileNameSave, results[14])
#FileSave(fileNameSave, "\n")
FileSave(fileNameSave, "Solid temperatures\n")
FileSaveMatrix(fileNameSave, results[15])
#FileSave(fileNameSave, "\n")
FileSave(fileNameSave, "Pressure drop accross the regenerator for the entire cycle\n")
FileSaveVector(fileNameSave, results[16])
FileSave(fileNameSave, "\nInternal Magnetic Field\n")
FileSaveMatrix(fileNameSave, results[28])
FileSave(fileNameSave, "\nHeat transfer coefficient between solid and fluid in the packed bed\n")
FileSaveMatrix(fileNameSave, results[29])
FileSave(fileNameSave, "\nMass flow rate\n")
FileSaveMatrix(fileNameSave, results[31])

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