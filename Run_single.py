from FAME_DP_V1 import runActive
import numpy as np


# ---------------------------------- Calculation of just one case --------------------------------------

#runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName,time_limit,cycle_toler,maxStepIter,maxCycleIter)

caseNumber    = 2

# Numerical parameters
nodes         = 300
timesteps     = 400
time_limit    = 600  # [min] Time limit for the simulation in minutes
cycle_toler   = 1e-3  # Maximum cycle tolerance: criterion for ending the iterative calculation process
maxStepIter   = 500  # Maximum time step iterations the simulation is allowed to take
maxCycleIter  = 500  # Maximum cycle iterations the simulation is allowed to take
cen_loc       = 0

# Simulation temperatures
Thot          = 300
Tcold         = 295
Tambset       = 300

# Frequency of AMR cycle
ff            = 0.5  # [Hz] frequency of AMR cycle

# Flow profile

# FAME cooler
# dispV         = 30.52e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
# acc_period    = 10
# max_flow_per  = 45
# full_magn_ang = 30
# unbal_rat     = 1
# from sourcefiles.device.FAME_V_flow import vol_flow_rate
# volum_flow_profile = vol_flow_rate(timesteps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)

# POLO cooler
dispV = 6.85e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
from sourcefiles.device.polo_V_flow import polo_vol_flow
volum_flow_profile = polo_vol_flow(timesteps, dispV, ff)

# Magnetic field profile

# FAME Cooler
# from sourcefiles.device import FAME_app_field
# app_field = FAME_app_field.app_field(timesteps, nodes)

# POLO cooler
from sourcefiles.device.polo_mag_field import polo_app_field
app_field = polo_app_field(timesteps, nodes, 0.1)

# Geometry of regenerator
# Dsp           = 600e-6
# er            = 0.36
cName   = "polo_1"  # Name of file where the geometric configuration of the regenerator is defined
jName   = "polo_trial" # DP: It is better to use underline to connect words because this is used as file name
num_reg = 1

# Switches for activating and deactivating terms in governing equations
CF   = 1
CS   = 1
CL   = 1
CVD  = 1
CMCE = 1

# Heat transfer models
htc_model_name = 'wakao_and_kagei_1982'  # Name of the file containing the function of the model for htc
leaks_model_name = 'polo_resistance'  # Name of the file containing the function of the model for heat leaks


results = runActive(caseNumber, Thot, Tcold, cen_loc, Tambset, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, cName,
                    jName, time_limit, cycle_toler, maxStepIter, maxCycleIter, volum_flow_profile, app_field,
                    htc_model_name, leaks_model_name,num_reg)


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
#  Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,
#  0       1   2   3     4         5     6           7      8    9      10        11       12
# sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh,cycleCount
#  13     14 15 16    17       18  19   20 21    22         23       24         25      26     27


fileName = "POLO_trial_4.txt"
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