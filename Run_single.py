from FAME_DP_V1 import runActive
import numpy as np


# ---------------------------------- Calculation of just one case --------------------------------------

#runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName,time_limit,cycle_toler,maxStepIter,maxCycleIter)

caseNumber    = 2
Thot          = 295
Tcold         = 292
cen_loc       = 0
Tambset       = 298
dispV         = 30.52e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
ff            = 1.7  # [Hz] DP: frequency of AMR cycle
CF            = 1
CS            = 1
CL            = 0
CVD           = 1
CMCE          = 1
nodes         = 400
timesteps     = 600
Dsp           = 600e-6
er            = 0.36
cName         = "R7"
jName         = "Int_htc" # DP: It is better to use underline to connect words because this is used as file name
time_limit    = 600  # [min] Time limit for the simulation in minutes
cycle_toler   = 1e-1  # Maximum cycle tolerance: criterion for ending the iterative calculation process
maxStepIter   = 300  # Maximum time step iterations the simulation is allowed to take
maxCycleIter  = 300  # Maximum cycle iterations the simulation is allowed to take

results = runActive(caseNumber, Thot, Tcold, cen_loc, Tambset, dispV, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, Dsp, er, cName, jName, time_limit,cycle_toler, maxStepIter, maxCycleIter)


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


fileName = "Int_htc2.txt"
fileNameSave = './output/' + fileName
#FileSave(fileNameSave,"{},{},{},{},{},{},{} \n".format(results[0], results[1], results[2], results[3], results[4], results[5],results[26]))
FileSave(fileNameSave, "{},{},{},{},{},{} \n".format('Tspan [K]', 'Qc_corr [W]', 'Qc [W]', 'Cycles [-]', 'run time [min]', 'Max. Pressure drop [Pa]'))
FileSave(fileNameSave, "{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f} \n".format(results[0]-results[1], results[3], results[2], results[27], results[4], results[17]))
FileSave(fileNameSave, "Fluid temperatures\n")
FileSaveMatrix(fileNameSave, results[14])
#FileSave(fileNameSave, "\n")
FileSave(fileNameSave, "Solid temperatures\n")
FileSaveMatrix(fileNameSave, results[15])
#FileSave(fileNameSave, "\n")
FileSave(fileNameSave, "Pressure drop accross the regenerator for the entire cycle\n")
FileSaveVector(fileNameSave, results[16])
#FileSave(fileNameSave, "\n")
print(results[16])


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