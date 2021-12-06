from FAME_DP_V1 import runActive
import numpy as np
import sys
from configurations import R8

# ------- Some useful functions for storing data --------

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


# ------- Definition of main variables of the cases to simulate -------

'''Note: In principle any variable could become x in xResolution, e.g. dispV. It does not mean that
the name xResolution needs to be adjusted. hotResolution must be 1 when xResolution is different than 1, but they both
can be 1 simultaneously. For each case to simulate, the value of the variable that is going to take the place of x, 
must be defined in the same way Thot is defined, i.e. by using the following equation (taking as an example the 
variable dispV): dispV = xArr[int(np.floor(case/coldResolution)%xResolution)]. This variable has to be defined outside 
the if conditions (if casenum == 0, 1, 2, etc).'''

numCases       = 6
hotResolution  = 1
coldResolution = 6
xResolution    = 1

Thotarr = np.linspace(273+27, 273+27, hotResolution)
xArr    = []
maxcase = numCases * hotResolution * coldResolution * xResolution

# ------- Input parameters common to all cases -------

# Numerical parameters
nodes         = 400
timesteps     = 600
time_limit    = 600  # [min] Time limit for the simulation in minutes
cycle_toler   = 1e-5  # Maximum cycle tolerance: criterion for ending the iterative calculation process
maxStepIter   = 500  # Maximum time step iterations the simulation is allowed to take
maxCycleIter  = 500  # Maximum cycle iterations the simulation is allowed to take
cen_loc       = 0

# Temperatures
MaxTSpan      = 30
Tambset       = 300

# Frequency of AMR cycle
# ff            = 0.25

# Flow profile

# - FAME cooler
dispV         = 30.52e-6
acc_period    = 10
max_flow_per  = 45
full_magn_ang = 30
unbal_rat     = 1
from sourcefiles.device.FAME_V_flow import vol_flow_rate
volum_flow_profile = vol_flow_rate(timesteps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)

# - POLO cooler
# dispV = 4.74e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
# from sourcefiles.device.polo_V_flow import polo_vol_flow
# volum_flow_profile = polo_vol_flow(timesteps, dispV, ff)

# Magnetic field profile

# - FAME cooler
from sourcefiles.device import FAME_app_field
app_field = FAME_app_field.app_field(timesteps, nodes)

# - POLO cooler
# from sourcefiles.device.polo_mag_field import polo_app_field
# app_field = polo_app_field(timesteps, nodes, 0.1)

# Geometric parameters
'''Geometric parameters that differ from the configuration from case to case can be adjusted here. This way it is not
necessary to create a new configuration file each time a single parameter need to be changed. Take the following lines
starting with R8 as an example.'''

# R8.species_discription = ['reg-M6', 'reg-M7', 'reg-M8', 'reg-M9', 'reg-M10', 'reg-M11', 'reg-M12', 'reg-M13', 'reg-M14', 'reg-M15']
# R8.x_discription = [0, 0.006, 0.012, 0.018, 0.024, 0.030, 0.036, 0.042, 0.048, 0.054, 0.060]
# R8.reduct_coeff = dict(M0=1, M1=0.55, M2=0.77, M6=1, M7=1, M8=1, M9=1, M10=1, M11=1, M12=1, M13=1, M14=1, M15=1)
# R8.mK = 6
# R8.mRho = 6100
cName = "R8"
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


if __name__ == '__main__':

    def RunCaseThotTcold(case, jobName):  # DP: this is necessary for running arrays of tasks in the cluster

        casenum  = int(np.floor(case/(hotResolution * coldResolution * xResolution)))
        fileName = "{}.txt".format(jobName)

        # ------- Parameters that change for the cases to study -------

        # Note: place in this section the parameter that is to be changed in the simulations. The number of 'if'
        # conditions must be at least equal to the number of cases (numCases) to run. The frequency of the
        # AMR cycle is set to be the changing parameter in what follows. It can be any of the input parameters listed
        # above. Please remember that the changing parameter must be commented out in the section above.

        if casenum == 0:
            # Frequency of AMR cycle
            ff = 0.25

        if casenum == 1:
            # Frequency of AMR cycle
            ff = 0.5

        if casenum == 2:
            # Frequency of AMR cycle
            ff = 0.75

        if casenum == 3:
            # Frequency of AMR cycle
            ff = 1

        if casenum == 4:
            # Frequency of AMR cycle
            ff = 1.25

        if casenum == 5:
            # Frequency of AMR cycle
            ff = 1.5

        # -------- Setting values of Thot and Tcold for simulation -------

        Thot = Thotarr[int(np.floor(case/coldResolution)%hotResolution)]
        Tcold = Thot - MaxTSpan*(case%(coldResolution))/(coldResolution)-0.1

        print("Iteration: {}/{} Case number: {} Thot: {} Tcold: {}".format(case, maxcase, casenum, Thot, Tcold))
        print("Tamb = {} [K], V_flow_rate = {} [m3/s], Freq AMR = {} [Hz]".format(Tambset, dispV, ff))

        results = runActive(case,Thot,Tcold,cen_loc,Tambset,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,cName,jobName,
                            time_limit,cycle_toler,maxStepIter,maxCycleIter, volum_flow_profile, app_field,
                            htc_model_name, leaks_model_name,num_reg)
        #  runActive():  returns
        #  Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,
        #  0       1   2   3     4         5     6           7      8    9      10        11       12
        # sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh,cycleCount
        #  13     14 15 16    17       18  19   20 21    22         23       24         25      26     27

        fileNameSave = './output/' + str(case) + fileName
        FileSave(fileNameSave, "{},{},{},{},{},{},{},{} \n".format('Tspan [K]', 'Qh [W]', 'Qc [W]', 'Cycles [-]', 'run time [min]', 'Max. Pressure drop [Pa]', 'Thot [K]', 'Tcold [K]'))
        FileSave(fileNameSave, "{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f},{},{} \n".format(results[0]-results[1], results[26], results[2], results[27], results[4], results[17], Thot, Tcold))
        FileSave(fileNameSave, "Fluid temperatures\n")
        FileSaveMatrix(fileNameSave, results[14])
        FileSave(fileNameSave, "Solid temperatures\n")
        FileSaveMatrix(fileNameSave, results[15])
        FileSave(fileNameSave, "Pressure drop accross the regenerator for the entire cycle\n")
        FileSaveVector(fileNameSave, results[16])

    RunCaseThotTcold(float(sys.argv[1]), sys.argv[2])
