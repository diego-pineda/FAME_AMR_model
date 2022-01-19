from FAME_DP_V1 import runActive
import numpy as np
import sys


# ------- Some useful functions for storing data --------


def FileSave(filename, content):
    with open(filename, "a") as myfile:
        myfile.write(content)


def FileSaveMatrix(filename, content):
    with open(filename, "a") as f:
        for line in content:
            f.write(" ".join("{:9.4f}\t".format(x) for x in line))
            f.write("\n")


def FileSaveVector(filename, content):
    with open(filename, "a") as f:
        f.write(" ".join("{:9.4f}\t".format(x) for x in content))
        f.write("\n")


# ------- Definition of the variables that change for the cases to simulate -------

'''Note: In principle any parameter could become one of the changing variables, i.e. vble1, vble2, and vble3. '''

Thotlow = 273+33  # [K]
Thothigh = 273+41  # [K]
hotResolution = 5

MinTspan = 3  # [K]
MaxTSpan = 30  # [K]
TspanResolution = 10

vble1name = 'Vflow'
vble1units = 'Lpm'
vble1lowvalue = 0.1
vble1highvalue = 4
vble1resolution = 13

vble2name = 'fAMR'
vble2units = 'Hz'
vble2lowvalue = 1
vble2highvalue = 1
vble2resolution = 1

vble3name = 'Dsp'
vble3units = 'm'
vble3lowvalue = 600e-6
vble3highvalue = 600e-6
vble3resolution = 1

vble1values = np.linspace(vble1lowvalue, vble1highvalue, vble1resolution)
vble2values = np.linspace(vble2lowvalue, vble2highvalue, vble2resolution)
vble3values = np.linspace(vble3lowvalue, vble3highvalue, vble3resolution)

Thotarr = np.linspace(Thotlow, Thothigh, hotResolution)
Tspanarr = np.linspace(MinTspan, MaxTSpan, TspanResolution)

numGroups = vble1resolution * vble2resolution * vble3resolution
maxcase = numGroups * hotResolution * TspanResolution


# ------- Input parameters common to all cases -------

# Numerical parameters
nodes         = 600
timesteps     = 600
time_limit    = 7200  # [min] Time limit for the simulation in minutes
cycle_toler   = 1e-4  # Maximum cycle tolerance: criterion for ending the iterative calculation process
maxStepIter   = 2000  # Maximum time step iterations the simulation is allowed to take
maxCycleIter  = 1000  # Maximum cycle iterations the simulation is allowed to take
cen_loc       = 0

# Temperatures

Tambset       = 300

# Frequency of AMR cycle
# ff            = 0.25

# Flow profile

# - FAME cooler
# dispV         = 30.52e-6
acc_period    = 5
max_flow_per  = 45
full_magn_ang = 30
unbal_rat     = 1
from sourcefiles.device.FAME_V_flow import vol_flow_rate
# volum_flow_profile = vol_flow_rate(timesteps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)

# - POLO cooler
# dispV = 4.74e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
# from sourcefiles.device.polo_V_flow import polo_vol_flow
# volum_flow_profile = polo_vol_flow(timesteps, dispV, ff)

# Magnetic field profile

# - FAME cooler
from sourcefiles.device import FAME_app_field
max_app_field = 0.875
app_field = FAME_app_field.app_field(timesteps, nodes, max_app_field)

# - POLO cooler
# from sourcefiles.device.polo_mag_field import polo_app_field
# app_field = polo_app_field(timesteps, nodes, 0.1)

# Geometric parameters
'''Geometric parameters that differ from the configuration from case to case can be adjusted here. This way it is not
necessary to create a new configuration file each time a single parameter need to be changed. Take the following lines
starting with R8 as an example.'''
from configurations import R8
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
htc_model_name = 'Macias_Machin_1991'  # Name of the file containing the function of the model for htc
leaks_model_name = 'flow_btw_plates'  # Name of the file containing the function of the model for heat leaks


if __name__ == '__main__':

    def RunCaseThotTcold(case, jobName):  # DP: this is necessary for running arrays of tasks in the cluster

        casegroup  = int(np.floor(case/(hotResolution * TspanResolution)))
        fileName = "{}.txt".format(jobName)

        Thotindex = int(np.floor(case / TspanResolution) % hotResolution)
        Tspanindex = int(case % TspanResolution)

        vble1index = int(np.floor((casegroup - vble1resolution * int(np.floor(casegroup / vble1resolution))) / 1))
        vble2index = int(np.floor((casegroup - vble1resolution * vble2resolution * int(np.floor(casegroup / (vble1resolution * vble2resolution)))) / vble1resolution))
        vble3index = int(np.floor((casegroup - vble1resolution * vble2resolution * vble3resolution * int(np.floor(casegroup / (vble1resolution * vble2resolution * vble3resolution)))) / (vble1resolution * vble2resolution)))

        # ------- Parameters that change for the cases to study -------

        # Note: place in this section the parameters that change in the simulations. The volume displacement and
        # frequency of the AMR cycle are taken as changing parameters in what follows, but it can be any of the input
        # parameters listed above. Please remember that the changing parameters must be commented out in the section
        # above. It is not necessary to provide three changing parameters necessarily

        dispV = vble1values[vble1index] * 16.667e-6
        ff = vble2values[vble2index]
        R8.Dsp = vble3values[vble3index]

        volum_flow_profile = vol_flow_rate(timesteps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)

        # -------- Setting values of Thot and Tcold for simulation -------

        Thot = Thotarr[Thotindex]
        Tcold = Thot - Tspanarr[Tspanindex]

        print("Iteration: {}/{} Case group number: {}".format(case, maxcase-1, casegroup))

        results = runActive(case, Thot, Tcold, cen_loc, Tambset, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, cName,
                            jobName, time_limit, cycle_toler, maxStepIter, maxCycleIter, volum_flow_profile, app_field,
                            htc_model_name, leaks_model_name, num_reg)
        #  runActive():  returns
        # Thot          0   eff_HB_CE   6   sHalfBlow   12  Uti         18  sMaxCBlow   24  fluid_dens  30
        # Tcold         1   eff_CB_HE   7   sEndBlow    13  freq        19  sMaxHBlow   25  mass_flow   31
        # qc            2   tFce        8   y           14  t           20  qh          26
        # qccor         3   tFhe        8   s           15  xloc        21  cycleCount  27
        # (t1-t0)/60    4   yHalfBlow   10  pt          16  yMaxCBlow   22  int_field   28
        # pave          5   yEndBlow    11  np.max(pt)  17  yMaxHBlow   23  htc_fs      29

        fileNameSave = './output/' + str(case) + fileName  # This is for the HPC11 cluster at TU Delft
        # fileNameSave = '/scratch/dpineda/' + str(case) + fileName  # This is for the THCHEM cluster at RU Nijmegen
        FileSave(fileNameSave, "{},{},{},{},{},{},{},{} \n".format('Tspan [K]', 'Qh [W]', 'Qc [W]', 'Cycles [-]', 'Run time [min]', 'Max. Pressure drop [Pa]', 'Thot [K]', 'Tcold [K]'))
        FileSave(fileNameSave, "{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f},{},{} \n".format(results[0]-results[1], results[26], results[2], results[27], results[4], results[17], Thot, Tcold))
        FileSave(fileNameSave, "Fluid temperatures\n")
        FileSaveMatrix(fileNameSave, results[14])
        FileSave(fileNameSave, "Solid temperatures\n")
        FileSaveMatrix(fileNameSave, results[15])
        FileSave(fileNameSave, "Pressure drop accross the regenerator for the entire cycle\n")
        FileSaveVector(fileNameSave, results[16])
        FileSave(fileNameSave, "\nInternal Magnetic Field\n")
        FileSaveMatrix(fileNameSave, results[28])
        FileSave(fileNameSave, "\nHeat transfer coefficient between solid and fluid in the packed bed\n")
        FileSaveMatrix(fileNameSave, results[29])
        FileSave(fileNameSave, "\nMass flow rate\n")
        FileSaveMatrix(fileNameSave, results[31])

    RunCaseThotTcold(float(sys.argv[1]), sys.argv[2])
