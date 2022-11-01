from FAME_DP_V1 import runActive
from tools.write_data_to_file import FileSave, FileSaveMatrix, FileSaveVector
from tools.reduce_matrix import reduce_matrix
import numpy as np
import sys
import importlib


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

Thotarr = np.linspace(Thotlow, Thothigh, hotResolution)
Tspanarr = np.linspace(MinTspan, MaxTSpan, TspanResolution)

vble1values = np.linspace(vble1lowvalue, vble1highvalue, vble1resolution)
vble2values = np.linspace(vble2lowvalue, vble2highvalue, vble2resolution)
vble3values = np.linspace(vble3lowvalue, vble3highvalue, vble3resolution)

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

cName = "R8"
num_reg = 1

configuration = importlib.import_module('configurations.' + cName)

# R8.species_discription = ['reg-M6', 'reg-M7', 'reg-M8', 'reg-M9', 'reg-M10', 'reg-M11', 'reg-M12', 'reg-M13', 'reg-M14', 'reg-M15']
# R8.x_discription = [0, 0.006, 0.012, 0.018, 0.024, 0.030, 0.036, 0.042, 0.048, 0.054, 0.060]
# R8.reduct_coeff = dict(M0=1, M1=0.55, M2=0.77, M6=1, M7=1, M8=1, M9=1, M10=1, M11=1, M12=1, M13=1, M14=1, M15=1)
# R8.mK = 6
# R8.mRho = 6100

# Switches for activating and deactivating terms in governing equations
CF   = 1  # Heat conduction in the fluid
CS   = 1  # Heat conducion in the solid
CL   = 1  # Heat leaks through AMR casing
CVD  = 1  # Viscous dissipation
CMCE = 1  # Magnetocaloric effect

# Flow and Heat transfer models
htc_model_name = 'Macias_Machin_1991'  # Name of the file containing the function of the model for htc
leaks_model_name = 'flow_btw_plates'  # Name of the file containing the function of the model for heat leaks
pdrop_model_name = 'pb_ergun_1952'

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

        # Note: place in this section the parameters that change in the simulations. The volume flow rate and
        # frequency of the AMR cycle are taken as changing parameters in what follows, but it can be any of the input
        # parameters listed above. Please remember that the changing parameters must be commented out in the section
        # above. Any of these three parameters can have a single value.

        dispV = vble1values[vble1index] * 16.667e-6
        ff = vble2values[vble2index]
        configuration.Dsp = vble3values[vble3index]

        volum_flow_profile = vol_flow_rate(timesteps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)

        # -------- Setting values of Thot and Tcold for simulation -------

        Thot = Thotarr[Thotindex]
        Tcold = Thot - Tspanarr[Tspanindex]

        print("Iteration: {}/{} Case group number: {}".format(case, maxcase-1, casegroup))

        results = runActive(case, Thot, Tcold, cen_loc, Tambset, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, cName,
                            jobName, time_limit, cycle_toler, maxStepIter, maxCycleIter, volum_flow_profile, app_field,
                            htc_model_name, leaks_model_name, pdrop_model_name, num_reg)
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
        # tFhe          9  |  xloc        21 |  k_stat      33 |
        # yHalfBlow     10 |  yMaxCBlow   22 |  k_disp      34 |
        # yEndBlow      11 |  yMaxHBlow   23 |  S_ht_hot    35 |

        fileNameSave = './output/' + str(case) + fileName  # This is for the HPC11 cluster at TU Delft
        #fileNameSave = '/scratch/dfpinedaquijan/' + str(case) + fileName  # This is for the DelftBlue cluster at TU Delft
        # fileNameSave = '/scratch/dpineda/' + str(case) + fileName  # This is for the THCHEM cluster at RU Nijmegen
        FileSave(fileNameSave, "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n".format('Tspan [K]', 'Qh [W]', 'Qc [W]', 'Cycles [-]', 'Run time [min]', 'Max. Pressure drop [Pa]', 'Thot [K]', 'Tcold [K]', 'S_ht_hot [W/K]', 'S_ht_cold [W/K]', 'S_ht_fs [W/K]', 'S_vd [W/K]', 'S_condu_stat [W/K]', 'S_condu_disp [W/K]', 'S_ht_amb [W/K]', 'Pump_power_input [W]', 'Mag_power_input [W]', 'Q_leak [W]'))
        FileSave(fileNameSave, "{},{:7.4f},{:7.4f},{},{:7.4f},{:7.4f},{},{},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f},{:7.6f} \n".format(results[0]-results[1], results[26], results[2], results[27], results[4], results[17], Thot, Tcold, results[35], results[36], results[37], results[38], results[39], results[40], results[41], results[42], results[43], results[44]))
        FileSave(fileNameSave, "Fluid temperatures\n")
        FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[14], 3, 2))
        FileSave(fileNameSave, "Solid temperatures\n")
        FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[15], 3, 2))
        FileSave(fileNameSave, "Pressure drop accross the regenerator for the entire cycle\n")
        FileSaveVector(fileNameSave, results[16])
        FileSave(fileNameSave, "\nInternal Magnetic Field\n")
        FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[28], 3, 2))
        FileSave(fileNameSave, "\nHeat transfer coefficient between solid and fluid in the packed bed\n")
        FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[29], 3, 2))
        FileSave(fileNameSave, "\nMass flow rate\n")
        FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[31], 3, 2))
        FileSave(fileNameSave, "\nPressure drop per unit length\n")
        FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[32], 3, 2))
        FileSave(fileNameSave, "\nEffective thermal conductivity of solid\n")
        FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[33], 3, 2))
        FileSave(fileNameSave, "\nEffective thermal conductivity of fluid\n")
        FileSaveMatrix(fileNameSave, reduce_matrix(nodes, timesteps, results[34], 3, 2))

    RunCaseThotTcold(float(sys.argv[1]), sys.argv[2])
