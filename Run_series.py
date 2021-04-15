from FAME_DP_V1 import runActive


if __name__ == '__main__':

    # The following section can be used to run an array of cases in a single computer, one after another

    # runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName,time_limit,cycle_toler,maxStepIter,maxCycleIter)

    caseNumber    = 10
    Thot          = 294.8  # [K]
    Tcold         = 274.8
    cen_loc       = 0
    Tambset       = 298  # [K] Temperature of room reported between 296 K to 300 K
    dispV         = 12.94e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min. 2.37 regenerators with simultaneous flow.
    acc_period    = 10
    max_flow_per  = 45
    full_magn_ang = 30
    unbal_rat     = 1
    ff            = 1.2  # [Hz] DP: frequency of AMR cycle
    CF            = 1
    CS            = 1
    CL            = 0  # Switch for heat leaks in the fluid GE. Set to 1 in the runActive function when CL_set="Tamb"
    CVD           = 1
    CMCE          = 1
    nodes         = 400
    timesteps     = 600
    Dsp           = 600e-6  # [m] Bowei reported that he used spheres in between 400 and 800 micrometers
    er            = 0.36
    cName         = "R7"
    jName         = "Validation_Gd_no_voids_er0.45_Tspan20"  # DP: It is better to use underline to connect words because this is used as file name
    time_limit    = 600  # [min] Time limit for completing the calculations of the runActive function in minutes
    cycle_toler   = 1e-4 # User defined criterion for convergence
    maxStepIter   = 400  # Maximum time step iterations the simulation is allowed to take
    maxCycleIter  = 600  # Maximum cycle iterations the simulation is allowed to take

    # Some useful functions for storing data

    def FileSave(filename, content):
        with open(filename, "a") as myfile:
            myfile.write(content)
    #
    def FileSaveMatrix(filename, content):
        with open(filename, "a") as f:
            for line in content:
                f.write(" ".join("{:9.6f}\t".format(x) for x in line))
                f.write("\n")

    fileName = jName
    fileNameSave = './' + fileName + '.txt'

    Qc = 1  # This is just for allowing the initialization of the following loop
    Qc_corr_data = []
    Tspan_data = []
    Qc_data = []

    while Qc > 0:
        results = runActive(caseNumber, Thot, Tcold, cen_loc, Tambset, dispV, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, Dsp, er, cName, jName, time_limit, cycle_toler, maxStepIter, maxCycleIter, acc_period, max_flow_per, full_magn_ang, unbal_rat)
        #  runActive():  returns
        #  Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,
        #  0       1   2   3     4         5     6           7      8    9      10        11       12
        # sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh,cycleCount
        #  13     14 15 16    17       18  19   20 21    22         23       24         25      26     27
        Tspan = Thot-Tcold
        Tspan_data.append(Tspan)
        Qc = 7*results[2]  # [W] Cooling capacity of the device without thermal losses correction
        Qc_data.append(Qc)
        Qc_corr = results[3]  # [W] Cooling capacity of the device corrected for thermal losses in CHEX
        Qc_corr_data.append(Qc_corr)
        Tcold = Tcold-5
        FileSave(fileNameSave,"{},{},{},{},{},{} \n".format('Tspan [K]', 'Qc_corr [W]', 'Qc [W]', 'Cycles [-]', 'run time [min]', 'Max. Pressure drop [Pa]'))
        FileSave(fileNameSave,"{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f} \n".format(results[0]-results[1], results[3], results[2], results[27], results[4], results[17]))
        FileSave(fileNameSave,"Fluid temperatures \n")
        FileSaveMatrix(fileNameSave, results[14])
        FileSave(fileNameSave,"\n")
        FileSave(fileNameSave,"Solid temperatures \n")
        FileSaveMatrix(fileNameSave, results[15])
        FileSave(fileNameSave,"\n")
        Qc = -1


    # Plotting Tspan vs Qc
    # import numpy as np
    # import matplotlib.pyplot as plt

    # plot1 = plt.figure(1)
    # plt.plot([0, 24, 51, 77, 103], [9.7, 8, 6, 3.3, 0.01])
    # plt.plot(Qc_data, Tspan_data)
    # plt.legend(["Bowei's experiments", "Simulation without heat losses in CHEX"])
    # plt.xlabel("Cooling capacity [W]")
    # plt.ylabel("Temperature span [K]")
    # plt.title("Temperature span vs Cooling capacity")
    # plt.grid(which='both', axis='both')
    #
    # plt.show()




#     results = runActive(caseNumber,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,cName,jName,time_limit,cycle_toler,maxStepIter,maxCycleIter)
#

#
# #  runActive():  return Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh, cycleCount
# #                       0       1   2   3     4         5     6           7      8    9      10        11       12       13     14 15 16    17       18  19   20 21    22         23       24         25      26     27
#
#     fileName = "Validation_val_span_11K.txt"
#     fileNameSave = './' + fileName
#     fileNameSliceTemp = './Blow/{:3.0f}-{:3.0f}-BlowSlice'.format(Thot, Tcold) + fileName
#     FileSave(fileNameSave,"{},{},{},{},{},{},{} \n".format(results[0], results[1], results[2], results[3], results[4], results[5],results[26]))
#     FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]', 'Uti [-]', 'freq [Hz]', 'run time [min]'))
#     FileSave(fileNameSliceTemp,"{},{},{:4.2f},{},{:4.2f} \n".format(results[0], results[1], results[18], results[19], results[4]))
#     BlowSliceTemperatures = np.stack((results[21], results[10], results[11], results[12], results[13], results[22], results[23], results[24], results[25]), axis=-1)
#     FileSaveMatrix(fileNameSliceTemp, BlowSliceTemperatures)
#
#     fluidtemperature = './' + "Fluid_Temp_val_span_11K.txt"
#     fluidtemperatures = results[14]
#     FileSaveMatrix(fluidtemperature,fluidtemperatures)
#
#     solidtemperature = './' + "Solid_Temp_val_span_11K.txt"
#     solidtemperatures = results[15]
#     FileSaveMatrix(solidtemperature,solidtemperatures)
