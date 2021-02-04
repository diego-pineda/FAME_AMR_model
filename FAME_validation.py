from FAME_DP_V1 import runActive
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


    #runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName,cycle_toler)
    #MaxTSpan      = 10
    caseNumber    = 6
    Thot          = 294.8  # [K]
    Tcold         = 294.79
    cen_loc       = 0
    Tambset       = 298  # [K] Temperature of room reported between 296 K to 300 K
    dispV         = 12.94e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min. 2.37 regenerators with simultaneous flow.
    ff            = 1.2  # [Hz] DP: frequency of AMR cycle
    CF            = 1
    CS            = 1
    CL            = 0
    CVD           = 1
    CMCE          = 1
    nodes         = 400
    timesteps     = 600
    Dsp           = 600e-6  # [m] Bowei reported that he used spheres in between 400 and 800 micrometers
    cName         = "R7"
    jName         = "Validation_Gd_span_11K"  # DP: It is better to use underline to connect words because this is used as file name
    time_limit    = 600  # [min] Time limit for completing the calculations of the runActive function in minutes
    cycle_toler   = 1e-4 # User defined criterion for convergence
    maxStepIter   = 200  # Maximum time step iterations the simulation is allowed to take
    maxCycleIter  = 300  # Maximum cycle iterations the simulation is allowed to take

    Qc_corr =1  # This is just for allowing the initialization of the following loop
    Qc_corr_data = []
    Tspan_data = []
    Qc_data = []

    while Qc_corr > 0:
        results = runActive(caseNumber, Thot, Tcold, cen_loc, Tambset, dispV, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, Dsp, cName, jName, time_limit, cycle_toler, maxStepIter, maxCycleIter)
        Tspan = Thot-Tcold
        Tspan_data.append(Tspan)
        Qc = results[2]  # [W] Cooling capacity of the device without thermal losses correction
        Qc_data.append(Qc)
        Qc_corr = results[3]  # [W] Cooling capacity of the device corrected for thermal losses in CHEX
        Qc_corr_data.append(Qc_corr)
        Tcold = Tcold-5

    plot1 = plt.figure(1)
    plt.plot(Qc_corr_data, Tspan_data)
    plt.plot([0, 24, 51, 77, 103], [9.7, 8, 6, 3.3, 0.01])
    plt.plot(Qc_data, Tspan_data)
    plt.legend(["Simulation with heat losses in CHEX", "Bowei's experiments", "Simulation without heat losses in CHEX"])
    plt.xlabel("Cooling capacity [W]")
    plt.ylabel("Temperature span [K]")
    plt.title("Temperature span vs Cooling capacity")
    plt.grid(which='both', axis='both')

    plt.show()

#     results = runActive(caseNumber,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,cName,jName,time_limit,cycle_toler,maxStepIter,maxCycleIter)
#
#     # Some useful functions for storing data.
#     def FileSave(filename, content):
#         with open(filename, "a") as myfile:
#             myfile.write(content)
#
#     def FileSaveMatrix(filename, content):
#         with open(filename, "a") as f:
#             for line in content:
#                 f.write(" ".join("{:9.6f}\t".format(x) for x in line))
#                 f.write("\n")
#
# #  runActive():  return Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh
# #                       0       1   2   3     4         5     6           7      8    9      10        11       12       13     14 15 16    17       18  19   20 21    22         23       24         25      26
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
