from FAME_DP_V1 import runActive
import numpy as np
import sys


if __name__ == '__main__':

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

    def RunCaseThotTcold(case, jobName):  # DP: this is necessary for running arrays of tasks in the cluster

        numCases       = 6
        hotResolution  = 1
        coldResolution = 6

        maxcase = numCases * hotResolution * coldResolution
        Thotarr = np.linspace(273+27, 273+27, hotResolution)
        casenum=int(np.floor(case/(hotResolution*coldResolution)))

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

        # ------- Input parameters common to all cases -------

        fileName      = "{}.txt".format(jobName)

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
        # dispV         = 30.52e-6
        # acc_period    = 10
        # max_flow_per  = 45
        # full_magn_ang = 30
        # unbal_rat     = 1
        # from sourcefiles.device.FAME_V_flow import vol_flow_rate
        # volum_flow_profile = vol_flow_rate(timesteps, dispV, acc_period, max_flow_per, full_magn_ang, unbal_rat)

        # - POLO cooler
        dispV = 4.74e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
        from sourcefiles.device.polo_V_flow import polo_vol_flow
        volum_flow_profile = polo_vol_flow(timesteps, dispV, ff)

        # Magnetic field profile

        # - FAME cooler
        # from sourcefiles.device import FAME_app_field
        # app_field = FAME_app_field.app_field(timesteps, nodes)

        # - POLO cooler
        from sourcefiles.device.polo_mag_field import polo_app_field
        app_field = polo_app_field(timesteps, nodes, 0.1)

        # Geometric parameters
        cName = "polo_1"
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
        FileSave(fileNameSave, "{},{},{},{},{},{} \n".format('Tspan [K]', 'Qh [W]', 'Qc [W]', 'Cycles [-]', 'run time [min]', 'Max. Pressure drop [Pa]'))
        FileSave(fileNameSave, "{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f} \n".format(results[0]-results[1], results[26], results[2], results[27], results[4], results[17]))
        FileSave(fileNameSave, "Fluid temperatures\n")
        FileSaveMatrix(fileNameSave, results[14])
        FileSave(fileNameSave, "Solid temperatures\n")
        FileSaveMatrix(fileNameSave, results[15])
        FileSave(fileNameSave, "Pressure drop accross the regenerator for the entire cycle\n")
        FileSaveVector(fileNameSave, results[16])

        # fileNameSave        = './' + fileName # DP: ./ is for specifying that the file is save to the working directory
        # fileNameEndTemp     = './Ends/{:3.0f}-{:3.0f}-PysicalEnd'.format(Thot,Tcold)+fileName
        # fileNameSliceTemp   = './Blow/{:3.0f}-{:3.0f}-BlowSlice'.format(Thot,Tcold)+fileName
        # FileSave(fileNameSave,"{},{},{},{},{},{},{} \n".format(results[0],results[1],results[2],results[3],results[4],results[5],results[26]) )
        # #FileSave(fileNameEndTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]','Uti [-]', 'freq [Hz]', 'run time [min]','Eff CE-HB [-]', 'Eff HE-CB [-]') )
        # #FileSave(fileNameEndTemp,"{},{},{},{},{} \n".format(results[0],results[1],results[18],results[19], results[4],results[6],results[7]) )
        # #EndTemperatures = np.stack((results[20], results[8],results[9]), axis=-1)
        # #FileSaveMatrix(fileNameEndTemp,EndTemperatures)
        # FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]','Uti [-]', 'freq [Hz]', 'run time [min]') )
        # FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format(results[0],results[1],results[18],results[19], results[4]) )
        # BlowSliceTemperatures = np.stack((results[21],results[10],results[11],results[12],results[13],results[22],results[23],results[24],results[25]), axis=-1)
        # FileSaveMatrix(fileNameSliceTemp,BlowSliceTemperatures)

    RunCaseThotTcold(float(sys.argv[1]), sys.argv[2])
