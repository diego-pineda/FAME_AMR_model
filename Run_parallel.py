from FAME_DP_V1 import runActive
import numpy as np
import sys


if __name__ == '__main__':

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


    def RunCaseThotTcold(case, jobName):  # DP: this is necessary for running arrays of tasks in the cluster
        numCases       = 1
        hotResolution  = 1
        coldResolution = 8

        maxcase = numCases * hotResolution * coldResolution
        Thotarr = np.linspace(273+22, 273+22, hotResolution)

        casenum=int(np.floor(case/(hotResolution*coldResolution))) # DP: I don't understand why making things complicated like this...

        if casenum==0:
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 24
            cen_loc       = 0
            Tambset       = 298
            dispV         = 30.52e-6
            ff            = 1.7
            Dsp           = 600e-6
            er = 0.36
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 600
            cName         = "R7"
            time_limit    = 600  # [min] Time limit for the simulation in minutes
            cycle_toler   = 1e-5  # Maximum cycle tolerance: criterion for ending the iterative calculation process
            maxStepIter   = 300  # Maximum time step iterations the simulation is allowed to take
            maxCycleIter  = 300  # Maximum cycle iterations the simulation is allowed to take
        if casenum==1:
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 24
            cen_loc       = 0
            Tambset       = 298
            dispV         = 30.52e-6
            ff            = 1.7
            Dsp           = 600e-6
            er = 0.36
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 600
            cName         = "R7"
            time_limit    = 700  # [min] Time limit for the simulation in minutes
            cycle_toler   = 1e-5  # Maximum cycle tolerance: criterion for ending the iterative calculation process
            maxStepIter   = 500  # Maximum time step iterations the simulation is allowed to take
            maxCycleIter  = 500  # Maximum cycle iterations the simulation is allowed to take
        if casenum==2:
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 24
            cen_loc       = 0
            Tambset       = 298
            dispV         = 30.52e-6
            ff            = 1.7
            Dsp           = 600e-6
            er = 0.36
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 600
            cName         = "R7"
            time_limit    = 800  # [min] Time limit for the simulation in minutes
            cycle_toler   = 1e-6  # Maximum cycle tolerance: criterion for ending the iterative calculation process
            maxStepIter   = 700  # Maximum time step iterations the simulation is allowed to take
            maxCycleIter  = 700  # Maximum cycle iterations the simulation is allowed to take
        if casenum==3:
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 24
            cen_loc       = 0
            Tambset       = 298
            dispV         = 30.52e-6
            ff            = 1.7
            Dsp           = 600e-6
            er = 0.36
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 600
            cName         = "R7"
            time_limit    = 900  # [min] Time limit for the simulation in minutes
            cycle_toler   = 1e-7  # Maximum cycle tolerance: criterion for ending the iterative calculation process
            maxStepIter   = 900  # Maximum time step iterations the simulation is allowed to take
            maxCycleIter  = 900  # Maximum cycle iterations the simulation is allowed to take

        Thot = Thotarr[int(np.floor(case/coldResolution)%hotResolution)]
        Tcold = Thot - MaxTSpan*(case%(coldResolution))/(coldResolution)-0.1

        print("iteration: {}/{} Case number: {} Thot: {} Tcold: {}".format(case, maxcase, casenum, Thot, Tcold))

        results = runActive(case,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,er,cName,jobName,time_limit,cycle_toler,maxStepIter,maxCycleIter)
        #  runActive():  returns
        #  Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,
        #  0       1   2   3     4         5     6           7      8    9      10        11       12
        # sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh,cycleCount
        #  13     14 15 16    17       18  19   20 21    22         23       24         25      26     27

        fileNameSave = './output/' + str(case) + fileName
        FileSave(fileNameSave, "{},{},{},{},{},{} \n".format('Tspan [K]', 'Qc_corr [W]', 'Qc [W]', 'Cycles [-]', 'run time [min]', 'Max. Pressure drop [Pa]'))
        FileSave(fileNameSave, "{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f} \n".format(results[0]-results[1], results[3], results[2], results[27], results[4], results[17]))
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
