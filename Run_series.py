from FAME_DP_V1 import runActive
from tools.write_data_to_file import FileSave, FileSaveMatrix, FileSaveVector
from tools.reduce_matrix import reduce_matrix


if __name__ == '__main__':

    # The following section can be used to run an array of cases in a single computer, one after another

    jName         = "Testing_new_func_multi_layer"  # DP: It is better to use underline to connect words because this is used as file name

    # Numerical parameters
    nodes         = 400
    timesteps     = 600
    time_limit    = 600  # [min] Time limit for the simulation in minutes
    cycle_toler   = 1e-5  # Maximum cycle tolerance: criterion for ending the iterative calculation process
    maxStepIter   = 500  # Maximum time step iterations the simulation is allowed to take
    maxCycleIter  = 500  # Maximum cycle iterations the simulation is allowed to take
    cen_loc       = 0

    # Temperatures
    Thot          = 295  # [K]
    Tcold         = 290
    Tambset       = 300

    # Frequency of AMR cycle
    ff            = 1

    # Flow profile

    # - FAME cooler
    dispV         = 30.52e-6
    acc_period    = 5
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
    mag_field = 1.4  # [T]
    app_field = FAME_app_field.app_field(timesteps, nodes, mag_field)

    # - POLO cooler
    # from sourcefiles.device.polo_mag_field import polo_app_field
    # app_field = polo_app_field(timesteps, nodes, 0.1)

    # Geometric parameters
    cName = "R8"
    num_reg = 1

    # Switches for activating and deactivating terms in governing equations
    CF   = 1
    CS   = 1
    CL   = 1
    CVD  = 1
    CMCE = 1

    # Flow and Heat transfer models
    htc_model_name = 'Macias_Machin_1991'  # Name of the file containing the function of the model for htc
    leaks_model_name = 'flow_btw_plates'  # Name of the file containing the function of the model for heat leaks
    pdrop_model_name = 'pb_ergun_1952'

    Qc = 1  # This is just for allowing the initialization of the following loop
    Qc_corr_data = []
    Tspan_data = []
    Qc_data = []
    caseNumber = 0

    while Qc > 0:
        results = runActive(caseNumber, Thot, Tcold, cen_loc, Tambset, ff, CF, CS, CL, CVD, CMCE, nodes,
                            timesteps, cName, jName, time_limit, cycle_toler, maxStepIter, maxCycleIter,
                            volum_flow_profile, app_field, htc_model_name, leaks_model_name, pdrop_model_name, num_reg)
        #  runActive():  returns
        # Thot          0   eff_HB_CE   6   sHalfBlow   12  Uti         18  sMaxCBlow   24  fluid_dens  30
        # Tcold         1   eff_CB_HE   7   sEndBlow    13  freq        19  sMaxHBlow   25  mass_flow   31
        # qc            2   tFce        8   y           14  t           20  qh          26
        # qccor         3   tFhe        8   s           15  xloc        21  cycleCount  27
        # (t1-t0)/60    4   yHalfBlow   10  pt          16  yMaxCBlow   22  int_field   28
        # pave          5   yEndBlow    11  np.max(pt)  17  yMaxHBlow   23  htc_fs      29

        Tspan = Thot-Tcold
        Tspan_data.append(Tspan)
        Qc = results[2]  # [W] Cooling capacity of the device without thermal losses correction
        Qc_data.append(Qc)
        Qc_corr = results[3]  # [W] Cooling capacity of the device corrected for thermal losses in CHEX
        Qc_corr_data.append(Qc_corr)
        Tcold = Tcold-5

        fileNameSave = './output/' + jName + '_' + str(caseNumber) + '.txt'

        FileSave(fileNameSave, "{},{},{},{},{},{},{},{} \n".format('Tspan [K]', 'Qh [W]', 'Qc [W]', 'Cycles [-]', 'Run time [min]', 'Max. Pressure drop [Pa]', 'Thot [K]', 'Tcold [K]'))
        FileSave(fileNameSave, "{},{:7.4f},{:7.4f},{},{:7.4f},{:7.4f},{},{} \n".format(results[0]-results[1], results[26], results[2], results[27], results[4], results[17], Thot, Tcold))
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

        caseNumber    = caseNumber + 1
