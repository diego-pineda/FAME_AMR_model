###### Volumetric flow rate ######

# General Assumptions

# 1) The volumetric flow rate is uniform in the direction of flow
# 2) The volumetric flow rate is uniform in any cross sectional area perpendicular to the flow
# 3) There is an acceleration period from stagnation to maximum flow, which is assumed linear
# 4) There is a deceleration period form maximum flow to stagnation, which is assumed linear
# 5) The flow periods are controlled by opening and closing solenoid valves, as described by Huang et al. 2019

import numpy as np

# acc_period = 10  # [°] angular duration of acceleration and deceleration
# max_flow_period = 45  # [°] angular duration of max flow
# full_magn_ang = 30  # [°] full magnetization angle or angle at which maximum flow is reached


def vol_flow_rate(nt, v_max, acc_period, max_flow_period, full_magn_ang, unbal_rat):
    vol_rate = np.zeros(nt+1)
    angle = 0
    for n in range(0, nt+1):
        if 0 <= angle < full_magn_ang - acc_period:
            vol_rate[n] = 0  # [m^3/s] Volumetric flow rate during stagnation period
        elif full_magn_ang - acc_period <= angle < full_magn_ang:
            vol_rate[n] = (v_max / acc_period) * (180 * (n / nt) - (full_magn_ang - acc_period))
            # [m^3/s] Vol flow rate increase during magnetization
        elif full_magn_ang <= angle < full_magn_ang + max_flow_period:
            vol_rate[n] = v_max  # [m^3/s] Volumetric flow rate during the cold to hot blow process
        elif full_magn_ang + max_flow_period <= angle < full_magn_ang + max_flow_period + acc_period:
            vol_rate[n] = (v_max / acc_period) * (full_magn_ang + max_flow_period + acc_period - 180 * n / nt)
            # [m^3/s] Vol flow rate decrease during the demagnetization
        elif full_magn_ang + max_flow_period + acc_period <= angle < 90 + full_magn_ang - acc_period:
            vol_rate[n] = 0  # [m^3/s] Volumetric flow rate during stagnation period
        elif 90 + full_magn_ang - acc_period <= angle < 90 + full_magn_ang:
            vol_rate[n] = (v_max * unbal_rat / acc_period)*(90 + full_magn_ang - acc_period - 180 * n / nt)
            # [m^3/s] Volumetric flow rate increase towards hot to cold blow
        elif 90 + full_magn_ang <= angle < 90 + full_magn_ang + max_flow_period:
            vol_rate[n] = -v_max * unbal_rat  # [m^3/s] Volumetric flow rate during the hot to cold blow process
        elif 90 + full_magn_ang + max_flow_period <= angle < 90 + full_magn_ang + max_flow_period + acc_period:
            vol_rate[n] = (v_max * unbal_rat / acc_period) * (180 * (n / nt) - (90 + full_magn_ang + max_flow_period + acc_period))
            # [m^3/s] Vol flow rate decrease during hot to cold blow
        else:
            vol_rate[n] = 0  # [m^3/s] Volumetric flow rate during stagnation period
        angle = 180 * (n + 1) / nt
    return vol_rate

########## The following lines are for testing the function ##########


if __name__ == '__main__':

    # Plotting magnetic field profile to visualize along with volumetric flow rate profile
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

    from sourcefiles.device.FAME_app_field import app_field

    nt = 400 # number of nodes in the time domain
    N = 1 # number of nodes in the spatial domain. This is just for the applied field function
    mag_field = 1.4

    ap_field = app_field(nt, N, mag_field)
    print(ap_field)

    fig, plot1 = plt.subplots()
    plot1.plot(np.linspace(0, 1, nt+1), ap_field[:, 0], '-', color='red')
    plot1.set_xlabel("Fraction of AMR cycle period [-]")
    plot1.set_ylabel("Applied magnetic field [T]", color='red')
    plot1.set_ylim([-1.5, 1.5])
    plot1.set_xlim([0, 1])
    plot1.set_yticks([-1.4, -0.7, 0, 0.7, 1.4])


    # plot1.set_xticks(np.arange(0, 1, 10))
    # plt.figure(1)
    # plt.plot(np.linspace(0, 180, nt+1), ap_field[:, 0], '-')
    # plt.xlabel("Angle [°]")
    # plt.ylabel("Applied field [T]")
    # plt.title("Applied magnetic field as a function of rotation angle")
    # plt.grid(which='both', axis='both')
    # plot1.xaxis.set_major_locator(MultipleLocator(15))
    # plt.show()



    # nt = 600 # number of nodes in the time domain
    V_rate = 1  # [m^3/s] Maximum volumetric flow rate. Set to 1 for considering all bed and constant flow rate from pump
    acc_period = 5  # [°] angular duration of acceleration and deceleration
    max_flow_period = 45  #360/7-20  # [°] angular duration of max flow
    full_magn_ang = 30  # [°] full magnetization angle or angle at which maximum flow is reached
    unbal_rat = 1  # []

    volumetric_rate = vol_flow_rate(nt, V_rate, acc_period, max_flow_period, full_magn_ang, unbal_rat)
    # print(np.abs(volumetric_rate[25] / (0.045*0.013)))

    # plot2 = plt.figure(1)
    plot2 = plot1.twinx()
    plot2.plot([0.9, 1], [0, 0], '-', color='red')  # This is just for the legend
    plot2.plot(np.linspace(0, 1, (nt+1)), volumetric_rate, '--', color='blue')
    plot2.set_ylabel("Normalized flow rate [-]", color='blue')
    plt.tight_layout()
    # plt.plot(np.linspace(0, 180, (nt+1)), volumetric_rate, '--')
    # plt.xlabel("Angle [°]")
    # plt.ylabel("Normalized flow rate [-] / Applied field [T]")
    # # plt.title("Normalized flow rate and applied field as function of rotation angle")
    plt.legend(['Applied magnetic field', 'Normalized flow rate'])
    # plt.xticks(np.arange(0, 181, 15))  # [30, 60, 75, 90, 105, 120, 150, 165, 180]
    # plt.grid(which='both', axis='both')
    #plt.minorticks_on()
    #plot1.xaxis.set_major_locator(MultipleLocator(10))
    # plt.show()


    # The following lines are for plotting flow rate profiles of 7 regenerators each in its one phase

    reg1twocycles = np.append(volumetric_rate[:], volumetric_rate[:]) # One full rotation of the device
    reg2ind = int(np.floor(1 * 2 * (nt + 1) / 7))
    reg2twocycles = np.concatenate((volumetric_rate[-reg2ind:], volumetric_rate[:], volumetric_rate[:-reg2ind]), axis=None)
    reg3ind = int(np.floor(2 * 2 * (nt + 1) / 7))
    reg3twocycles = np.concatenate((volumetric_rate[-reg3ind:], volumetric_rate[:], volumetric_rate[:-reg3ind]), axis=None)
    reg4ind = int(np.floor(3 * 2 * (nt + 1) / 7))
    reg4twocycles = np.concatenate((volumetric_rate[-reg4ind:], volumetric_rate[:], volumetric_rate[:-reg4ind]), axis=None)
    reg5ind = int(np.floor(4 * 2 * (nt + 1) / 7))
    reg5twocycles = np.concatenate((reg1twocycles[-reg5ind:], reg1twocycles[:-reg5ind]), axis=None)
    reg6ind = int(np.floor(5 * 2 * (nt + 1) / 7))
    reg6twocycles = np.concatenate((reg1twocycles[-reg6ind:], reg1twocycles[:-reg6ind]), axis=None)
    reg7ind = int(np.floor(6 * 2 * (nt + 1) / 7))
    reg7twocycles = np.concatenate((reg1twocycles[-reg7ind:], reg1twocycles[:-reg7ind]), axis=None)
    # print(reg2ind,reg3ind,reg4ind,reg5ind,reg6ind,reg7ind)
    # print(reg1twocycles,reg5twocycles)

    fig2, plot3 = plt.subplots()

    plt.plot(np.linspace(0, 360, 2*(nt+1)), reg1twocycles)
    # plt.plot(np.linspace(0, 360, 2*(nt+1)), reg2twocycles)
    # plt.plot(np.linspace(0, 360, 2*(nt+1)), reg3twocycles)
    # plt.plot(np.linspace(0, 360, 2*(nt+1)), reg4twocycles)
    # plt.plot(np.linspace(0, 360, 2*(nt+1)), reg5twocycles)
    # plt.plot(np.linspace(0, 360, 2*(nt+1)), reg6twocycles)
    # plt.plot(np.linspace(0, 360, 2*(nt+1)), reg7twocycles)

    plt.xlabel("Angle [°]")
    plt.ylabel("Volumetric flow rate [$m^{3}/s$]")
    plt.title("Volumetric flow rate as a function of rotation angle")
    plt.grid(which='both', axis='both')
    plot3.xaxis.set_major_locator(MultipleLocator(15))
    # plt.show()

    # Recalculating the flow profile

    V_pump = 19.67e-6  # [m3/s] (Group B = 43.17e-6)
    f_AMR = 0.5  # [Hz]
    dt = (1 / f_AMR) / nt

    reg1_vol_flow = np.ones(2*(nt+1))
    Vtotal = np.ones(2*(nt+1))
    vol_per_reg = np.ones(2*(nt+1))
    disp_vol = 0


    for i in range(2*(nt+1)):
        Vtotal[i] = max(0,reg1twocycles[i])+max(0,reg2twocycles[i])+max(0,reg3twocycles[i])+max(0,reg4twocycles[i])+max(0,reg5twocycles[i])+max(0,reg6twocycles[i])+max(0,reg7twocycles[i])
        vol_per_reg[i] = V_pump / Vtotal[i]
        reg1_vol_flow[i] = max(0,reg1twocycles[i]) * vol_per_reg[i]
        rect_area = reg1_vol_flow[i]*dt
        disp_vol = disp_vol + rect_area

    rel_error = 1
    V_reg_guess = 11e-6
    iteration = 0

    while np.abs(rel_error) > 0.00005:
        disp_vol2 = 0
        for i in range(2*(nt+1)):
            rect_area2 = V_reg_guess * max(0, reg1twocycles[i]) * dt
            disp_vol2 = disp_vol2 + rect_area2
        rel_error = (disp_vol2-disp_vol)/disp_vol
        V_reg_guess = V_reg_guess * (1-rel_error)
        iteration = iteration + 1
        print(iteration,rel_error,V_reg_guess)
    V_reg_calc = V_reg_guess

    print(disp_vol/2,disp_vol2/2, V_reg_calc)

    plt.figure(3)
    plt.plot(np.linspace(0, 360, 2*(nt+1)), reg1_vol_flow)
    plt.xlabel("Angle [°]")
    plt.ylabel("Normalized flow rate [-]")
    plt.title("Flow rate one direction / one regenerator / constant pump flow rate")
    plt.grid(which='both', axis='both')
    plt.show()





# ----The next lines correspond to the function that Theo implemented to describe the volumetric flow rate vs time----

# This is just for comparison

# import matplotlib.pyplot as plt
# import sys
#
# nt = 100
# freq = 1
# t = np.linspace(0,1/freq,nt+1)
# Ac = np.pi*0.008**2
# Vd = 3.91e-6
#
# vf = lambda at, Ac, Vd, sf: (Vd) * sf * np.pi * np.sin(2 * np.pi * sf * at) + np.sign(np.sin(2 * np.pi * sf * at))*sys.float_info.epsilon*2
# V = vf(t, Ac, Vd, freq)
#
# plot3 = plt.figure(3)
# plt.plot(np.linspace(0,360,nt+1),V[:])
# plt.xlabel("Angle [°]")
# plt.ylabel("Volumetric flow rate [$m^{3}/s$]")
# plt.title("Volumetric flow rate as a function of rotation angle - Victoria's Model")
# plt.grid(which='both', axis='both')
# plt.show()