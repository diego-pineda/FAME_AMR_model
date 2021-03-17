###### Volumetric flow rate ######

# General Assumptions

# 1) The volumetric flow rate is uniform in the direction of flow
# 2) The volumetric flow rate is uniform in any cross sectional area perpendicular to the flow
# 3) There is an acceleration period from stagnation to maximum flow, which is assumed linear
# 4) There is a deceleration period form maximum flow to stagnation, which is assumed linear
# 5) The flow periods are controlled by opening and closing solenoid valves, as described by Huang et al. 2019

import numpy as np

acceleration_period = 10  # [°]
max_flow_period = 45  # [°]
full_magnetization_angle = 30  # [°]


def vol_flow_rate(nt, v_max):
    vol_rate = np.zeros(nt+1)
    angle = 0
    for n in range(0, nt+1):
        if 0 <= angle < 20:
            vol_rate[n] = 0  # [m^3/s] Volumetric flow rate during stagnation period
        elif 20 <= angle < 30:
            vol_rate[n] = (v_max / 10) * (180 * (n / nt)-20)  # [m^3/s] Vol flow rate increase during magnetization
        elif 30 <= angle < 75:
            vol_rate[n] = v_max  # [m^3/s] Volumetric flow rate during the cold to hot blow process
        elif 75 <= angle < 85:
            vol_rate[n] = (v_max / 10) * (85 - 180 * n / nt)  # [m^3/s] Vol flow rate decrease during the demagnetizatn
        elif 85 <= angle < 110:
            vol_rate[n] = 0  # [m^3/s] Volumetric flow rate during stagnation period
        elif 110 <= angle < 120:
            vol_rate[n] = (v_max/10)*(110-180*n/nt)  # [m^3/s] Volumetric flow rate increase towards hot to cold blow
        elif 120 <= angle < 165:
            vol_rate[n] = -v_max  # [m^3/s] Volumetric flow rate during the hot to cold blow process
        elif 165 <= angle < 175:
            vol_rate[n] = (v_max / 10) * (180 * (n / nt)-175)  # [m^3/s] Vol flow rate decrease during hot to cold blow
        else:
            vol_rate[n] = 0  # [m^3/s] Volumetric flow rate during stagnation period
        angle = 180 * (n + 1) / nt
    return vol_rate

########## The following lines are for testing the function ##########


if __name__ == '__main__':

    nt = 600 # number of nodes in the time domain
    V_rate = 15.33e-6 # [m^3/s] Maximum volumetric flow rate

    volumetric_rate = vol_flow_rate(nt, V_rate)
    # print(np.abs(volumetric_rate[25] / (0.045*0.013)))

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
    print(reg2ind,reg3ind,reg4ind,reg5ind,reg6ind,reg7ind)
    print(reg1twocycles,reg5twocycles)


    import matplotlib.pyplot as plt
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

    fig, plot2 = plt.subplots()

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
    plot2.xaxis.set_major_locator(MultipleLocator(15))
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