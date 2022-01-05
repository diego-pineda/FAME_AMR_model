import numba as nb
from numba import jit, f8, int32,b1
import numpy as np
import matplotlib.pyplot as plt

# Fluid Density
percGly=20
@jit(f8 (f8 ,f8 ),nopython=True)
def fRho(Tref,percglyvol=percGly): #
    if(Tref<230):
        return 1000
    else:
        return (-0.517378e3 + 0.143157e2 * Tref - 0.4405e-1 * Tref ** 2 + 0.4384e-4 * Tref ** 3) * (0.1088984040e1 - 0.10227e-1 * Tref + 0.31258e-4 * Tref ** 2 - 0.32614e-7 * Tref ** 3 - 0.2790099948e-3 * percglyvol + 0.2743325832e-6 * percglyvol ** 2 - 0.60141e-3 * (0.6336420000e-3 + 0.1107710000e-1 * percglyvol - 0.1089140000e-4 * percglyvol ** 2) ** 2) / (0.9999884348e0 - 0.9426e-2 * Tref + 0.28927e-4 * Tref ** 2 - 0.30449e-7 * Tref ** 3 - 0.2021792292e-3 * percglyvol + 0.1987898328e-6 * percglyvol ** 2)


if __name__ == '__main__':
    Temperatures = np.linspace(273, 323, 51)
    gly_perc = np.linspace(1, 20, 20)
    mass_flow_rate = np.array([0.016667])  #np.linspace(0.1, 2, 20)
    legends = []
    dens = np.zeros((len(gly_perc), len(Temperatures)))
    i = 0
    for g in gly_perc:
        j = 0
        for T in Temperatures:
            dens[i, j] = fRho(T, g)
            j = j+1
        plt.plot(Temperatures, dens[i, :])
        legends.append("{}".format(g))
        i = i+1

    plt.legend(legends, title="Glycol [%]", ncol=3)
    plt.grid(True)
    plt.xlabel("T [K]")
    plt.ylabel(r'$\rho$ [kg/m$^3$]')
    # plt.show()
    # print(len(dens))

    # For info on text in matplotlib go to: https://matplotlib.org/stable/tutorials/text/text_intro.html

    X, Y = np.meshgrid(Temperatures, gly_perc)
    vol_flow_rate = []
    fig, ax = plt.subplots(len(mass_flow_rate))

    for i in range(len(mass_flow_rate)):
        vol_flow_rate.append(mass_flow_rate[i] / 16.667e-6 / dens)
        Z = mass_flow_rate[i] / 16.667e-6 / dens
        # print(len(Z))
        CS = ax.contourf(X, Y, Z, levels=100, cmap='jet')
        # # Set level to np.amax(Z) if desired that colorbar cover all values of Z only, otherwise something like np.linspace(0, 10, 100)
        plt.colorbar(mappable=CS, aspect=10)

    # print(mass_flow_rate)
    plt.show()


