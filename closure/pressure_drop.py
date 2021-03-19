

import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np

# This is the pressure drop term
@jit(nb.types.Tuple((f8, f8))(f8, f8, f8, f8, f8,  f8,  f8),nopython=True)
def SPresM(Dsp, Ud, V, er, flMu, flRho, Af):
    # The following corresponds to Ergun's correlation for calculation of pressure drop as presented in Mills
    dP = (1.75 * Ud ** 2 * (1 - er) / Dsp / er ** 3 * flRho + 150 * Ud * (1 - er) ** 2 / Dsp ** 2 / er ** 3 * flMu)
    Spress = dP * V
    return Spress, dP

# DP: It seems that V is volumetric flow rate. In that case, Spress would be the viscous disipation
# DP: dP/dx is defined according to the correlation presented in Mills end of chapter 4 table

# DP: I could not find this particular correlation in the book of heat transfer in porous media.
# I found a similar correlation with different coefficients, 4 instead of 1.75 and 180 instead of 150.


# ---- Following lines for testing the function and getting an idea of the order of magnitude of the pressure drop
if __name__ == '__main__':

    from sourcefiles.fluid.density import fRho
    from sourcefiles.fluid.dynamic_viscosity import fMu

    T_fluid = 280  # [K] Temperature of fluid
    perc_gly = 20  # [-] Percentage of glycol in the mixture
    D_sphere = 400e-6  # [m] Diameter of sphere
    Vol_flow_rate = 30.52e-6  # [m3/s] Volumetric flow rate
    Area_cross = 0.045*0.013  # [m2] Cross sectional area of bed
    Porosity = 0.36  # [] Porosity of be

    rho_fluid = fRho(T_fluid, perc_gly)
    mu_fluid = fMu(T_fluid, perc_gly)
    pressure_drop = SPresM(D_sphere, Vol_flow_rate/Area_cross, Vol_flow_rate, Porosity, mu_fluid, rho_fluid, 1)
    print('Pumping power = {} [W] \nPressure drop = {} [Pa]'.format(pressure_drop[0], pressure_drop[1]*0.12))

    import matplotlib.pyplot as plt
    diameters = [400e-6, 500e-6, 600e-6, 700e-6, 800e-6]
    porosities = [0.3, 0.35, 0.4, 0.45, 0.5]
    press_drop = np.ones((len(porosities), len(diameters)))

    b = 0
    for j in porosities:
        a = 0
        for i in diameters:
            #print(SPresM(i, Vol_flow_rate/Area_cross, Vol_flow_rate, j, mu_fluid, rho_fluid, 1)[1])
            press_drop[b, a] = SPresM(i, Vol_flow_rate/Area_cross, Vol_flow_rate, j, mu_fluid, rho_fluid, 1)[1]*0.060
            a = a+1
        b = b+1

    fig = plt.figure()
    plt.plot(diameters, press_drop)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Porosity = 0.3', 'Porosity = 0.35', 'Porosity = 0.4', 'Porosity = 0.45', 'Porosity = 0.5'])
    plt.xlabel("Diameter of spheres [m]")
    plt.ylabel("Pressure drop along regenerator [Pa]")
    plt.title("Pressure drop in packed bed using Ergun's correlation")
    plt.show()
