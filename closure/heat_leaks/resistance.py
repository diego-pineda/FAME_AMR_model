
import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np


# Comment by DP: this is the calculation of overall heat transfer coefficient U in terms of the internal area defined by r1
# rather than just the thermal resistance

################# The following functions are the ones for the PM1 device ################

# This calculates the thermal resistance for the regenerator.
@jit(f8(              f8,  f8,  f8,   f8,   f8, f8,   f8, f8, f8, f8,        f8,   f8, f8),nopython=True)
def ThermalResistance(Dsp, Ud, fMu, fRho, kair, kf, kg10, r1, r2, r3, casing_th, fAMR, air_th):
    # Overall htc for the case of a Couette flow between AMR casing and magnets of cylindrical shape
    Ucouette = 0.1e1 / (0.5882352941e1 * (fRho * np.abs(Ud) * Dsp / fMu) ** (-0.79e0) / kf * Dsp + 0.1e1 / kg10 * r1 * np.log(r2 / r1) + 0.1e1 / kair * r1 * np.log(r3 / r2))


    return Ucouette


# This calculates the thermal resistance for the void space.
@jit(f8(                  f8,  f8,  f8,   f8,   f8, f8, f8, f8),nopython=True)
def ThermalResistanceVoid(kair, kf, kg10, kult, r0, r1, r2, r3):
    Ucyl = 0.1e1 / (0.4587155963e0 / kf * r0 + 0.1e1 / kult * r0 * np.log(r1 / r0) + 0.1e1 / kg10 * r0 * np.log(r2 / r1) + 0.1e1 / kair * r0 * np.log(r3 / r2))
    return Ucyl

# DP: the coefficient 0.4587... results from dividing 2/4.36, which comes from Nu=4.36=h*Dh/k
# Constant heat flux is assumed (see table 8.1 of Incropera for circular tube)
#######################################################################################
