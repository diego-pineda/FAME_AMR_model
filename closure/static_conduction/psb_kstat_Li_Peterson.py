import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np


# Static conduction for a packed screen bed by Li and Peterson 2006
# See eq. 24 in: C. Li, G.P. Peterson. International Journal of Heat and Mass Transfer 49 (2006) 4095–4105
@jit(f8(  f8, f8, f8, f8), nopython=True)
def kStat(er, fK, mK, d):

    M = np.sqrt(np.sqrt((64 * er**2 - 128 * er + np.pi**2 + 64) * d**4)/(np.pi * d**4) - 1/d**2) / np.sqrt(2)

    return (M*d)**2*mK + (1-M*d)**2*fK + 2*M*d*(1-M*d) / ((np.pi * np.sqrt(1+(M*d)**2) / 8) / mK + (1 - np.pi * np.sqrt(1+(M*d)**2) / 8) / fK)
# mK is the thermal conductivity of the MCM, independent of temperature, which is defined in the configuration files
# fK is thermal conductivity of the fluid, which is a function of temperature
# er is the porosity
# d is wire diameter

