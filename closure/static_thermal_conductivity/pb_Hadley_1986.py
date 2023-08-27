import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np


# Static conduction by Hadley 1986
@jit(f8(  f8, f8, f8, f8), nopython=True)
def kStat(er, fK, mK, Dsp):
    return fK * ((1 - 10 ** (0.935844e0 - 0.6778e1 * er)) * (er * (0.8e0 + 0.1e0 * er) + (-er * (0.8e0 + 0.1e0 * er) + 1) * mK / fK) / (1 - er * (0.2e0 - 0.1e0 * er) + mK / fK * er * (0.2e0 - 0.1e0 * er)) + 10 ** (0.935844e0 - 0.6778e1 * er) * (2 * mK ** 2 / fK ** 2 * (1 - er) + (1 + 2 * er) * mK / fK) / ((2 + er) * mK / fK + 1 - er))
# Dsp was added (25/08/2023) because the correlation for the PSB required wire diameter. All functions for calculating
# effective thermal conductivity of the bed, aka static thermal conductivity, must have the same input parameters.
# mK is the thermal conductivity of the MCM, independent of temperature, which is defined in the configuration files
# fK is thermal conductivity of the fluid, which is a function of temperature