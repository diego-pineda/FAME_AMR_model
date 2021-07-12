import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np

# Engelbrechts Correlation for packed beds
@jit(f8    (f8, f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8),nopython=True)
def beHeff(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):
    hefff = (0.7*(fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** 0.23) * fK / Dsp
    beta  = 6 * (1 - er) / Dsp
    return hefff*beta