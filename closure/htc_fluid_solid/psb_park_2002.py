from numba import jit, f8, int32, b1
import numpy as np

# Heat transfer coefficient between fluid and solid for a packed screen bed

# Based on the correlation proposed by Park et al, 2002.
# Nu = 1.315 Re^0.35 Pr ^ (1/3) * ((1-er)/er)^0.2

@jit(f8    (f8, f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8), nopython=True)
def beHeff(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):
    # Heat transfer coefficient based on the correlation developed by Park et al. (2002)
    dl = (er/(1-er)) * Dsp
    Re = fRho * Ud * dl / fMu / er
    Nu = 1.315 * (fMu * fCp / fK) ** (1/3) * Re ** 0.35 * ((1 - er)/er) ** 0.2
    hefff = Nu * fK / dl
    beta = 4 * (1 - er) / Dsp
    return hefff * beta