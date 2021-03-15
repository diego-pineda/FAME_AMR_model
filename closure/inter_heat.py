import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np


# Engelbrechts Correlation for packed beds suggeted used by Lei.
@jit(f8     (f8 , f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8),nopython=True)
def beHeff_E(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):
    hefff = (0.7*(fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** 0.23) * fK / Dsp
    beta  = 6 * (1 - er) / Dsp
    return hefff*beta


# Beta*Heff based on the work of Iman.
@jit(f8     (f8 , f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8),nopython=True)
def beHeff_I(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):
    # Iman uses the DF factor, and a Wakao Nusselt. The decomposition of the hefff is as follows:
    # Nu_sp = 2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)
    # Biot = Nu_sp * fK / (2 * mK)
    # Fourier = 4 * mK / (mRho * mCp * freq * Dsp ** 2)
    # phi_H = 1 - 4 / (35 * Fourier)
    # DF = 1 / (1 + Biot / 5 * phi_H)
    # hefff = Nu_sp * fK / Dsp * DF
    # DP: I think Theo avoids creating as many variables as above to save computing memory
    hefff = (2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)) * fK / Dsp / (1 + (2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)) * fK / mK / 10 * (1 - mRho * mCp * freq * Dsp ** 2 / mK / 35))
    beta = 6 * (1 - er) / Dsp  # [m2/m3] specific surface area of a packed bed of spheres
    return hefff*beta



