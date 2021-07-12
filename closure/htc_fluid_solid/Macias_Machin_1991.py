from numba import jit, f8, int32, b1
import numpy as np

# Heat transfer coefficient between fluid and solid for a packed bed of spherical particles

# Based on the correlation proposed by Macias-Machin, 1991.
# Nu = 2 + 0.11 Re^0.6 Pr ^ (1/3)

# A degradation factor, DF, is also included to take into account the temperature gradients inside the solid.
# See paper: K.L. Engelbrecht, G.F. Nellis, and S.A. Klein, “The effect of internal temperature gradients on regenerator matrix performance,” 2006.

# The decomposition of the hefff is as follows:
# Nu_sp = 2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)
# Biot = Nu_sp * fK / (2 * mK)
# Fourier = 4 * mK / (mRho * mCp * freq * Dsp ** 2)
# phi_H = 1 - 4 / (35 * Fourier)
# DF = 1 / (1 + Biot / 5 * phi_H)
# hefff = Nu_sp * fK / Dsp * DF


@jit(f8     (f8 , f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8),nopython=True)
def beHeff(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):
    # Heat transfer coefficient based on the correlation developed by Macias-Machin (1991)
    hefff = (1.27 + 2.66 * (fRho * Ud * Dsp / fMu / er) ** 0.56e0 * (fMu * fCp / fK) ** (-0.41) * ((1 - er) / er) ** 0.29) * fK / Dsp / (1 + (1.27 + 2.66 * (fRho * Ud * Dsp / fMu / er) ** 0.56 * (fMu * fCp / fK) ** (-0.41) * ((1 - er) / er) ** 0.29) * fK / mK / 10 * (1 - mRho * mCp * freq * Dsp ** 2 / mK / 35))
    beta = 6 * (1 - er) / Dsp  # [m2/m3] specific surface area of a packed bed of spheres
    return hefff * beta