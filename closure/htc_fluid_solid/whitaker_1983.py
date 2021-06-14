from numba import jit, f8, int32, b1
import numpy as np

# Heat transfer coefficient between fluid and solid for a packed bed of spherical particles

# Based on the correlation proposed by Whitaker, 1983. (Kaviany section 7.2.2)
# Nu = 2 + [ 0.4 Re ^ (1/2) + 0.2 Re ^ (2/3) ] Pr ^ (0.4)
# Reynolds number based on pore velocity and hydraulic diameter

# A degradation factor, DF, is also included to take into account the temperature gradients inside the solid.
# See paper: K.L. Engelbrecht, G.F. Nellis, and S.A. Klein, “The effect of internal temperature gradients on regenerator matrix performance,” 2006.

# The decomposition of the hefff is as follows:
# Nu_sp = 2 + [ 0.4 * (fRho * (Ud / er) * (er / (1-er)) * Dsp / fMu) ** (1/2) + 0.2 * (fRho * (Ud / er) * (er / (1-er)) * Dsp / fMu) ** (2/3) ] * (fMu * fCp / fK) ** (0.4)
# Biot = Nu_sp * fK * (1 - er) / (2 * mK * er)
# Fourier = 4 * mK / (mRho * mCp * freq * Dsp ** 2)
# phi_H = 1 - 4 / (35 * Fourier)
# DF = 1 / (1 + Biot / 5 * phi_H)
# hefff = Nu_sp * fK / Dsp * DF


@jit(f8     (f8 , f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8),nopython=True)
def beHeff(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):

    hefff = (2 + (0.4e0 * (fRho * (Ud / er) * Dsp * (er / (1 - er)) / fMu) ** (0.1e1 / 0.2e1) + 0.2e0 * (fRho * (Ud / er) * Dsp * (er / (1 - er)) / fMu) ** (0.2e1 / 0.3e1)) * (fMu * fCp / fK) ** (0.4e0)) * fK / (Dsp * (er / (1 - er))) / (1 + (2 + (0.4e0 * (fRho * (Ud / er) * Dsp * (er / (1 - er)) / fMu) ** (0.1e1 / 0.2e1) + 0.2e0 * (fRho * (Ud / er) * Dsp * (er / (1 - er)) / fMu) ** (0.2e1 / 0.3e1)) * (fMu * fCp / fK) ** (0.4e0)) * fK * ((1 - er) / er) / mK / 10 * (1 - mRho * mCp * freq * Dsp ** 2 / mK / 35))
    beta = 6 * (1 - er) / Dsp  # [m2/m3] specific surface area of a packed bed of spheres
    return hefff * beta