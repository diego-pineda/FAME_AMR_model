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
    # Iman uses the DF factor, and a Wakao Nusselt.
    #
    # The decomposition of the hefff is as follows:
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


if __name__ == '__main__':

    from sourcefiles.fluid.density import fRho
    from sourcefiles.fluid.dynamic_viscosity import fMu
    from sourcefiles.fluid.conduction import fK
    from sourcefiles.fluid.specific_heat import fCp

    T_fluid = 280  # [K] Temperature of fluid
    perc_gly = 20  # [-] Percentage of glycol in the mixture
    Dsp = 400e-6  # [m] Diameter of sphere
    Vol_flow_rate = 30.52e-6  # [m3/s] Volumetric flow rate
    Area_cross = 0.045*0.013  # [m2] Cross sectional area of bed
    Porosity = 0.36  # [] Porosity of be
    fAMR = 1.7  # [Hz]
    Gd_K = 14  # [W/(m*K)]
    Gd_rho = 7900  # [kg/m3]
    Gd_cp = 240  # [J/(kg*K)]

    rho_fluid = fRho(T_fluid, perc_gly)
    mu_fluid = fMu(T_fluid, perc_gly)
    cp_fluid = fCp(T_fluid,perc_gly)
    k_fluid = fK(T_fluid,perc_gly)
    Ud = Vol_flow_rate/Area_cross

    beta = 6 * (1 - Porosity) / Dsp
    # Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er
    hI = beHeff_I(Dsp, Ud, cp_fluid, k_fluid, mu_fluid, rho_fluid, fAMR, Gd_cp, Gd_K, Gd_rho, Porosity)/beta
    hE = beHeff_E(Dsp, Ud, cp_fluid, k_fluid, mu_fluid, rho_fluid, fAMR, Gd_cp, Gd_K, Gd_rho, Porosity)/beta

    # Heat transfer coefficient calculated by using the correlation presented in the book of Mills

    Vel_char = Ud / Porosity
    Len_char = Dsp * (Porosity / (1 - Porosity))
    Re = Vel_char * Len_char * rho_fluid / mu_fluid
    Pr = cp_fluid * mu_fluid / k_fluid
    Nu = (0.5 * Re ** 0.5 + 0.2 * Re ** (0.2e1/0.3e1)) * Pr ** (0.1e1 / 0.3e1)
    hM = Nu * k_fluid / Len_char

    print("hE = {} [W/(m2*K)], hI = {} [W/(m2*K)], hM = {} [W/(m2*K)]".format(hE, hI, hM))
