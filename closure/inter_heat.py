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

    import matplotlib.pyplot as plt

    from sourcefiles.fluid.density import fRho
    from sourcefiles.fluid.dynamic_viscosity import fMu
    from sourcefiles.fluid.conduction import fK
    from sourcefiles.fluid.specific_heat import fCp

    T_fluid = 280  # [K] Temperature of fluid
    perc_gly = 20  # [-] Percentage of glycol in the mixture
    Vol_flow_rate = 30.52e-6  # [m3/s] Volumetric flow rate
    Area_cross = 0.045*0.013  # [m2] Cross sectional area of bed
    fAMR = 1.7  # [Hz]
    Gd_K = 14  # [W/(m*K)]
    Gd_rho = 7900  # [kg/m3]
    Gd_cp = 240  # [J/(kg*K)]

    Dsphere = [400e-6, 500e-6, 600e-6, 700e-6, 800e-6]  # [m] Diameter of sphere
    Porosity = [0.3, 0.35, 0.4, 0.45, 0.5]   # [] Porosity of bed
    hI_beta = np.ones((len(Porosity), len(Dsphere)))
    hE_beta = np.ones((len(Porosity), len(Dsphere)))
    hM_beta = np.ones((len(Porosity), len(Dsphere)))
    hM_beta_DF = np.ones((len(Porosity), len(Dsphere)))

    rho_fluid = fRho(T_fluid, perc_gly)
    mu_fluid = fMu(T_fluid, perc_gly)
    cp_fluid = fCp(T_fluid,perc_gly)
    k_fluid = fK(T_fluid,perc_gly)
    Ud = Vol_flow_rate/Area_cross

    x = 0

    for er in Porosity:
        y = 0
        for Dsp in Dsphere:

            beta = 6 * (1 - er) / Dsp

            hI_beta[x, y] = beHeff_I(Dsp, Ud, cp_fluid, k_fluid, mu_fluid, rho_fluid, fAMR, Gd_cp, Gd_K, Gd_rho, er)
            hE_beta[x, y] = beHeff_E(Dsp, Ud, cp_fluid, k_fluid, mu_fluid, rho_fluid, fAMR, Gd_cp, Gd_K, Gd_rho, er)

            # Heat transfer coefficient calculated by using the correlation presented in the book of Mills

            Vel_char = Ud / er
            Len_char = Dsp * (er / (1 - er))
            Re = Vel_char * Len_char * rho_fluid / mu_fluid
            Pr = cp_fluid * mu_fluid / k_fluid
            Nu = (0.5 * Re ** 0.5 + 0.2 * Re ** (0.2e1/0.3e1)) * Pr ** (0.1e1 / 0.3e1)
            hM_beta[x, y] = (Nu * k_fluid / Len_char) * beta

            Biot = Nu * k_fluid / (2 * Gd_K)
            Fourier = 4 * Gd_K / (Gd_rho * Gd_cp * fAMR * Dsp ** 2)
            phi_H = 1 - 4 / (35 * Fourier)
            DF = 1 / (1 + Biot / 5 * phi_H)
            hM_beta_DF[x, y] = (Nu * k_fluid / Len_char) * DF * beta

            y = y + 1

        x = x + 1

    #print("hE = {} [W/(m2*K)], hI = {} [W/(m2*K)], hM = {} [W/(m2*K)]".format(hE, hI, hM))
    # print(hI)
    # print(hM)
    fig1 = plt.figure(1)
    plt.plot(Dsphere, hI_beta)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Porosity = 0.3', 'Porosity = 0.35', 'Porosity = 0.4', 'Porosity = 0.45', 'Porosity = 0.5'])
    plt.xlabel("Diameter of spheres [m]")
    plt.ylabel("h*beta [W/($m^3$*K)]")
    plt.title("Wakao & Kagei (1982) Correlation")
    plt.grid(which='major',axis='both')

    fig2 = plt.figure(2)
    plt.plot(Dsphere, hM_beta)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Porosity = 0.3', 'Porosity = 0.35', 'Porosity = 0.4', 'Porosity = 0.45', 'Porosity = 0.5'])
    plt.xlabel("Diameter of spheres [m]")
    plt.ylabel("h*beta [W/($m^3$*K)]")
    plt.title("Whitaker (1972) Correlation")
    plt.grid(which='major',axis='both')

    fig3 = plt.figure(3)
    plt.plot(Dsphere, hM_beta_DF)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Porosity = 0.3', 'Porosity = 0.35', 'Porosity = 0.4', 'Porosity = 0.45', 'Porosity = 0.5'])
    plt.xlabel("Diameter of spheres [m]")
    plt.ylabel("h*beta [W/($m^3$*K)]")
    plt.title("Correlation in Mills's book with Degradation Factor (DF)")
    plt.grid(which='major',axis='both')

    plt.show()
