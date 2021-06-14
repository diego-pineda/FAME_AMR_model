import numba as nb
from numba import jit, f8, int32, b1
import numpy as np

# TODO: include ambient temperature, relative humidity and total pressure as input parameters
# TODO: find correlations for the calc. of transport properties of air as function of temp., rel humid and pressure

rho_air = 1.184  # [kg/m3] Density of air at 298 [K] and 1 [atm] pressure
Mu_air = 1.849e-5  # [kg/(m*s)] Dynamic viscosity of air at 298 [K] and 1 [atm] pressure
Pr_air = 0.7296  # [-] Prandlt number of air at 298 [K] and 1 [atm] pressure
W_reg = 0.045  # [m] Width of regenerator. This is the characteristic length on the external side of regenerator

@jit(f8(               f8, f8,  f8,   f8,   f8, f8,   f8, f8, f8, f8,        f8,   f8, f8),nopython=True)
def ThermalResistance(Dsp, Ud, fMu, fRho, kair, kf, kg10, r1, r2, r3, casing_th, fAMR, air_th):

    u_air = 0.1075 * np.pi * fAMR  # velocity of air. The velocity of the center of the magnet was considered r=0.1075 m
    # Laminar flow between parallel plates
    film_th = air_th
    Re_film = u_air * 2 * film_th / (Mu_air / rho_air)
    Nu_D = 7.54 + 0.03 * (2*film_th/W_reg) * Re_film * Pr_air / (1 + 0.016 * ((2 * film_th / W_reg) * Re_film * Pr_air) ** (2 / 3))
    h_film = Nu_D*kair/(2*film_th)
    if Ud == 0:
        h_int = 100  # TODO: estimate heat transfer coefficient between liquid and casing for stagnation periods
        U_film = 0.1e1 / (1 / h_int + casing_th / kg10 + 1 / h_film)
    else:
        U_film = 0.1e1 / (0.5882352941e1 * (fRho * np.abs(Ud) * Dsp / fMu) ** (-0.79e0) / kf * Dsp + casing_th / kg10 + 1 / h_film)

    return U_film


# This calculates the overall heat transfer coefficient for the void space.

# Assumptions:
#
# Fully developed laminar flow in tubes of rectangular cross section
# Uniform heat flux in the control surface
# Nusselt number equal to 5.04, found by interpolation from table 8.1 of Incropera based on relation width/height
# Film flow between parallel plates for the calculation of the external heat transfer coefficient

@jit(f8(                         f8, f8,   f8,  f8,  f8,        f8, f8),nopython=True)
def ThermalResistanceVoid(kair, kf, kg10, kult, r0, r1, r2, r3, fAMR, Ud, A_c, P_c, casing_th, air_th):

    u_air = 0.1075 * np.pi * fAMR  # velocity of air. The velocity of the center of the magnet was considered r=0.1075 m
    # Laminar flow between parallel plates
    film_th = air_th
    Re_film = u_air * 2 * film_th / (Mu_air / rho_air)
    Nu_D = 7.54 + 0.03 * (2*film_th/W_reg) * Re_film * Pr_air / (1 + 0.016 * ((2 * film_th / W_reg) * Re_film * Pr_air) ** (2 / 3))
    h_film = Nu_D*kair/(2*film_th)
    if Ud == 0:
        h_int = 100  # TODO: estimate heat transfer coefficient between liquid and casing for stagnation periods
        U_film_void = 0.1e1 / (1 / h_int + casing_th / kg10 + 1 / h_film)
    else:
        U_film_void = 0.1e1 / ((1/5.04e0) / kf * (4*A_c/P_c) + casing_th / kg10 + 1 / h_film)

    return U_film_void
