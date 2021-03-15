# Calculation of Overall heat transfer coefficient for heat leaks through the casing of AMR of the FAME cooler

import numba as nb
from numba import jit, f8, int32, b1
import numpy as np

# OPTION 1: considering air as a stagnant layer of insulating material

# General assumptions

# 1) Air around the regenerators is stagnant. In the real device, the movement of the magnets causes air to move
# turbulently over the regenerators. It would be a good idea to fill the space between the magnets with foam
# or any other light material to avoid the turbulent sweeping air. Similarly, the space in between regenerators
# could be filled with some insulating material so that a layer of air of constant thickness is formed in between
# regenerators and magnets. In that way a planar Couette flow would develop between the magnet and regenerator casings.
# 2) It is assumed that heat only leaks in the vertical direction upwards and downwards.
# Heat does not leak in the azimuthal direction.

@jit(f8(                    f8, f8,  f8,   f8,   f8, f8,   f8,        f8, f8),nopython=True)
def FAME_ThermalResistance(Dsp, Ud, fMu, fRho, kair, kf, kg10, casing_th, air_th):

    return 0.1e1 / (0.5882352941e1 * (fRho * np.abs(Ud) * Dsp / fMu) ** (-0.79e0) / kf * Dsp + casing_th / kg10  + air_th / kair)

# This calculates the overall heat transfer coefficient for heat transfer from surrounding to regenerator bed
# Heat conduction in the flow direction inside the casing material is neglected
# Constant fluid properties outside the regenerator at ambient temperature (air)
# Negligible viscous disipation outside the regenerator
# Turbulent flow over an isothermal plate is considered.
# Completely turbulent boundary layer. Equation 7.31 from Incropera with A=0
# The Nusselt number is calculated using the expression: Nu=0.037*Re^(4/5)*Pr^(1/3) with L = 0.045 the width of the regenerator.


@jit(nb.types.Tuple((f8, f8))(f8, f8,  f8,   f8,   f8, f8,   f8,        f8, f8, f8),nopython=True)
def FAME_ThermalResistance2(Dsp, Ud, fMu, fRho, kair, kf, kg10, casing_th, fAMR, air_th):
    u_air = 0.1075 * np.pi * fAMR  # velocity of air. The velocity of the center of the magnet was considered r=0.1075 m
    rho_air = 1.184  # [kg/m3] Density of air at 298 [K] and 1 [atm] pressure
    Mu_air = 1.849e-5  # [kg/(m*s)] Dynamic viscosity of air at 298 [K] and 1 [atm] pressure
    Pr_air = 0.7296  # [-] Prandlt number of air at 298 [K] and 1 [atm] pressure

    Re_HTF = fRho * np.abs(Ud) * Dsp / fMu  # Based on the velocity in the channel rather than interstitial velocity
    h_int = 0.17*Re_HTF**0.79*kf/Dsp  # Correlation used in Theo's thesis

    # Turbulent flow over a flat plate
    W_reg = 0.045  # [m] Width of regenerator. This is the characteristic length on the external side of regenerator
    Re_flat = rho_air * u_air * W_reg / Mu_air  # [-] Reynolds number on the air side
    Nu_flat = 0.037 * Re_flat ** (4 / 5) * Pr_air ** (1 / 3)
    h_flat = Nu_flat * kair / W_reg
    # TODO: change this hard programming style. Pass W_reg, film_th as argument as well as temperature of air
    U_flat = 0.1e1 / (0.5882352941e1 * (fRho * np.abs(Ud) * Dsp / fMu) ** (-0.79e0) / kf * Dsp + casing_th / kg10 + 1 / h_flat)

    # Laminar flow between parallel plates
    film_th = air_th
    Re_film = u_air * 2 * film_th / (Mu_air / rho_air)
    Nu_D = 7.54 + 0.03 * (2*film_th/W_reg) * Re_film * Pr_air / (1 + 0.016 * ((2 * film_th / W_reg) * Re_film * Pr_air) ** (2 / 3))
    h_film = Nu_D*kair/(2*film_th)
    U_film = 0.1e1 / (0.5882352941e1 * (fRho * np.abs(Ud) * Dsp / fMu) ** (-0.79e0) / kf * Dsp + casing_th / kg10 + 1 / h_film)
    return U_flat, U_film


# This calculates the thermal resistance for the void space.
@jit(f8(                         f8, f8,   f8,  f8,  f8,        f8, f8),nopython=True)
def FAME_ThermalResistanceVoid(kair, kf, kg10, A_c, P_c, casing_th, air_th):
    return 0.1e1 / ((1/5.04e0) / kf * (4*A_c/P_c) + casing_th / kg10 + air_th / kair)


# ---- Testing this functions ----

#Dsp, Ud, fMu, fRho, kair, kf, kg10, casing_th, air_th
if __name__ == '__main__':

    from sourcefiles.fluid.density import fRho
    from sourcefiles.fluid.dynamic_viscosity import fMu
    from sourcefiles.fluid.conduction import fK

    diam_spheres = 600e-6  # [m] Diameter of spherical particles composing the regenerator
    Vol_flow_rate = 0# 30.52e-6  # [m3/s] Maximum volumetric flow rate of HFT passing the regenerator
    Area_cross = 0.045*0.013  # [m2] Cross sectional area of regenerator
    Perimeter = 0.045*2+0.013*2  # [m] Perimeter of regenerator
    T_fluid = 290  # [K] Temperature of fluid
    perc_gly = 20  # [-] Percentage of glycol in the mixture
    Tamb = 298  # [K] Ambient temperature
    kg10 = 0.608  # [W/(m K)] Thermal conductivity g10 material
    thick_casing = 3.5e-3  # [m] Thickness of casing
    thick_air = 1e-3  # [m] Thickness of air layer
    fAMR = 1.7  # [Hz] Frequency of AMR


    Tair = Tamb
    kair = 1.5207e-11*Tair**3-4.8574e-8*Tair**2+1.0184e-4*Tair-3.9333e-4  # [W/(m K)] Thermal conductivity of air
    print('Thermal conductivity of air is: {}'.format(kair))

    rho_fluid = fRho(T_fluid, perc_gly)
    mu_fluid = fMu(T_fluid, perc_gly)
    k_fluid = fK(T_fluid,perc_gly)

    U1 = FAME_ThermalResistance(diam_spheres, Vol_flow_rate/Area_cross, mu_fluid, rho_fluid, kair, k_fluid, kg10, thick_casing, thick_air)
    U2 = FAME_ThermalResistance2(diam_spheres, Vol_flow_rate / Area_cross, mu_fluid, rho_fluid, kair, k_fluid, kg10,
                                 thick_casing, fAMR, thick_air)
    #Uvoid = FAME_ThermalResistanceVoid(kair, k_fluid, kg10, Area_cross, Perimeter, thick_casing, thick_air)

    print("Overall heat transfer coefficient stagnant air layer: {}".format(U1))
    print("Overall heat transfer coefficient flow over flat plate: {}\nOverall heat transfer coefficient flow between plates: {}".format(U2[0], U2[1]))
    #print("Overall heat transfer coefficient in voids with stagnant air layer: {}".format(Uvoid))