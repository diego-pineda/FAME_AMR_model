import numba as nb
from numba import jit, f8, int32,b1
import numpy as np

# This function calculates the pressure drop term for a packed screen bed based on the correlation by Armour and Cannon


@jit(nb.types.Tuple((f8, f8))(f8, f8, f8, f8, f8, f8, f8),nopython=True)
def SPresM(Dsp, Ud, V, er, flMu, flRho, Af):
    # V: volumetric flow rate [m3/s]
    # Dsp: wire diameter [m]
    # Ud: superficial velocity [m/s]
    # er: porosity []
    # flMu: fluid dynamic viscosity [kg/m*s]
    # flRho: flid density [kg/m3]
    # Af: area for fluid flow defined as Ac * er [m2]

    # Note: Msc is the mesh number [1/m] of the woven screen. This is one of the input geometric parameters. However,
    # for the sake of not changing the number of arguments of the SPresM function it was calculated back from porosity
    # and wire diameter Dsp

    # Reference: Lei et al. Appl. Therm. Eng. 111 (2017) 1232-1243

    Msc = np.sqrt(np.sqrt((64 * er**2 - 128 * er + np.pi**2 + 64) * Dsp**4)/(np.pi * Dsp**4) - 1/Dsp**2) / np.sqrt(2)
    dl = (1 - Msc * Dsp) / Msc
    beta = 4 * (1 - er) / Dsp
    Re = flRho * Ud / (beta**2 * flMu * dl)
    ff = 8.61 / Re + 0.52
    dP = ff * flRho * (Ud / er)**2 / dl
    Spress = dP * V
    return Spress, dP
