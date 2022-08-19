import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np


@jit(nb.types.Tuple((f8, f8))(f8, f8, f8, f8, f8,  f8,  f8),nopython=True)
def SPresM(Dsp, Ud, V, er, flMu, flRho, Af):
    # The following corresponds to Ergun's correlation for calculation of pressure drop as presented in Mills
    dP = (1.75 * Ud ** 2 * (1 - er) / Dsp / er ** 3 * flRho + 150 * Ud * (1 - er) ** 2 / Dsp ** 2 / er ** 3 * flMu)
    Spress = dP * V
    return Spress, dP