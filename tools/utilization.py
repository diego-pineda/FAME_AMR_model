import numba as nb
from numba import jit, f8, int32, b1
import numpy as np

# vol_flow_rate in Lpm
# density_fluid and density_AMR in kg/m3
# freq_AMR in Hz
# AMR_vol in m3
# cp_fluid and cp_MCM in J/(kg K)


@jit(f8(f8, f8, f8, f8, f8, f8, f8, f8, f8), nopython=True)
def utilization(vol_flow_rate, density_fluid, cp_fluid, porosity, freq_AMR, blow_fraction, AMR_vol, density_MCM, cp_MCM):

    return (vol_flow_rate * 16.667e-6) * density_fluid * cp_fluid * (1/freq_AMR) * blow_fraction / (AMR_vol * (1-porosity) * density_MCM * cp_MCM)


if __name__ == '__main__':

    U = np.zeros((3, 6))
    j = 0
    for vol in np.linspace(1, 6, 6):
        i = 0
        for freq in np.linspace(1, 3, 3):
            U[i, j] = utilization(vol, 1000, 4200, 0.36, freq, 50/180, 0.020*0.030*0.160, 6000, 800)
            i += 1
        j += 1

print(U)