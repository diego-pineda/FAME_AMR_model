import numpy as np

def polo_vol_flow(nt,Vd,freq):
    tau_c = 1/freq                       # [s] Period of AMR cycle
    t     = np.linspace(0, tau_c, nt+1)

    vol_flow_rate = Vd * np.pi * freq * np.sin(2 * np.pi * freq * t)

    return vol_flow_rate

