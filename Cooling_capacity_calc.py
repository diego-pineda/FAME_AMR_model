import numpy as np
from sourcefiles.device import FAME_V_flow
# Density
from sourcefiles.fluid.density import fRho
# Specific Heat
from sourcefiles.fluid.specific_heat import fCp
from configurations.R7 import percGly

Thot = 294.8 # [K]
Tcold = 286.8 # [K]
Vd = 12.94e-6  # [m3/s]
time_steps = 600 # [-]
nodes = 400 # [-]
freq = 1.2 # [Hz] Frequency of AMR cycle. Frequency of device would be half this frequency
#fluidTemp = np.loadtxt('Fluid_Temp_val.txt')
fluidTemp = np.loadtxt('Fluid_Temp_val_near_zero_qc.txt')
#fluidTemp = np.loadtxt('Fluid_Temp_val_zero_span.txt')
#fluidTemp = np.loadtxt('Fluid_Temp_val_span_11K.txt')
Tf_cold_side = fluidTemp[:, 0]*(Thot-Tcold)+Tcold
time_amr = np.linspace(0, 1/freq, time_steps+1)
volumetric_rate = FAME_V_flow.vol_flow_rate(time_steps, Vd)

Tf_cold_side_reg1 = np.append(Tf_cold_side,Tf_cold_side) # One full rotation of the device
V_flow_reg1 = np.append(volumetric_rate[:], volumetric_rate[:]) # One full rotation of the device

reg2ind = int(np.floor(1 * 2 * (time_steps + 1) / 7))
Tf_cold_side_reg2 = np.concatenate((Tf_cold_side[-reg2ind:], Tf_cold_side[:], Tf_cold_side[:-reg2ind]), axis=None) # One full rotation of the device
V_flow_reg2 = np.concatenate((volumetric_rate[-reg2ind:], volumetric_rate[:], volumetric_rate[:-reg2ind]), axis=None)

reg3ind = int(np.floor(2 * 2 * (time_steps + 1) / 7))
Tf_cold_side_reg3 = np.concatenate((Tf_cold_side[-reg3ind:], Tf_cold_side[:], Tf_cold_side[:-reg3ind]), axis=None)
V_flow_reg3 = np.concatenate((volumetric_rate[-reg3ind:], volumetric_rate[:], volumetric_rate[:-reg3ind]), axis=None)

reg4ind = int(np.floor(3 * 2 * (time_steps + 1) / 7))
Tf_cold_side_reg4 = np.concatenate((Tf_cold_side[-reg4ind:], Tf_cold_side[:], Tf_cold_side[:-reg4ind]), axis=None)
V_flow_reg4 = np.concatenate((volumetric_rate[-reg4ind:], volumetric_rate[:], volumetric_rate[:-reg4ind]), axis=None)

reg5ind = int(np.floor(4 * 2 * (time_steps + 1) / 7))
Tf_cold_side_reg5 = np.concatenate((Tf_cold_side_reg1[-reg5ind:], Tf_cold_side_reg1[:-reg5ind]), axis=None)
V_flow_reg5 = np.concatenate((V_flow_reg1[-reg5ind:], V_flow_reg1[:-reg5ind]), axis=None)

reg6ind = int(np.floor(5 * 2 * (time_steps + 1) / 7))
Tf_cold_side_reg6 = np.concatenate((Tf_cold_side_reg1[-reg6ind:], Tf_cold_side_reg1[:-reg6ind]), axis=None)
V_flow_reg6 = np.concatenate((V_flow_reg1[-reg6ind:], V_flow_reg1[:-reg6ind]), axis=None)

reg7ind = int(np.floor(6 * 2 * (time_steps + 1) / 7))
Tf_cold_side_reg7 = np.concatenate((Tf_cold_side_reg1[-reg7ind:], Tf_cold_side_reg1[:-reg7ind]), axis=None)
V_flow_reg7 = np.concatenate((V_flow_reg1[-reg7ind:], V_flow_reg1[:-reg7ind]), axis=None)

print(len(Tf_cold_side_reg7))
print(len(V_flow_reg7))

tau = 1/freq  # Period time of one AMR cycle
DT = tau/(time_steps+1)
cooling_power_sum = 0


for n in range(2*(time_steps+1)-1):
    coolPn_reg1 = (freq/2) * fCp((Tf_cold_side_reg1[n]+Tf_cold_side_reg1[n+1])/2, percGly) * fRho((Tf_cold_side_reg1[n]+Tf_cold_side_reg1[n+1])/2, percGly) * V_flow_reg1[n] * DT * ((Tf_cold_side_reg1[n]+Tf_cold_side_reg1[n+1])/2 - Tcold)
    coolPn_reg2 = (freq/2) * fCp((Tf_cold_side_reg2[n]+Tf_cold_side_reg2[n+1])/2, percGly) * fRho((Tf_cold_side_reg2[n]+Tf_cold_side_reg2[n+1])/2, percGly) * V_flow_reg2[n] * DT * ((Tf_cold_side_reg2[n]+Tf_cold_side_reg2[n+1])/2 - Tcold)
    coolPn_reg3 = (freq/2) * fCp((Tf_cold_side_reg3[n]+Tf_cold_side_reg3[n+1])/2, percGly) * fRho((Tf_cold_side_reg3[n]+Tf_cold_side_reg3[n+1])/2, percGly) * V_flow_reg3[n] * DT * ((Tf_cold_side_reg3[n]+Tf_cold_side_reg3[n+1])/2 - Tcold)
    coolPn_reg4 = (freq/2) * fCp((Tf_cold_side_reg4[n]+Tf_cold_side_reg4[n+1])/2, percGly) * fRho((Tf_cold_side_reg4[n]+Tf_cold_side_reg4[n+1])/2, percGly) * V_flow_reg4[n] * DT * ((Tf_cold_side_reg4[n]+Tf_cold_side_reg4[n+1])/2 - Tcold)
    coolPn_reg5 = (freq/2) * fCp((Tf_cold_side_reg5[n]+Tf_cold_side_reg5[n+1])/2, percGly) * fRho((Tf_cold_side_reg5[n]+Tf_cold_side_reg5[n+1])/2, percGly) * V_flow_reg5[n] * DT * ((Tf_cold_side_reg5[n]+Tf_cold_side_reg5[n+1])/2 - Tcold)
    coolPn_reg6 = (freq/2) * fCp((Tf_cold_side_reg6[n]+Tf_cold_side_reg6[n+1])/2, percGly) * fRho((Tf_cold_side_reg6[n]+Tf_cold_side_reg6[n+1])/2, percGly) * V_flow_reg6[n] * DT * ((Tf_cold_side_reg6[n]+Tf_cold_side_reg6[n+1])/2 - Tcold)
    coolPn_reg7 = (freq/2) * fCp((Tf_cold_side_reg7[n]+Tf_cold_side_reg7[n+1])/2, percGly) * fRho((Tf_cold_side_reg7[n]+Tf_cold_side_reg7[n+1])/2, percGly) * V_flow_reg7[n] * DT * ((Tf_cold_side_reg7[n]+Tf_cold_side_reg7[n+1])/2 - Tcold)

    cooling_power_sum = cooling_power_sum+coolPn_reg1+coolPn_reg2+coolPn_reg3+coolPn_reg4+coolPn_reg5+coolPn_reg6+coolPn_reg7

qc = cooling_power_sum

print(qc)

total_vol_flow = np.zeros(2*(time_steps+1))
for n in range(2*(time_steps+1)-1):
    if V_flow_reg1[n] < 0:
        V_flow_reg1[n] = 0
    if V_flow_reg2[n] < 0:
        V_flow_reg2[n] = 0
    if V_flow_reg3[n] < 0:
        V_flow_reg3[n] = 0
    if V_flow_reg4[n] < 0:
        V_flow_reg4[n] = 0
    if V_flow_reg5[n] < 0:
        V_flow_reg5[n] = 0
    if V_flow_reg6[n] < 0:
        V_flow_reg6[n] = 0
    if V_flow_reg7[n] < 0:
        V_flow_reg7[n] = 0

    total_vol_flow[n] = V_flow_reg1[n]+V_flow_reg2[n]+V_flow_reg3[n]+V_flow_reg4[n]+V_flow_reg5[n]+V_flow_reg6[n]+V_flow_reg7[n]

print(np.max(total_vol_flow))
