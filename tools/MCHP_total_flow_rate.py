import numpy as np
import matplotlib.pyplot as plt


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Calculate total flow rate for any number of AMRs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Note: this applies for a MCHP with a magnet assembly consisting of two poles so that 2 AMR cycles are completed per
# revolution of the magnet assembly. This does the same that the Simulink model does but with enhanced flexibility.

number_of_AMRs = 14
# frequency_AMR = 1  # Hz
# Max_vflow_per_AMR = 1  # Lpm
time_steps = 2800  # One AMR cycle
Fb = 55/180  # % tau. Blow fraction given as a percentage of cycle period


flow_profile_AMRs = np.zeros((number_of_AMRs, 2*time_steps))

for i in range(number_of_AMRs):
    blow_slice = slice(int(i*time_steps/number_of_AMRs), int(i*time_steps/number_of_AMRs)+int(Fb*time_steps), 1)

    if int(i*time_steps/number_of_AMRs) - time_steps < 0:
        a = 0
    else:
        a = int(i*time_steps/number_of_AMRs) - time_steps

    if int(i*time_steps/number_of_AMRs) + int(Fb*time_steps) - time_steps < 0:
        b = 0
    else:
        b = int(i*time_steps/number_of_AMRs) + int(Fb*time_steps) - time_steps

    blow_slice2 = slice(a, b, 1)

    if int(i*time_steps/number_of_AMRs)+int(Fb*time_steps)+time_steps < 2 * time_steps:
        c = int(i*time_steps/number_of_AMRs)+int(Fb*time_steps)+time_steps
    else:
        c = 2 * time_steps

    blow_slice3 = slice(int(i*time_steps/number_of_AMRs) + time_steps, c, 1)

    flow_profile_AMRs[i, blow_slice] = 1
    flow_profile_AMRs[i, blow_slice2] = 1
    flow_profile_AMRs[i, blow_slice3] = 1
    plt.plot(np.linspace(1, 2*time_steps, 2*time_steps), flow_profile_AMRs[i, :])

total_flow = np.sum(flow_profile_AMRs, axis=0)

average_total_flow = np.average(total_flow)
print(average_total_flow)
plt.plot(np.linspace(1, 2*time_steps, 2*time_steps), total_flow)
plt.show()
