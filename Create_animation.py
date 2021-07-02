# Animation of the Time Dependent Temperature Distributions in an AMR

# README!

# This script creates an animation from the time dependent temperature distributions in the axial direction of an AMR.
# The time dependent temperature distributions must be in a text file, where each row contains the temperatures along
# the regenerator for each time step in the simulation. Solid and fluid temperatures can be in different files, or in
# the same file. If they are in different files, section 3) can be used to read the files, otherwise section 4) must be
# used.


# 0) Importing libraries

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

# 1) Simulation parameters

Thot        = 300    # [K]
Tcold       = 290    # [K]
dThot       = 4      # [K]
dTcold      = 2      # [K]
Reng_Length = 105.2  # [mm]
nodes       = 400    # [-]
time_steps  = 600    # [-]

# 2) Input and output file names

text_file_input = "./output/polo_low_uti_tol/8.0polo_low_uti_tol.sh-8.txt"
gif_file_output = "./output/polo_low_uti_tol/Tspan_10K_f0.5Hz_Uti0.28.gif"

# 3) Getting temperature data if the file only contains data corresponding to the temperature of solid or fluid

# #Fluid_temp_data_file_name = '../../../Simulation results/Fluid_Temp_val_span_11K.txt'
# #Solid_temp_data_file_name = '../../../Simulation results/Solid_Temp_val_span_11K.txt'
# #Fluid_temp_data_file_name = '../../Ftemp_10KTspan_6mm_void.txt'
# Fluid_temp_data_file_name = '../../Ends/9KGroup3FluidTemp.txt'
# Solid_temp_data_file_name = '../../Ends/9KGroup3SolidTemp.txt'

# fluidTemp = np.loadtxt('../../../Simulation results/Ftemp_10KTspan_6mm_void.txt')
# solidTemp = np.loadtxt('../../Ends/9KGroupC_Tol10e-7-SolidTemp.txt')

# 4) Getting temperature data if the input file contains more than just the temp of solid or fluid

myfile = open(text_file_input, "rt")
contents = myfile.read()
myfile.close()

fluidTemp = np.ones((time_steps+1, nodes+1))
solidTemp = np.ones((time_steps+1, nodes+1))

for j in range(time_steps+1):
    # fluid T matrix must start in row 3
    fluidTemp[j] = [float(i) for i in ((contents.split('\n'))[3+j].split())]
    # Solid T matrix must start in row time_steps+6
    solidTemp[j] = [float(i) for i in ((contents.split('\n'))[time_steps+5+j].split())]

# 5) Converting normalized temperatures to temperatures in [K]

fluidTemp = fluidTemp*(Thot-Tcold)+Tcold
solidTemp = solidTemp*(Thot-Tcold)+Tcold

# 6) Plotting temperature of node at cold side of regenerator for verification purposes

cold = plt.figure(1)
plt.plot(fluidTemp[:, 0])
plt.xlabel("Time steps [-]")
plt.ylabel("T° of node on cold side of AMR [K]")
plt.title("Temperature of cold side as a function of time")
plt.grid(which='both',axis='both')


# 6) Plotting the animated figure

x = []
y = []
s = []

fig, ax = plt.subplots()
ln, = plt.plot([], [], 'r', animated=True)
sn, = plt.plot([], [], 'b', animated=True)
f = time_steps+1


def init():
    ax.set_xlim(0, Reng_Length)
    ax.set_ylim(Tcold-dTcold, Thot+dThot)
    ln.set_data(x, y)
    sn.set_data(x, s)
    return ln, sn


def update(frame):
    x = np.linspace(0, Reng_Length, nodes + 1)
    y = fluidTemp[frame, :]
    s = solidTemp[frame, :]
    ln.set_data(x, y)
    sn.set_data(x, s)
    return ln, sn


ani = animation.FuncAnimation(fig, update, frames=f, init_func=init, blit=True, interval=50,repeat=True)
plt.ylabel("Temperature [K]")
plt.xlabel("Regenerator length [mm]")
plt.title("Temperature distributions along the regenerator")
plt.legend(["Fluid temperature", "Solid temperature"])
plt.grid(which='both', axis='both')
plt.show()

# 7) Saving the animation in a .gif file

# writergif = animation.PillowWriter(fps=30)
# ani.save(gif_file_output, writer=writergif)
