import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import sys
import importlib

# directory = "../../output/FAME_20layer_infl_Thot_flow"
directory = "output/FAME_20layer_Dsp"
# inputs_file_name = 'FAME_20layer_infl_Thot_flow'  # File were the values of the input variables were defined.
inputs_file_name = "Run_parallel"

# inputs = importlib.import_module(directory.replace('/', '.').replace('.', '', 6)+'.'+inputs_file_name)
# inputs = importlib.import_module(directory.replace('/', '.')+'.'+inputs_file_name)
inputs = importlib.import_module(inputs_file_name)

variable_1_name = inputs.vble1name  # This must be either Thot or any other variable used in X_resolution
variable_1_units = inputs.vble1units
variable_1_values = inputs.vble1values
variable_1_resolution = inputs.vble1resolution
variable_2_name = inputs.vble2name # This must be the variable changed inside the if conditions in the inputs file
variable_2_units = inputs.vble2units
variable_2_values = inputs.vble2values  # [units] Variable name. Note: values used for variable 2 in the cases simulated
variable_2_resolution = inputs.vble2resolution
variable_3_name = inputs.vble3name # This must be the variable changed inside the if conditions in the inputs file
variable_3_units = inputs.vble3units
variable_3_values = inputs.vble3values  # [units] Variable name. Note: values used for variable 2 in the cases simulated
variable_3_resolution = inputs.vble3resolution

cases = inputs.numGroups
hot_resolution = inputs.hotResolution
span_resolution = inputs.TspanResolution

Thot = inputs.Thotarr
Tspan = inputs.Tspanarr

legends = []
legends2 = []

# if X_resolution != 1:
#     variable_1_range = X_resolution
#     variable_1_values = X_array
# else:
#     variable_1_range = hot_resolution
#     variable_1_values = Thot

Qc = np.ones((variable_3_resolution, variable_2_resolution, variable_1_resolution, hot_resolution, span_resolution))
# Tspan = np.ones((variable_3_resolution, variable_2_resolution, variable_1_resolution, hot_resolution, span_resolution))
# dP = np.ones((variable_3_resolution, variable_2_resolution, variable_1_resolution, hot_resolution, span_resolution))

for files in os.listdir(directory):  # Goes over all files in the directory

    if '.txt' in files:
        if 'index' in files:
            continue
        # case = int(files.split('-')[1].split('.')[0])
        case = int(files.split('.')[0])
        casegroup = int(np.floor(case / (span_resolution * hot_resolution)))

        a = int(np.floor((casegroup - variable_1_resolution * int(np.floor(casegroup / variable_1_resolution))) / 1))
        b = int(np.floor((casegroup - variable_1_resolution * variable_2_resolution * int(np.floor(casegroup / (variable_1_resolution * variable_2_resolution)))) / variable_1_resolution))
        c = int(np.floor((casegroup - variable_1_resolution * variable_2_resolution * variable_3_resolution * int(np.floor(casegroup / (variable_1_resolution * variable_2_resolution * variable_3_resolution)))) / (variable_1_resolution * variable_2_resolution)))
        y = int(np.floor(case / span_resolution) % hot_resolution)
        x = case % span_resolution
        # print(case, z, x, y)
        myfile = open(directory + '/' + files, "rt")
        contents = myfile.read()
        myfile.close()
        Qc[c, b, a, y, x] = float(((contents.split('\n'))[1].split(','))[2])
        # Tspan[c, b, a, y, x] = float(((contents.split('\n'))[1].split(','))[0])
        # dP[c, b, a, y, x] = float(((contents.split('\n'))[1].split(','))[5])
        # 1. split contents into lines with contents.split('\n')
        # 2. split line with index 1 (second line) using the ',' character (contents.split('\n'))[1].split(',')
        # 3. Takes the element with index 2 (3rd element) from the 2nd line: ((contents.split('\n'))[1].split(','))[2]
        # 4. converts third element of second line from a string into a float number


mark = ["<", ">", "v", "x", "^", "s", "o", "+", "x", 'p']
col = ["orangered", "blue", "green", "orange", "brown", "red", 'black', 'gray', 'indigo', 'olive']
# TODO from this point the code must be edited
# ------ Plotting Tspan vs Qc ------ one figure for each value of variable_1

if cases != 1:

    Question1 = input('Do you want to make one plot of Tspan vs Qc per {} in your data? y/n: '.format(variable_1_name))

    while Question1 != 'y' and Question1 != 'n':
        print('Invalid input. Please type either y or n.')
        Question1 = input()

    if Question1 == 'y':

        for i in range(variable_1_range):
            for k in range(cases):
                plt.figure(i+1)  # This is in order to not plot over any of the previous figures
                plt.plot(Qc[k, i, :], Tspan[k, i, :], marker=mark[k], color=col[k])
                legends.append('{} = {} [{}]'.format(variable_2_name, variable_2_values[k], variable_2_units))

            # plt.title('Influence of aspect ratio')
            plt.xlabel('Cooling capacity [W]')
            plt.ylabel('$T_{span}$ [K]')
            # plt.xlim(0, np.amax(Qc[:, :, :])+2)
            plt.grid(which='major', axis='both')
            plt.legend(legends, title='{} = {} [{}]'.format(variable_1_name, variable_1_values[i], variable_1_units)) # Use when there is more than one Thot in the results
            # plt.legend(legends)  # Use when there is only one Thot in the results

# ------ Plotting Tspan vs Qc ------ one figure for each value of variable_2

Question2 = input('Do you want to make one plot of Tspan vs Qc per {} in your data? y/n: '.format(variable_2_name))

while Question2 != 'y' and Question2 != 'n':
    print('Invalid input. Please type either y or n.')
    Question2 = input()

if Question2 == 'y':

    for k in range(cases):
        for i in range(variable_1_range):
            plt.figure(k+variable_1_range+1)  # This is in order to not plot over any of the previous figures
            plt.plot(Qc[k, i, :], Tspan[k, i, :], marker=mark[i], color=col[i])
            legends2.append('{} = {} [{}]'.format(variable_1_name, variable_1_values[i], variable_1_units))

        # plt.title('Influence of aspect ratio')
        plt.xlabel('Cooling capacity [W]')
        plt.ylabel('$T_{span}$ [K]')
        # plt.xlim(0, np.amax(Qc[:, :, :])+2)
        plt.grid(which='major', axis='both')
        plt.legend(legends2, title='{} = {} [{}]'.format(variable_2_name, variable_2_values[k], variable_2_units)) # Use when there is more than one Thot in the results
        # plt.legend(legends2)  # Use when there is only one Thot in the results


# ------------- Contour plot of Qc vs Vflow and Thot

if variable_1_range != 1 and len(variable_2_values) != 1:

    print('It is possible to make a contour plot with the data available. Do you want to proceed? y/n: ', end="")
    create_contour = input()

    while create_contour != 'y' and create_contour != 'n':
        print("\nInvalid input! Do you want to proceed? y/n: ", end="")
        create_contour = input()

    if create_contour == 'n':
        plt.show()
        sys.exit()
    else:
        print('\nThe following list shows the temperature spans used in the simulations. Tspan = {} [K] \n'.format(Tspan[0, 0, :]))
        Tspan_contour = float(input('Enter the value of Tspan do you want to consider for the contour plot: '))

        while (Tspan_contour not in Tspan[0, 0, :]):
            print('The entered value is not in the list above. Please enter one of the numbers in the list: ')
            Tspan_contour = float(input())

        index = list(Tspan[0, 0, :]).index(Tspan_contour)

        Z = Qc[:, :, index]
        X, Y = np.meshgrid(variable_1_values, variable_2_values)

        fig, ax = plt.subplots()
        CS = ax.contourf(X, Y, Z, levels=np.linspace(0, 10, 100), extend='neither', cmap='jet')
        # Set level to np.amax(Z) if desired that colorbar cover all values of Z only
        plt.colorbar(mappable=CS, aspect=10)
        CD = ax.contour(X, Y, Z, levels=[1, 3, 5, 7, 9], colors='grey', linewidths=0.75)
        plt.clabel(CD, fontsize=6, inline=True)
        # NOTE: levels must be defined according to the needs of a particular plot.
        plt.xlabel('{} [{}]'.format(variable_1_name, variable_1_units))
        plt.ylabel('{} [{}]'.format(variable_2_name, variable_2_units))
        plt.title('Qc [W] for Tspan = {} [K]'.format(Tspan_contour))
        plt.show()
        '''Check the following documentation for further details on how to make contour plots and colorbars:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
        https://matplotlib.org/stable/api/contour_api.html#matplotlib.contour.ContourLabeler.clabel'''

else:
    plt.show()


# ------------- Surface plot of Qc vs Lreg and Vflow

# fig = plt.figure(12)
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, Qc[:, :, 0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlabel("Qc$_{@T_{span}=5K}$ [W]")
# plt.ylabel("AMR length [mm]")
# plt.xlabel("Vflow [Lpm]")
#
# ax.set_xlim(0.5, 4.5)
# ax.xaxis.set_major_locator(LinearLocator(9))
# ax.set_ylim(40, 80)
# ax.yaxis.set_major_locator(LinearLocator(5))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

# ----------- Figures of Qc vs Utilization and Penetration percentage

# phi = np.loadtxt('utilization.txt')
# # pen_perc = np.loadtxt('pen_percentage.txt')
#
# fig, ax = plt.subplots(constrained_layout=True)
#
# for k in range(cases):
#
#     # plt.figure(k+1)  # This is in order to not plot over any of the previous figures
#     # plt.figure(1)
#     ax.plot(phi[k,:], Qc[k, :, 2], marker=mark[k], color=col[k])
#     # plt.figure(2)
#     # plt.plot(pen_perc[k,:]*100, Qc[k, :, 1], marker=mark[k], color=col[k])
#     legends2.append('L = {} [mm]'.format(groups[k]))
#
# # plt.figure(1)
# # plt.title('Qc vs utilization')
# ax.set_xlabel('Utilization [-]')
# ax.set_ylabel('Cooling capacity [W]')
# ax.grid(which='major',axis='both')
# ax.legend(legends2, title='Tspan = 15 [K]') # Use when there is more than one Thot in the results
# #plt.legend(legends)  # Use when there is only one Thot in the results
#
# # plt.figure(2)
# # plt.title('Qc vs penetration percentage')
# # plt.xlabel('L_{pen}/L_{reg} [%]')
# # plt.ylabel('Cooling capacity [W]')
# # plt.grid(which='major',axis='both')
# # plt.legend(legends2, title='Tspan = 10 [K]') # Use when there is more than one Thot in the results
# #plt.legend(legends)  # Use when there is only one Thot in the results
# def uti2penper(x):
#     return x*0.64*7900*300*100/(0.36*1000*4200)
#
# def penper2uti(x):
#     return x*0.36*1000*4200/(100*0.64*7900*300)
#
# secax = ax.secondary_xaxis('top', functions=(uti2penper, penper2uti))
# secax.set_xlabel('$L_{pen}/L_{reg}$ [%]')
# plt.show()

# ------ Plots of pressure drop

# plt.figure(2)
# plt.plot([50, 60, 70, 40, 80], dP[:, 0, [1, 2]])
#
# plt.legend(['Tspan = 5 k','Tspan = 10 K'])
# plt.show()

# # print(dP)
# temp = np.zeros_like(dP)
# temp[0,0,:] = dP[3, 0, :]
# temp[1,0,:] = dP[0, 0, :]
# temp[2,0,:] = dP[1, 0, :]
# temp[3,0,:] = dP[2, 0, :]
# temp[4,0,:] = dP[4, 0, :]
# # print(np.may_share_memory(temp,dP))
# print(temp)
# # dP[3, 0, :] = dP[0, 0, :]
# # dP[0, 0, :] = temp
# print(dP)


# plt.figure(3)
# plt.plot([40, 50, 60, 70, 80], dP[:, 0, [1]]/1000, marker="o")
# plt.plot([40, 50, 60, 70, 80], dP[:, 0, [2]]/1000, marker="s")
# plt.legend(['Tspan = 5 k','Tspan = 10 K'])
# plt.title('Pressure drop along the bed')
# plt.xlabel('Bed length [mm]')
# plt.ylabel("Pressure drop [kPa]")
# plt.grid(which='major',axis='both')
# plt.show()


# temp2 = np.zeros_like(Qc)
# temp2[0,0,:] = Qc[3, 0, :]
# temp2[1,0,:] = Qc[0, 0, :]
# temp2[2,0,:] = Qc[1, 0, :]
# temp2[3,0,:] = Qc[2, 0, :]
# temp2[4,0,:] = Qc[4, 0, :]


# plt.figure(4)
# plt.plot([40, 50, 60, 70, 80], Qc[:, 0, [1]], marker="o")
# plt.plot([40, 50, 60, 70, 80], Qc[:, 0, [2]], marker="s")
# plt.plot([40, 50, 60, 70, 80], Qc[:, 0, [3]], marker="v")
# plt.legend(['Tspan = 5 [K]','Tspan = 10 [K]', 'Tspan = 15 [K]'])
# plt.title('Cooling capacity as function of bed length')
# plt.xlabel('Bed length [mm]')
# plt.ylabel("Qc [W]")
# plt.grid(which='major', axis='both')
# plt.show()