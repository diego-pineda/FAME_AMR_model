# import tkinter as tk
# from tkinter import *
# from tkinter import filedialog
#
# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# from matplotlib.figure import Figure
#
#
# def calculate():
#
#     f = Figure(figsize=(2, 2), dpi=100)
#     # a = f.add_subplot(111)
#     # a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])
#     a.set_ylabel(tkvar.get())
#     a.set_xlabel("Something")
#
#     # canvas = FigureCanvasTkAgg(f, Outputs)
#     # # canvas.show()
#     # canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
#
#
#
#
#
# Outputs = Tk()
# Outputs.title('AMR model outputs')
# Outputs.geometry("1000x1000")
# Outputs.config(background="white")
#
#
#
#
# # ----------------------
#
# # def browseFiles():
# #     filename = filedialog.askopenfilename(
# #         initialdir="/Users/dfpinedaquijan/surfdrive/PhD Project/Numerical Model/MCHP_model_DP/output/",
# #         title="Select a File",
# #         filetypes=(("Text files", "*.txt*"), ("all files", "*.*")))
# #
# #     # Change label contents
# #     label_file_explorer.configure(text="File Opened: " + filename)
# #
# #
# # # Create a File Explorer label
# # label_file_explorer = Label(Outputs,
# #                             text = "File Explorer using Tkinter",
# #                             width = 100, height = 8,
# #                             fg = "blue")
# #
# #
# # button_explore = Button(Outputs,
# #                         text = "Browse Files",
# #                         command = browseFiles)
# #
# # button_exit = Button(Outputs,
# #                      text="Exit",
# #                      command=exit)
# #
# # # Grid method is chosen for placing  the widgets at respective positions in a table like structure by specifying rows
# # # and columns
# #
# # label_file_explorer.grid(column = 1, row = 1)
# #
# # button_explore.grid(column=1, row=2)
# #
# # button_exit.grid(column=1, row=3)
#
# # -----------------
# mainframe = Frame(Outputs)
# mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
# mainframe.columnconfigure(0, weight = 1)
# mainframe.rowconfigure(0, weight = 1)
# mainframe.pack(pady = 100, padx = 100)
#
# tkvar = StringVar(Outputs)
#
# # Dictionary with options
# choices = { 'Pizza','Lasagne','Fries','Fish','Potatoe'}
# tkvar.set('Pizza') # set the default option
#
# popupMenu = OptionMenu(mainframe, tkvar, *choices)
# Label(mainframe, text="Choose a dish").grid(row = 1, column = 1)
# popupMenu.grid(row = 2, column =1)
#
# # on change dropdown value
# # def change_dropdown(*args):
#     # if tkvar.get() == 'Lasagne':
#     #     print("Your favorite dish is: {}".format(tkvar.get()))
#     # if tkvar.get() == 'Potatoe':
#     #     print("You love: {}".format(tkvar.get()))
#     # print( tkvar.get() )
#
#
# # link function to change dropdown
#
#
#
# #------------------------------
#
#
#
#
# f = Figure(figsize=(2, 2), dpi=100)
# a = f.add_subplot(111)
# a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])
# # # a.set_ylabel(tkvar.get())
# # a.set_xlabel("Something")
# #
# canvas = FigureCanvasTkAgg(f, Outputs)
# # canvas.show()
# canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
#
# # toolbar = NavigationToolbar2Tk(canvas, Outputs)
# # toolbar.update()
# # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
#
#
#
#
#
# mybutton = Button(Outputs, text="Calculate", command=calculate).pack()
#
#
#
# # -----------------
#
# # tkvar.trace('w', change_dropdown)
#
#
#
# Outputs.mainloop()





# ---------------------------- Good Code ----------------------------------

import numpy as np

from tkinter import *
from math import  sin
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


import os
import sys
import importlib

# directory = "../../output/FAME_20layer_infl_Thot_flow"
directory = "output/FAME_20layer_ff_vfl_Dsp"  # "output/FAME_20layer_infl_Thot_flow2"
# inputs_file_name = 'FAME_20layer_infl_Thot_flow'  # File were the values of the input variables were defined.
inputs_file_name = "FAME_20layer_ff_vfl_Dsp"  # "Run_parallel"

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


Qc = np.ones((variable_3_resolution, variable_2_resolution, variable_1_resolution, hot_resolution, span_resolution))
Qh = np.ones((variable_3_resolution, variable_2_resolution, variable_1_resolution, hot_resolution, span_resolution))

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
        Qh[c, b, a, y, x] = float(((contents.split('\n'))[1].split(','))[1])




# Tspan = np.linspace(3, 30, 10)
# Thot = np.linspace(306, 314, 5)
# variable_1_name = 'Vflow'
# variable_1_units = 'Lpm'
# variable_1_values = np.linspace(0.5, 4.5, 9)
# variable_2_name = 'fAMR'
# variable_2_units = 'Hz'
# variable_2_values = np.linspace(1, 2, 11)
# variable_3_name = "Not used"  #'Dsp'
# variable_3_units = 'm'
# variable_3_values = [0] #np.linspace(200,600,5) * 1e-6

parameter_values = [Tspan, Thot, variable_1_values, variable_2_values, variable_3_values]
vble_names = ['Tspan', 'Thot', variable_1_name, variable_2_name, variable_3_name]
vble_units = ['K', 'K', variable_1_units, variable_2_units, variable_3_units]
Qc_indices = [0, 0, 0, 0, 0]
Qc_indices1 = [0, 0, 0, 0, 0]
Qc_indices2 = [0, 0, 0, 0, 0]


def main():
    root = Tk()
    gui = Window(root)
    gui.root.mainloop()
    return None


class Window:
    def __init__(self, root):
        self.root = root
        self.root.title("AMR Simulation Plots")
        self.root.geometry('1920x1130')

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SECTION I - Contour plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # %%%%%%%%%%%%%%%%% Top grid labels %%%%%%%%%%%%%%%%%%%

        Label(self.root, text="Variables", font='Helvetica 10 bold').grid(row=7, column=0, sticky='w', pady=10, ipadx=10)
        Label(self.root, text="Values"   , font='Helvetica 10 bold').grid(row=7, column=1, sticky='w', pady=10)
        Label(self.root, text="X"        , font='Helvetica 10 bold').grid(row=7, column=2, pady=10)
        Label(self.root, text="Y"        , font='Helvetica 10 bold').grid(row=7, column=3, pady=10)
        Label(self.root, text="Slices"   , font='Helvetica 10 bold').grid(row=7, column=4, sticky='w', pady=10)

        # %%%%%%%%%%%% Variable Labels and values %%%%%%%%%%%%%

        Label(self.root, text=vble_names[0]).grid(row=8, column=0, sticky='w', pady=8, ipadx=10)
        Label(self.root, text=str(Tspan)).grid(row=8, column=1, sticky='w', pady=8)

        Label(self.root, text=vble_names[1]).grid(row=9, column=0, sticky='w', pady=8, ipadx=10)
        Label(self.root, text=str(Thot)).grid(row=9, column=1, sticky='w', pady=8)

        Label(self.root, text=vble_names[2]).grid(row=10, column=0, sticky='w', pady=8, ipadx=10)
        Label(self.root, text=str(variable_1_values)).grid(row=10, column=1, sticky='w', pady=8)

        Label(self.root, text=vble_names[3]).grid(row=11, column=0, sticky='w', pady=8, ipadx=10)
        Label(self.root, text=str(variable_2_values)).grid(row=11, column=1, sticky='w', pady=8)

        Label(self.root, text=vble_names[4]).grid(row=12, column=0, sticky='w', pady=8, ipadx=10)
        Label(self.root, text=str(variable_3_values)).grid(row=12, column=1, sticky='w', pady=8)

        # %%%%%%%%%%%%%%%%%%% Radiobuttons %%%%%%%%%%%%%%%%%%%%

        self.X_option = IntVar(self.root)
        self.Y_option = IntVar(self.root)
        self.X_option.set(1)  # Note: the default X variable is set to be Tspan
        self.Y_option.set(2)  # Note: the default Y variable is set to be Thot

        Radiobutton(self.root, variable=self.X_option, value=1, command=self.deactivate_entries).grid(row=8, column=2)
        Radiobutton(self.root, variable=self.Y_option, value=1, command=self.deactivate_entries).grid(row=8, column=3)

        Radiobutton(self.root, variable=self.X_option, value=2, command=self.deactivate_entries).grid(row=9, column=2)
        Radiobutton(self.root, variable=self.Y_option, value=2, command=self.deactivate_entries).grid(row=9, column=3)

        Radiobutton(self.root, variable=self.X_option, value=3, command=self.deactivate_entries).grid(row=10, column=2)
        Radiobutton(self.root, variable=self.Y_option, value=3, command=self.deactivate_entries).grid(row=10, column=3)

        Radiobutton(self.root, variable=self.X_option, value=4, command=self.deactivate_entries).grid(row=11, column=2)
        Radiobutton(self.root, variable=self.Y_option, value=4, command=self.deactivate_entries).grid(row=11, column=3)

        Radiobutton(self.root, variable=self.X_option, value=5, command=self.deactivate_entries).grid(row=12, column=2)
        Radiobutton(self.root, variable=self.Y_option, value=5, command=self.deactivate_entries).grid(row=12, column=3)

        # %%%%%%%%%%%%%%%%%%% Drop down menus %%%%%%%%%%%%%%%%%

        # Note: the following frames are necessary for deactivating the dropdown menus of the selected radio-buttons
        self.frames = [Frame(self.root), Frame(self.root), Frame(self.root), Frame(self.root), Frame(self.root)]
        # self.frames[0].grid(row=8, column=3)  # Note: This is not included because X_option is set to 1
        # self.frames[1].grid(row=9, column=3)  # Note: This is not included because Y_option is set to 2
        self.frames[2].grid(row=10, column=4, sticky='ew')
        self.frames[3].grid(row=11, column=4, sticky='ew')
        self.frames[4].grid(row=12, column=4, sticky='ew')

        self.menu_vbles = [StringVar(self.root), StringVar(self.root), StringVar(self.root), StringVar(self.root), StringVar(self.root)]
        self.menu_vbles[2].set(parameter_values[2][0])
        self.menu_vbles[3].set(parameter_values[3][0])
        self.menu_vbles[4].set(parameter_values[4][0])

        self.menus = [OptionMenu(self.frames[0], self.menu_vbles[0], *parameter_values[0]),
                      OptionMenu(self.frames[1], self.menu_vbles[1], *parameter_values[1]),
                      OptionMenu(self.frames[2], self.menu_vbles[2], *parameter_values[2]),
                      OptionMenu(self.frames[3], self.menu_vbles[3], *parameter_values[3]),
                      OptionMenu(self.frames[4], self.menu_vbles[4], *parameter_values[4])]

        self.menus[0].grid(row= 8, column=4)
        self.menus[1].grid(row= 9, column=4)
        self.menus[2].grid(row=10, column=4)
        self.menus[3].grid(row=11, column=4)
        self.menus[4].grid(row=12, column=4)

        self.menus[0].configure(width=6)
        self.menus[1].configure(width=6)
        self.menus[2].configure(width=6)
        self.menus[3].configure(width=6)
        self.menus[4].configure(width=6)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SECTION II - Tspan vs Qcool plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Label(self.root, text="Variable", font='Helvetica 10 bold').grid(row=7, column=5, sticky='e', pady=10, ipadx=35)
        Label(self.root, text="Parameters", font='Helvetica 10 bold').grid(row=7, column=6, sticky='w', pady=10)

        # %%%%%%%%%%%%%%%%%%% Radiobuttons %%%%%%%%%%%%%%%%%%%%

        self.Z_option = IntVar(self.root)
        # self.Y_option = IntVar(self.root)
        self.Z_option.set(2)  # Note: the default X variable is set to be Thot
        # self.Y_option.set(2)  # Note: the default Y variable is set to be Thot

        # Radiobutton(self.root, variable=self.Z_option, value=1, command=self.deactivate_entries).grid(row=8, column=5)
        # Radiobutton(self.root, variable=self.Y_option, value=1, command=self.deactivate_entries).grid(row=8, column=3)

        Radiobutton(self.root, variable=self.Z_option, value=2, command=self.deactivate_menus2).grid(row=9, column=5, sticky='e', ipadx=50)
        # Radiobutton(self.root, variable=self.Y_option, value=2, command=self.deactivate_entries).grid(row=9, column=3)

        Radiobutton(self.root, variable=self.Z_option, value=3, command=self.deactivate_menus2).grid(row=10, column=5, sticky='e', ipadx=50)
        # Radiobutton(self.root, variable=self.Y_option, value=3, command=self.deactivate_entries).grid(row=10, column=3)

        Radiobutton(self.root, variable=self.Z_option, value=4, command=self.deactivate_menus2).grid(row=11, column=5, sticky='e', ipadx=50)
        # Radiobutton(self.root, variable=self.Y_option, value=4, command=self.deactivate_entries).grid(row=11, column=3)

        Radiobutton(self.root, variable=self.Z_option, value=5, command=self.deactivate_menus2).grid(row=12, column=5, sticky='e', ipadx=50)
        # Radiobutton(self.root, variable=self.Y_option, value=5, command=self.deactivate_entries).grid(row=12, column=3)

        # %%%%%%%%%%%%%%%%%%% Drop down menus %%%%%%%%%%%%%%%%%

        # Note: the following frames are necessary for deactivating the dropdown menus of the selected radio-buttons
        self.frames2 = [Frame(self.root), Frame(self.root), Frame(self.root), Frame(self.root), Frame(self.root)]
        # self.frames[0].grid(row=8, column=3)  # Note: not included because Tspan is a fixed variable for these plots
        # self.frames[1].grid(row=9, column=3)  # Note: This is not included because Z_option is set to 2
        self.frames2[2].grid(row=10, column=6, sticky='ew')
        self.frames2[3].grid(row=11, column=6, sticky='ew')
        self.frames2[4].grid(row=12, column=6, sticky='ew')

        self.menu_vbles2 = [StringVar(self.root), StringVar(self.root), StringVar(self.root), StringVar(self.root), StringVar(self.root)]
        self.menu_vbles2[2].set(parameter_values[2][0])
        self.menu_vbles2[3].set(parameter_values[3][0])
        self.menu_vbles2[4].set(parameter_values[4][0])

        self.menus2 = [OptionMenu(self.frames2[0], self.menu_vbles2[0], *parameter_values[0]),
                      OptionMenu(self.frames2[1], self.menu_vbles2[1], *parameter_values[1]),
                      OptionMenu(self.frames2[2], self.menu_vbles2[2], *parameter_values[2]),
                      OptionMenu(self.frames2[3], self.menu_vbles2[3], *parameter_values[3]),
                      OptionMenu(self.frames2[4], self.menu_vbles2[4], *parameter_values[4])]

        self.menus2[0].grid(row= 8, column=6)
        self.menus2[1].grid(row= 9, column=6)
        self.menus2[2].grid(row=10, column=6)
        self.menus2[3].grid(row=11, column=6)
        self.menus2[4].grid(row=12, column=6)

        self.menus2[0].configure(width=6)
        self.menus2[1].configure(width=6)
        self.menus2[2].configure(width=6)
        self.menus2[3].configure(width=6)
        self.menus2[4].configure(width=6)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SECTION III - Qcool vs X plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Label(self.root, text="Variable in X", font='Helvetica 10 bold').grid(row=7, column=7, sticky='e', pady=10, ipadx=30)
        Label(self.root, text="Series", font='Helvetica 10 bold').grid(row=7, column=8, sticky='w', pady=10, ipadx=35)
        Label(self.root, text="Parameters", font='Helvetica 10 bold').grid(row=7, column=9, sticky='w', pady=10)

        # %%%%%%%%%%%%%%%%%%% Radiobuttons %%%%%%%%%%%%%%%%%%%%

        self.Qc_X_option = IntVar(self.root)
        self.Qc_Z_option = IntVar(self.root)
        self.Qc_X_option.set(1)  # Note: the default X variable is set to be Tspan
        self.Qc_Z_option.set(2)  # Note: the default Y variable is set to be Thot

        Radiobutton(self.root, variable=self.Qc_X_option, value=1, command=self.deactivate_menus3).grid(row=8, column=7, sticky='e', ipadx=50)
        Radiobutton(self.root, variable=self.Qc_Z_option, value=1, command=self.deactivate_menus3).grid(row=8, column=8, sticky='w', ipadx=50)

        Radiobutton(self.root, variable=self.Qc_X_option, value=2, command=self.deactivate_menus3).grid(row=9, column=7, sticky='e', ipadx=50)
        Radiobutton(self.root, variable=self.Qc_Z_option, value=2, command=self.deactivate_menus3).grid(row=9, column=8, sticky='w', ipadx=50)

        Radiobutton(self.root, variable=self.Qc_X_option, value=3, command=self.deactivate_menus3).grid(row=10, column=7, sticky='e', ipadx=50)
        Radiobutton(self.root, variable=self.Qc_Z_option, value=3, command=self.deactivate_menus3).grid(row=10, column=8, sticky='w', ipadx=50)

        Radiobutton(self.root, variable=self.Qc_X_option, value=4, command=self.deactivate_menus3).grid(row=11, column=7, sticky='e', ipadx=50)
        Radiobutton(self.root, variable=self.Qc_Z_option, value=4, command=self.deactivate_menus3).grid(row=11, column=8, sticky='w', ipadx=50)

        Radiobutton(self.root, variable=self.Qc_X_option, value=5, command=self.deactivate_menus3).grid(row=12, column=7, sticky='e', ipadx=50)
        Radiobutton(self.root, variable=self.Qc_Z_option, value=5, command=self.deactivate_menus3).grid(row=12, column=8, sticky='w', ipadx=50)

        # %%%%%%%%%%%%%%%%%%% Drop down menus %%%%%%%%%%%%%%%%%

        # Note: the following frames are necessary for deactivating the dropdown menus of the selected radio-buttons
        self.frames3 = [Frame(self.root), Frame(self.root), Frame(self.root), Frame(self.root), Frame(self.root)]
        # self.frames[0].grid(row=8, column=3)  # Note: not included because Qc_X_option is set to 1
        # self.frames[1].grid(row=9, column=3)  # Note: not included because Qc_Z_option is set to 2
        self.frames3[2].grid(row=10, column=9, sticky='ew')
        self.frames3[3].grid(row=11, column=9, sticky='ew')
        self.frames3[4].grid(row=12, column=9, sticky='ew')

        self.menu_vbles3 = [StringVar(self.root), StringVar(self.root), StringVar(self.root), StringVar(self.root), StringVar(self.root)]
        self.menu_vbles3[2].set(parameter_values[2][0])
        self.menu_vbles3[3].set(parameter_values[3][0])
        self.menu_vbles3[4].set(parameter_values[4][0])

        self.menus2 = [OptionMenu(self.frames3[0], self.menu_vbles3[0], *parameter_values[0]),
                       OptionMenu(self.frames3[1], self.menu_vbles3[1], *parameter_values[1]),
                       OptionMenu(self.frames3[2], self.menu_vbles3[2], *parameter_values[2]),
                       OptionMenu(self.frames3[3], self.menu_vbles3[3], *parameter_values[3]),
                       OptionMenu(self.frames3[4], self.menu_vbles3[4], *parameter_values[4])]

        self.menus2[0].grid(row= 8, column=9)
        self.menus2[1].grid(row= 9, column=9)
        self.menus2[2].grid(row=10, column=9)
        self.menus2[3].grid(row=11, column=9)
        self.menus2[4].grid(row=12, column=9)

        self.menus2[0].configure(width=6)
        self.menus2[1].configure(width=6)
        self.menus2[2].configure(width=6)
        self.menus2[3].configure(width=6)
        self.menus2[4].configure(width=6)

        # %%%%%%%%%%%%%%%%%%% Contour plot update button %%%%%%%%%%%%%%%%%%%%

        button1 = Button(self.root, text="Update contour plot", font='Helvetica 10 bold', command=self.update_values)
        button1.grid(row=13, columnspan=5, pady=15)

        # %%%%%%%%%%%%%%%%%%% Tspan vs Qc update button %%%%%%%%%%%%%%%%%%%%%

        button2 = Button(self.root, text="Update Tspan vs Qc plot", font='Helvetica 10 bold', command=self.update_values)
        button2.grid(row=13, column=5, columnspan=2, pady=15)

        # %%%%%%%%%%%%%%%%%%%%% Qc vs X update button %%%%%%%%%%%%%%%%%%%%%%%

        button3 = Button(self.root, text="Update Qc vs X plot", font='Helvetica 10 bold', command=self.update_values)
        button3.grid(row=13, column=7, columnspan=3, pady=15)

        self.root.bind("<Return>", self.update_values)
        self.plot_values()
        self.plot_Tspan_vs_Qc()
        self.plot_Qc_vs_X()
        pass

    def deactivate_entries(self):
        indices = [1, 2, 3, 4, 5]
        a = self.X_option.get()
        b = self.Y_option.get()
        indices.remove(a)
        if b in indices:
            indices.remove(b)
        print(b)
        self.frames[a-1].grid_forget()
        self.frames[b-1].grid_forget()
        for index in indices:
            self.frames[index-1].grid(row=7+index, column=4, sticky='ew')
        return None

    def deactivate_menus2(self):
        indices = [2, 3, 4, 5]
        a = self.Z_option.get()
        # b = self.Y_option.get()
        indices.remove(a)
        # if b in indices:
        #     indices.remove(b)
        # print(b)
        self.frames2[a-1].grid_forget()
        # self.frames[b-1].grid_forget()
        for index in indices:
            self.frames2[index-1].grid(row=7+index, column=6, sticky='ew')
        return None

    def deactivate_menus3(self):
        indices = [1, 2, 3, 4, 5]
        a = self.Qc_X_option.get()
        b = self.Qc_Z_option.get()
        indices.remove(a)
        if b in indices:
            indices.remove(b)
        # print(b)
        self.frames3[a-1].grid_forget()
        self.frames3[b-1].grid_forget()
        for index in indices:
            self.frames3[index-1].grid(row=7+index, column=9, sticky='ew')
        return None

    def update_values(self, event=None):
        self.plot_values()
        self.plot_Tspan_vs_Qc()
        self.plot_Qc_vs_X()
        return None

    def plot_values(self):
        indices = [1, 2, 3, 4, 5]
        aa = self.X_option.get()
        bb = self.Y_option.get()

        Qc_indices[aa-1] = slice(0, len(parameter_values[aa-1]), None)
        Qc_indices[bb-1] = slice(0, len(parameter_values[bb-1]), None)

        indices.remove(aa)
        if bb in indices:
            indices.remove(bb)

        for index in indices:
            Qc_indices[index-1] = list(parameter_values[index-1]).index(float(self.menu_vbles[index-1].get()))

        print(Qc_indices)

        Z = Qc[Qc_indices[4], Qc_indices[3], Qc_indices[2], Qc_indices[1], Qc_indices[0]]
        if bb < aa:
            Z = np.transpose(Z)
        X, Y = np.meshgrid(parameter_values[aa-1], parameter_values[bb-1])
        print(Z)

        xlab = vble_names[aa-1]
        ylab = vble_names[bb-1]

        figure = plt.figure(figsize=(6, 4), dpi=100)
        # figure.add_subplot(111).plot(t,y)
        CS = figure.add_subplot(111).contourf(X, Y, Z, levels=np.linspace(0, np.amax(Z), 100), extend='neither', cmap='jet')
        plt.colorbar(mappable=CS, aspect=10)
        chart = FigureCanvasTkAgg(figure, self.root)
        chart.get_tk_widget().grid(row=14, columnspan=5, padx=10)

        toolbar = NavigationToolbar2Tk(chart, self.root, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=15, columnspan=5, sticky='w', padx=10)

        # plt.grid()
        axes = plt.axes()
        # axes.set_xlim([0, 6.3])
        # axes.set_ylim([-3, 3])
        axes.set_xlabel(xlab)
        axes.set_ylabel(ylab)

    def plot_Tspan_vs_Qc(self):
        indices = [2, 3, 4, 5]
        aa = self.Z_option.get()
        # bb = self.Y_option.get()
        Qc_indices1[0] = slice(0, len(parameter_values[0]), None)
        Qc_indices1[aa-1] = slice(0, len(parameter_values[aa-1]), None)
        # Qc_indices[bb-1] = slice(0, len(parameter_values[bb-1]), None)

        indices.remove(aa)
        # if bb in indices:
        #     indices.remove(bb)
        legend_title = []
        for index in indices:
            Qc_indices1[index-1] = list(parameter_values[index-1]).index(float(self.menu_vbles2[index-1].get()))
            legend_title.append('{} = {} [{}]'.format(vble_names[index-1], self.menu_vbles2[index-1].get(), vble_units[index-1]))

        print(Qc_indices)

        x_for_plot = Qc[Qc_indices1[4], Qc_indices1[3], Qc_indices1[2], Qc_indices1[1], Qc_indices1[0]]
        # if bb < aa:
        #     Z = np.transpose(Z)
        # X, Y = np.meshgrid(parameter_values[aa-1], parameter_values[bb-1])
        print(x_for_plot)

        # xlab = vble_names[aa-1]
        # ylab = vble_names[bb-1]
        legends = []

        figure = plt.figure(figsize=(6, 4), dpi=100)
        for i in range(len(parameter_values[aa-1])):
            figure.add_subplot(111).plot(x_for_plot[i, :], parameter_values[0])
            legends.append("{}".format(parameter_values[aa-1][i]))

        # CS = figure.add_subplot(111).contourf(X, Y, Z, levels=np.linspace(0, np.amax(Z), 100), extend='neither', cmap='jet')
        # plt.colorbar(mappable=CS, aspect=10)
        chart = FigureCanvasTkAgg(figure, self.root)
        chart.get_tk_widget().grid(row=14, column=5, columnspan=2, padx=10)

        toolbar = NavigationToolbar2Tk(chart, self.root, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=15, column=5, columnspan=2, sticky='w', padx=10)

        # plt.grid(True)
        axes = plt.axes()
        axes.set_xlim([0, np.amax(x_for_plot)+2])
        # axes.set_ylim([-3, 3])
        # axes.set_xlabel(xlab)
        # axes.set_ylabel(ylab)
        axes.legend(legends, title="{} [{}]".format(vble_names[aa-1], vble_units[aa-1]))  # , title=legend_title
        axes.grid(True)
        axes.set_xlabel("Qc [W]")
        axes.set_ylabel("Tspan [K]")

    def plot_Qc_vs_X(self):
        indices = [1, 2, 3, 4, 5]
        aa = self.Qc_X_option.get()
        bb = self.Qc_Z_option.get()
        # Qc_indices2[0] = slice(0, len(parameter_values[0]), None)
        Qc_indices2[aa-1] = slice(0, len(parameter_values[aa-1]), None)
        Qc_indices2[bb-1] = slice(0, len(parameter_values[bb-1]), None)

        indices.remove(aa)
        if bb in indices:
            indices.remove(bb)
        legend_title = []
        for index in indices:
            Qc_indices2[index-1] = list(parameter_values[index-1]).index(float(self.menu_vbles3[index-1].get()))
            legend_title.append('{} = {} [{}]'.format(vble_names[index-1], self.menu_vbles3[index-1].get(), vble_units[index-1]))

        print(Qc_indices)

        y_for_plot = Qc[Qc_indices2[4], Qc_indices2[3], Qc_indices2[2], Qc_indices2[1], Qc_indices2[0]]
        if bb < aa:
            y_for_plot = np.transpose(y_for_plot)
        # X, Y = np.meshgrid(parameter_values[aa-1], parameter_values[bb-1])
        print(y_for_plot)

        xlab = "{} [{}]".format(vble_names[aa-1], vble_units[aa-1])
        # ylab = vble_names[bb-1]
        legends = []

        figure = plt.figure(figsize=(6, 4), dpi=100)
        for i in range(len(parameter_values[bb-1])):
            figure.add_subplot(111).plot(parameter_values[aa-1], y_for_plot[i, :])
            legends.append("{}".format(parameter_values[bb-1][i]))

        # CS = figure.add_subplot(111).contourf(X, Y, Z, levels=np.linspace(0, np.amax(Z), 100), extend='neither', cmap='jet')
        # plt.colorbar(mappable=CS, aspect=10)
        chart = FigureCanvasTkAgg(figure, self.root)
        chart.get_tk_widget().grid(row=14, column=7, columnspan=3, padx=10)

        toolbar = NavigationToolbar2Tk(chart, self.root, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=15, column=7, columnspan=3, sticky='w', padx=10)

        # plt.grid(True)
        axes = plt.axes()
        axes.set_ylim([0, np.amax(y_for_plot)+2])
        # axes.set_ylim([-3, 3])
        # axes.set_xlabel(xlab)
        # axes.set_ylabel(ylab)
        axes.legend(legends, title="{} [{}]".format(vble_names[bb-1], vble_units[bb-1]))  # , title=legend_title
        axes.grid(True)
        axes.set_xlabel(xlab)
        axes.set_ylabel("Qc [W]")

        # index = list(Tspan[0, 0, :]).index(Tspan_contour)
        #
        # Z = Qc[Qc_indices[4], Qc_indices[3], Qc_indices[2], Qc_indices[1], Qc_indices[0]]
        # X, Y = np.meshgrid(parameter_values[self.a-1], parameter_values[self.b-1])
        #
        # fig, ax = plt.subplots()
        # CS = ax.contourf(X, Y, Z, levels=np.linspace(0, 10, 100), extend='neither', cmap='jet')
        # # Set level to np.amax(Z) if desired that colorbar cover all values of Z only
        # plt.colorbar(mappable=CS, aspect=10)
        # CD = ax.contour(X, Y, Z, levels=[1, 3, 5, 7, 9], colors='grey', linewidths=0.75)
        # plt.clabel(CD, fontsize=6, inline=True)
        # # NOTE: levels must be defined according to the needs of a particular plot.
        # plt.xlabel('{} [{}]'.format(variable_1_name, variable_1_units))
        # plt.ylabel('{} [{}]'.format(variable_2_name, variable_2_units))
        # plt.title('Qc [W] for Tspan = {} [K]'.format(Tspan_contour))
        # plt.show()

        return None

    pass

main()

# https://github.com/hendog993/Python-Youtube/blob/master/Graph.py
# https://www.dummies.com/article/technology/programming-web-design/python/using-tkinter-widgets-in-python-141443
# https://stackoverflow.com/questions/53390947/how-can-i-hide-tkinter-widgets-based-a-radiobutton-selection

# https://stackoverflow.com/questions/44010469/disabling-a-radio-button-does-nothing