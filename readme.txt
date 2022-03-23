Numerical model of an Active Magnetocaloric Regenerator

This code is a model of an Active Magnetocaloric Regenerator. It was originally developed by Theo Christiaanse as part
of his PhD thesis at the University of Victoria. For details of the original model go to the following sources:

[1] Christiaanse T V. Characterization, Experimentation and Modeling of Mn-Fe-Si-P Magnetocaloric Materials. University
of Victoria, 2018.
[2] Christiaanse T V., Trevizoli P V., Rowe A. Modelling two layer Mn–Fe–Si–P materials in an active magnetic
regenerator. Int J Refrig 2019;106:225–35. https://doi.org/10.1016/j.ijrefrig.2019.07.002.

This is a 1D model in which the energy balance equations for the solid matrix and fluid are solved simultaneously. The
main outputs of this model are the temperature distributions of the solid MCM and the fluid along the AMR.

Instructions for running the codebase

This codebase can be run in three different ways. A single case, meaning a single combination of design, operating,
and numerical parameters, can be run from the script 'Run_single.py'. A series of cases, in which the user simulate, for
example, 5 different values of the boundary condition Thot to see the influence of this parameter on the performance of
the AMR, can be run from the script 'Run_series.py'. Finally, a batch of cases, in which the user can simulate a large
number of cases resulting from the combination of different values of up to 5 parameters, can be run in parallel in a
computer cluster from the script 'Run_parallel.py'. Run_single.py is useful mainly for testing the code. Run_series.py
is useful for running small number of cases on a single computer. Run_parallel.py is the most useful as it allows to
define the values of the parameters when running large number of cases in parallel in a computer cluster.

The values of most of the inputs of the model must be defined in these three running scripts, but there are some inputs
that are defined in some other scripts as indicated in what follows.

1) All details of the AMR are defined in a configuration file located in the folder configurations/. Parameters related to
the dimensions, materials, number of layers, length of each layer, void layers, length of void layers, among some other
are defined in this configuration file. The value of any of the parameters included in the configuration file can be
modified / overwritten from the Run_parallel.py, Run_series.py or Run_single.py scripts. To facilitate things, in the
folder configurations/ copy the file R8.py, which has predefined values of the input parameters, and create a new file
with the values of your own AMR. Set to zero the parameters that do not apply to your configuration. For example, if the
overall shape of your AMR is cylindrical the width and height of the regenerator, W_reg and H_reg respectively, must be
zero.

2) In the folder closure/ define the closure relations that you want to implement in your calculations.
    a)  In the subfolder heat_leaks/ define the correlations for the calculation of the overall heat transfer coefficient
        between fluid and ambient for the regenerator section and void sections inside the casing.
    b)  In the subfolder htc_fluid_solid/ define the correlation for the calculation of the heat transfer coefficient
        between solid and fluid in your regenerator.
    c)  If needed, you can modify the correlation for the calculation of the pressure drop in the AMR, which is defined
        in the file pressure_drop.py
    d)  If needed, you can modify the correlation for the calculation of the effective thermal conductivity of the AMR
        bed, which is defined in the file static_conduction.py
    e)  If needed, you can modify the correlation for the calculation of the effective thermal conductivity of the fluid
        in the AMR, which is defined in the file dynamic_conduction.py

3) Define your volume flow profile. It has to be provided in the files Run_parallel.py, Run_series, or Run_single.py in
the form of a vector with the name volum_flow_profile = size(time_steps). Normally, functions for the definition of the
flow profile are located in the folder sourcefiles/device/

4) Define your magnetic field profile. It has to be provided in the files Run_parallel.py, Run_series, or Run_single.py
in the form of a matrix with the name: app_field = size(time_steps, nodes). Normally, functions for the definition of
the magnetic field profile are located in the folder sourcefiles/device/

5) Provide the properties of each layer of MCM in the folder sourcefiles/new_mat/. Create a new subfolder for each
magnetocaloric material, and place there magnetization, heat capacity, and total entropy data. These data has to be in
the form of tables in .txt files containing the values of the property as a function of temperature (rows) and magnetic
field (columns). Make sure to include two files per property, one for the cooling measuring protocol, and one for the
heating measuring protocol even if the material does not exhibit any thermal hysteresis. For the last case the same data
can be used for the cooling and heating files.

The script model_MCMs.py can be used to create folders with corresponding files of modeled materials created departing
from the properties of one material and copying these properties to different Tc's for the new materials.

Structure of the code base

The main function of this codebase is called RunActive(), and it is located in the script 'FAME_DP_V1.py'. This function
takes all inputs defined as indicated above and returns the following outputs:

runActive():  returns

# Thot          0   eff_HB_CE   6   sHalfBlow   12  Uti         18  sMaxCBlow   24  fluid_dens  30
# Tcold         1   eff_CB_HE   7   sEndBlow    13  freq        19  sMaxHBlow   25  mass_flow   31
# qc            2   tFce        8   y           14  t           20  qh          26
# qccor         3   tFhe        8   s           15  xloc        21  cycleCount  27
# (t1-t0)/60    4   yHalfBlow   10  pt          16  yMaxCBlow   22  int_field   28
# pave          5   yEndBlow    11  np.max(pt)  17  yMaxHBlow   23  htc_fs      29

Some of these returned values are just inputs of the model such as Thot, Tcold, and freq. The main results are y and s, fluid and
solid temperatures respectively, which are 2D matrices containing the temperature of each spatial node for each time step.
pt is a vector containing the pressure drop along the AMR at each time step. int_field and htc_fs are 2D matrix containing internal
magnetic field and heat transfer coefficient between solid and fluid calculated for each spatial node and time step.
These main outputs are writen in an output .txt file that is sent to an output folder, generally called just output/,
which must be created by the user.

The user must also create a folder 'pickleddata' where partial results are saved in a pickle file when the simulation is
interrupted. The pickle files are only created when the simulation is interrupted upon reaching a predefined (by the user)
time limit or a predefined number of AMR cycle iterations.

The FAME_DP_V1.py script also contains two other important functions called SolveFluid() and SolveSolid() where system
of algebraic equations are built and solved. The folder core/ contains a script called tdma_solver.py with a function
TDMAsolver() which is the solver of the system of algebraic equations based on the Three Diagonal Matrix Algorithm.

In the folder sourcefiles/new_mat/ there is a script called 'int_funct.py', which creates spline interpolating functions
to calculate the properties of the MCM based on the material data provided (heat capacity, magnetization, and total entropy)

Functions for the calculation of themal conductivity, density, dynamic viscosity, and specific heat capacity of a water
glycol mixture as a function of temperature and volumetric fraction of glycol in the mixture are placed in the folder
sourcefiles/fluid/

List of dependencies

The codebase can be run in Python 3.6 and more recent versions. Last developments have been performed based on Python 3.8.
Some of the packages that are needed to run the codebase are:

numpy 1.20.0
scipy 1.6.0
numba 0.52.0

Testing of the codebase

Testing can be easily performed by using the script Run_single.py 