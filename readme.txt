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

0) This codebase can be run in three different ways. A single case, meaning a single combination of design, operating,
and numerical parameters, can be run from the script 'Run_single.py'. A series of cases, in which the user simulate, for
example, 5 different values of the boundary condition Thot to see the influence of this parameter on the performance of
the AMR, can be run from the script 'Run_series.py'. Finally, a batch of cases, in which the user can simulate a large
number of cases resulting from the combination of different values of up to 5 parameters, can be run in parallel in a
computer cluster from the script 'Run_parallel.py'. Run_single.py is useful mainly for testing the code. Run_series.py
is useful for running small number of cases on a single computer. Run_parallel.py is the most useful as it allows to
define the values of the parameters when running large number of cases in parallel in a computer cluster.

The values of most of the inputs must be defined in these three running scripts, but there are some inputs that are
defined in some other scripts as indicated in what follows.

1) In the folder configurations/ copy configuration R8.py and create a new file with the parameters of your own AMR.

All details of the AMR are defined in a configuration file located in the folder configurations/. Parameters related to
the dimensions, materials, number of layers, length of each layer, void layers, length of void layers, among some other
are defined in this configuration file. The value of any of the parameters included in the configuration file can be
modified from the Run_parallel.py, Run_series.py or Run_single.py scripts. Set to zero the value of the parameters that do not apply to your configuration. For example, if the overall shape of
                                        your AMR is cylindrical the width and height of the regenerator, W_reg and H_reg respectively, must be zero.

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

6) Fill the simulation parameters in the file Run_parallel.py if running several cases simultaneously in a computer
cluster, or in the file Run_series.py if running several cases one after another in a single computer. You can also run
a single case using the file Run_single.
TODO: version control model_MCMs.py

Coming....

5) Define the