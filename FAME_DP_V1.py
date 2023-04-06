
import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np
# Pickle Data
import pickle
# Interpolation Functions
# Plotting
# Debugging libraries
import sys
import os
# Import time to time the simulation
import time
import importlib

# -------------------------- Material properties -----------------------------

# from sourcefiles.mat import si1
# from sourcefiles.mat import si2
# from sourcefiles.mat import si3
# from sourcefiles.mat import si4
# from sourcefiles.mat import si5
# from sourcefiles.mat import Gd

from sourcefiles.new_mat.int_funct import material_data, matCp_c, matCp_h, matMag_c, matMag_h, matS_c, matS_h

# -------------------- Naming convention for MCM properties -----------------

# Specific heat function: mCp_c/h
# Magnetization: mMag_c/h
# Entropy: mS_c/h
# Temp: mTemp_c/h

# --------------------------------- Fluid properties ----------------------------------

from sourcefiles.fluid.density import fRho  # Density
from sourcefiles.fluid.dynamic_viscosity import fMu  # Dynamic Viscosity
from sourcefiles.fluid.specific_heat import fCp  # Specific Heat
from sourcefiles.fluid.conduction import fK  # Conduction

# -------------------- CLOSURE RELATIONSHIPS -------------------------

from closure.dynamic_conduction import kDyn_P  # Dynamic conduction
from closure.static_conduction import kStat  # Static conduction
#from closure.inter_heat import beHeff_I, beHeff_E  # Internal Heat transfer coefficient * Specific surface area
#from closure.pressure_drop import SPresM  # pressure Drop
from closure.heat_leaks.resistance import ThermalResistance  # Resistance Term in the Regenerator and void
# TODO: change the way how ThermalResistance is used for glass and lead spheres
# ----------------------- SOLVER ---------------------------------

from core.tdma_solver import TDMAsolver

# ----------------------- EXPONENTIAL SCHEME -----------------------------


@jit
def alpha_exp(Pe):
    # Exponential Scheme
    val = np.abs(Pe)/(np.expm1(np.abs(Pe)))
    return np.max([0, val])


# ------------------------ SOLID SOLVER -------------------------------
@jit(f8[:]     (f8[:],  f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8[:],  f8[:], f8[:], f8,   f8, int32, f8, f8),nopython=True)
def SolveSolid(iynext, isnext, yprev, sprev, Vd,    Cs,   Kse,   Ksw, Omegas,  Smce, CS, CMCE,     N, dx, dt):
    '''
    Solve the NTU/U solid matrix.
    See notebook "Theo Christiaanse 2017" pg. 92-97

    pg. 109
    Cs dy/dt-d/dx(Ks*dy/dx)=Smce+Omegas*(y-s)
    '''
    # Prepare material properties
    a = np.zeros(N-1)
    b = np.zeros(N-1)
    c = np.zeros(N-1)
    d = np.zeros(N-1)
    snext = np.zeros(N+1)
    if Vd>0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        for j in range(N-1): # note range will give numbers from 0->N-2
            # Build tridagonal matrix coefficients
            # pg. 115 Theo Christiaanse 2017
            a[j] = -CS*Ksw[j]/ (dx*2)
            b[j] = Cs[j+1]/dt+CS*Ksw[j]/(dx*2)+CS*Kse[j]/(dx*2)+Omegas[j+1]/2
            c[j] = -CS*Kse[j]/(dx*2)
            d[j] = (iynext[j]+iynext[j+1])*Omegas[j+1]/2+ sprev[j]*(CS*Ksw[j]/(2*dx)) + sprev[j+1]*(Cs[j+1]/dt-Omegas[j+1]/2-CS*Ksw[j]/(2*dx)-CS*Kse[j]/(2*dx)) + sprev[j+2]*(CS*Kse[j]/(2*dx)) + CMCE*Smce[j+1]
        # CS         <- Enable/Disable Conduction term in Solid
        # CMCE       <- Enable/Disable MCE
        # Add in BC
        # Neumann @ i=0
        b[0] = b[0]  + a[0]
        # Neumann @ i=-1
        b[-1]= b[-1] + c[-1]
        # Solve the unknown matrix 1-> N-1 DP: i.e. without the ghost nodes
        snext[1:-1] = TDMAsolver(a[1:], b, c[:-1], d)
        # Ghost node on either side of the Solid boundary
        snext[-1] = snext[-2]
        snext[0]  = snext[1]
        return snext
    else:
        # Add a value at the end
        for j in range(N-1):  # This will loop through 0 to N-2 which aligns with 1->N-1
            # Build tridagonal matrix coefficients
            # pg. 115 Theo Christiaanse 2017
            a[j] = -CS*Ksw[j]/(dx*2)
            b[j] = Cs[j+1]/dt+CS*Ksw[j]/(dx*2)+CS*Kse[j]/(dx*2)+Omegas[j+1]/2
            c[j] = -CS*Kse[j]/(dx*2)
            d[j] = (iynext[j+1]+iynext[j+2])*Omegas[j+1]/2+sprev[j]*(CS*Ksw[j]/(2*dx)) + sprev[j+1]*(Cs[j+1]/dt-CS*Ksw[j]/(2*dx)-CS*Kse[j]/(2*dx)-Omegas[j+1]/2) + sprev[j+2]*(CS*Kse[j]/(2*dx)) + CMCE*Smce[j+1]

        # Add in BC
        # Neumann @ i=0 : dT/dX|x=0 = 0 : (No heat transfer at the boundary)
        # T_i=0 = T_i=1 and a_1*T_0+b_1*T_1+c_1*T_2 = d_1 => b_1 = b_1+a_1 (in python first element has index 0)
        b[0] = b[0]  + a[0]
        # Neumann @ i=-1
        b[-1]= b[-1] + c[-1]
        # Solve the unknown matrix 1-> N-1
        snext[1:-1] = TDMAsolver(a[1:], b, c[:-1], d)
        # Ghost node on either side of the Solid boundary
        snext[-1] = snext[-2]
        snext[0]  = snext[1]
        return snext


@jit(f8(f8), nopython=True)
def alpha_pow(Pe):
    # Powerlaw
    val = (1-0.1*abs(Pe))**5
    return 1  # max(0, val)


# -------------------------- FLUID SOLVER ---------------------------------
@jit(f8[:]     (f8[:], f8[:],f8[:],f8[:],f8, f8[:], f8[:], f8[:],  f8[:],  f8[:], f8[:], f8[:], f8, f8[:], f8,int32,f8,f8,f8[:]),nopython=True)
def SolveFluid(iynext,isnext,yprev,sprev,Vd,    Cf,   Kfe,   Kfw,     Ff, Omegaf,    Lf,    Sp, CF, CL,   CVD,    N,dx,dt, yamb):
    '''
    Solve the NTU/U Fluid matrix.
    See notebook "Theo Christiaanse 2017" pg. 92-97

    pg. 109
    Cf dy/dt+d/dx(Ff*y)-d/dx(Kf*dy/dx)=Sp+Lf(yamb-y)+Omegaf*(s-y)
    '''
    # DP: This function builds and solves the system of linear algebraic equations to find temperatures of the
    # regenerator, i.e. from nodes 1 to N-1. The temperature of the ghost nodes are prescribed as boundary conditions.
    # DP: because of the discretization method implemented, the number of equations to solve is N-1.
    # The total amount of spatial nodes is N+1, with N being the index of the last node starting from 0.
    # Two temperatures are known: temperature of ghost nodes i=0 and i=N, T_0 = T_cold and T_N = T_N-1.
    # These are introduced as the boundary conditions. N-1 temperatures remain unknown (from i=1 to i=N-1) meaning
    # that N-1 equations are required. Therefore, the number of coefficients has to be also N-1.

    # Prepare material properties

    a = np.zeros(N-1)
    b = np.zeros(N-1)
    c = np.zeros(N-1)
    d = np.zeros(N-1)
    ynext = np.zeros(N+1)  # DP: this creates a position for the temperature of every node including the ghost nodes.

    if Vd > 0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        # Dirichlet ghost node
        ynext[0] = 0
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N.
            # DP: this actually loops from indices 0 to N-2. The system of algebraic equations to solve has the form:
            # DP: a[j]y[i-1]+b[j]y[i]+c[j]y[i+1]=d[j]
            # This is easier to understand when considering j and i ranging from index 1 to N-1. This way it is
            # clear that N-1 equations are created with its coefficients and the temperatures of all nodes from 0 to N
            # are considered in the system of equations. Note that the index of the nodes are relevant in the right hand
            # side terms of the following equations.
            # Build tridagonal matrix coefficients
            # pg 112-113 Theo Christiaanse 2017
            Aw = alpha_pow(Ff[j]/Kfw[j])  # TODO: determine why using this?
            Ae = alpha_pow(Ff[j+1]/Kfe[j])  # TODO: as I changed line 131, this is doing nothing at the moment
            # print(Aw, Ae)
            a[j] = -Ff[j]/(dx)-Aw*CF*Kfw[j]/(dx*2)+Omegaf[j+1]/2 # DP: indices coincide with thesis. Kfw is defined from node index i=1 to N-1
            # DP: Omegaf[0] is ignored, Kfw[0] corresponds to node i=1, Ff[0] corresponds to node i=0
            b[j] = Cf[j+1]/(dt)+Aw*CF*Kfw[j]/(2*dx)+Ae*CF*Kfe[j]/(2*dx)+CL[j+1]*Lf[j+1]/2+Omegaf[j+1]/2+Ff[j+1]/(dx)
            c[j] = -Ae*CF*Kfe[j]/(dx*2)
            # DP: when the index is j=N-2, the last one of this loop, Kfe[N-2] refers to node i=N-1
            d[j] = yprev[j]*(Aw*CF*Kfw[j]/(2*dx)) + yprev[j+1]*(Cf[j+1]/dt-Aw*CF*Kfw[j]/(dx*2)-Ae*CF*Kfe[j]/(dx*2) - CL[j+1]*Lf[j+1]/2) + yprev[j+2]*(Ae*CF*Kfe[j]/(dx*2)) + yamb[j+1]*(CL[j+1]*Lf[j+1])+isnext[j+1]*(Omegaf[j+1]/2)+sprev[j+1]*(Omegaf[j+1]/2)+CVD*Sp[j+1]
        # CF         <- Enable/Disable Conduction term in Fluid
        # CL         <- Enable/Disable Heat leaks term in the Fluid GE
        # CVD        <- Enable/Disable Viscous Dissipation Term in the Fluid GE
        # Add in boundary conditions
        # Dirichlet @ i=0 : T_i=0 = Tcold => y_i=0 = 0 (cold to hot blow). As y_i=0 is known the term a1y0 goes to the
        # right hand side of the equation and can be subtracted from d1
        d[0] = d[0]  - a[0]*ynext[0]
        # Neumann @ i=-1
        b[-1]= b[-1] + c[-1]
        # Solve the unknown matrix 1-> N-1
        ynext[1:-1] = TDMAsolver(a[1:],  b, c[:-1], d)
        # d\dx=0 ghost node.
        ynext[-1] = ynext[-2]
        return ynext
    elif Vd < 0:
        # Add a value at the end
        ynext[-1] = 1
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N
            # Build tridagonal matrix coefficients
            # pg 112-113 Theo Christiaanse 2017
            Aw=alpha_pow(Ff[j+1]/Kfw[j])
            Ae=alpha_pow(Ff[j+2]/Kfe[j])
            # print(Aw, Ae)
            a[j] = -Aw*CF*Kfw[j]/(2*dx)
            b[j] = Cf[j+1]/dt+Aw*CF*Kfw[j]/(2*dx)+Ae*CF*Kfe[j]/(2*dx)+CL[j+1]*Lf[j+1]/2+Omegaf[j+1]/2-Ff[j+1]/(dx)
            c[j] = Ff[j+2]/(dx)-Ae*CF*Kfe[j]/(2*dx)+Omegaf[j+1]/2
            d[j] = yprev[j]*(Aw*CF*Kfw[j]/(2*dx)) + yprev[j+1]*(Cf[j+1]/dt-Aw*CF*Kfw[j]/(dx*2)-Ae*CF*Kfe[j]/(dx*2)-CL[j+1]*Lf[j+1]/2) + yprev[j+2]*(Ae*CF*Kfe[j]/(dx*2)) + yamb[j+1]*(CL[j+1]*Lf[j+1]) + isnext[j+1]*(Omegaf[j+1]/2) + sprev[j+1]*(Omegaf[j+1]/2) + CVD*Sp[j+1]
        # Add in bc
        # Dirichlet @ i=-1
        d[-1] = d[-1] - c[-1] * ynext[-1]
        # Neumann @ i=0
        b[0]  = b[0]  + a[0]
        # Solve the unknown matrix 0->N-1
        ynext[1:-1] = TDMAsolver(a[1:], b, c[:-1], d)
        # d\dx=0 ghost node.
        ynext[0] = ynext[1]
        return ynext
    else:

        for j in range(N-1):  # DP: this loops from indices 0 to N-2.

            # Build tridiagonal matrix coefficients
            # DP: centered discretization implemented

            Aw=alpha_pow(Ff[j]/Kfw[j]) # TODO: determine why using this?
            Ae=alpha_pow(Ff[j+1]/Kfe[j])
            # print(Aw, Ae)
            a[j] = -Ff[j]/(dx)-Aw*CF*Kfw[j]/(dx*2) # DP: indices coincide with thesis. Kfw is defined from node index i=1 to N-1
            # DP: Omegaf[0] is ignored, Kfw[0] corresponds to node i=1, Ff[0] corresponds to node i=0
            b[j] = Cf[j+1]/(dt)+Aw*CF*Kfw[j]/(2*dx)+Ae*CF*Kfe[j]/(2*dx)+CL[j+1]*Lf[j+1]/2+Omegaf[j+1]/2+Ff[j+1]/(dx)
            c[j] = -Ae*CF*Kfe[j]/(dx*2)
            # DP: when the index is j=N-2, the last one of this loop, Kfe[N-2] refers to node i=N-1
            d[j] = yprev[j]*(Aw*CF*Kfw[j]/(2*dx)) + yprev[j+1]*(Cf[j+1]/dt-Aw*CF*Kfw[j]/(dx*2)-Ae*CF*Kfe[j]/(dx*2) - CL[j+1]*Lf[j+1]/2-Omegaf[j+1]/2) + yprev[j+2]*(Ae*CF*Kfe[j]/(dx*2)) + yamb[j+1]*(CL[j+1]*Lf[j+1])+isnext[j+1]*(Omegaf[j+1]/2)+sprev[j+1]*(Omegaf[j+1]/2)+CVD*Sp[j+1]
        # CF         <- Enable/Disable Conduction term in Fluid
        # CL         <- Enable/Disable Heat leaks term in the Fluid GE
        # CVD        <- Enable/Disable Viscous Dissipation Term in the Fluid GE
        # Add in boundary conditions
        # Neumann @ i=0
        b[0] = b[0]  + a[0]
        # Neumann @ i=-1
        b[-1]= b[-1] + c[-1]
        # Solve the unknown matrix 1-> N-1
        ynext[1:-1] = TDMAsolver(a[1:],  b, c[:-1], d)
        # d\dx=0 ghost nodes.
        ynext[0] = ynext[1]
        ynext[-1] = ynext[-2]
        return ynext


# ---------------------------- LOOP FUNC -----------------------------------


@jit(nb.types.Tuple((b1,f8))(f8[:],f8[:],f8))
def AbsTolFunc(var1,var2,Tol):
    maximum_val=np.max(np.abs(var1-var2))
    return maximum_val<=Tol,maximum_val

@jit(nb.types.Tuple((b1,f8))(f8[:,:],f8[:,:],f8))
def AbsTolFunc2d(var1,var2,Tol):
    maximum_val=np.max(np.abs(var1-var2))
    return maximum_val<=Tol,maximum_val

# ---------------------------- RUN ACTIVE ------------------------------


def runActive(caseNum, Thot, Tcold, cen_loc, Tambset, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, ConfName, jobName, time_lim, cycle_tol, max_step_iter, max_cycle_iter, vol_flow_profile, app_field, htc_model_name, leaks_model_name, pdrop_model_name, num_reg, gain):
    '''
    # runActive : Runs a AMR simulation of a pre-setup geometry
    # Arguments :
    # caseNum          <- caseNum number
    # Thot             <- Hot side heat exchanger
    # Tcold            <- Cold side heat exchanger
    # cen_loc          <- offset of the regenerator to the magnet TODO: remove from code. Particular for PM1 device
    # Tambset          <- Set ambient temperature
    # dispV            <- Displaced volume [m^3]. DP: for the FAME cooler this represents the maximum volumetric flow rate
    # ff               <- frequency [Hz]
    # CF               <- Enable/Disable Conduction term in Fluid
    # CS               <- Enable/Disable Conduction term in Solid
    # CL               <- Enable/Disable Heat leaks term in the Fluid GE
    # CVD              <- Enable/Disable Viscous Dissipation Term in the Fluid GE
    # CMCE             <- Enable/Disable MCE
    # nodes            <- Number of Spacial nodes used
    # timesteps        <- Number of Timesteps per cycle
    # ConfName         <- Load a certain configuration file
    # jobName          <- The name of the job
    # time_lim         <- Simulation time limit in minutes (Added by DP)
    # cycle_tol        <- Maximum cycle tolerance: criterion for the end of the iterative calculation process
    # max_step_iter    <- Maximum time step iterations the simulation is allowed to take
    # max_cycle_iter   <- Maximum cycle iterations the simulation is allowed to take
    # vol_flow_profile <- Matrix with flow rate values for every time step considered
    # app_field        <- Matrix with applied field data for each one of the nodes and time steps considered
    # htc_model_name   <- Name of file where function for convective heat transfer coefficient model is
    # leaks_model_name <- Name of file where functions for heat leak calculations are
    # pdrop_model_name <- Name of file where function for pressure drop is defined

    TODO: (16/06/2021) implement new feature - each layer of MCM can have its own porosity in the same way glass sphere
     and lead sphere layers have their own porosities.

    '''

    # Import the flow and heat transfer models

    htc = importlib.import_module('closure.htc_fluid_solid.' + htc_model_name)  # htc between solid and fluid
    leaks = importlib.import_module('closure.heat_leaks.' + leaks_model_name)  # heat leaks through regenerator casing
    predrop = importlib.import_module('closure.press_drop.' + pdrop_model_name)  # pressure drop in the AMR bed

    # ------- Import the geometric configuration of the regenerator -------

    config = importlib.import_module('configurations.' + ConfName)

    ''' README. !IMPORTANT! 
    - All configurations MUST have the same variables.
    - If new variables are needed they must be added also here and to any configuration file to be used afterwards
    - The advantage of importing the configuration variables in this way is to avoid having to add the import statement
    for any new configuration file that is needed. This way any number of configuration files can be created,
    and they will never have to be added here again.
    - (Deprecated 03/01/2022) Any variable in the configuration file can be now changed to study its influence on AMR 
    performance. In order to do so, as many configuration files as values of that variable are wanted need to be created
    - (03/01/2022) Variables in the configuration file can now be modified from the inputs file, Run_parallel.py, 
    Run_single.py, or Run_series.py. This means that in order to simulate different configurations it is no longer 
    needed to add as many configuration files as values of the variable to be changed are wanted. Only one configuration 
    file is needed and the variable to be changed will be modified from the inputs file by using a syntax similar to: 
    R8.Dsp = 150e-6, where R8 would be the configuration file where the other configuration parameters are defined.
    '''

    # For a cuboid regenerator

    W_reg     = config.W_reg      # [m] Width of regenerator
    H_reg     = config.H_reg      # [m] Height of regenerator
    L_reg1    = config.L_reg1     # [m] Length of first regenerator in the arrangement
    L_reg2    = config.L_reg2     # [m] Length of second regenerator in the arrangement
    casing_th = config.casing_th  # [m] Thickness of casing material
    air_th    = config.air_th     # [m] Thickness of air layer between regnerator and magnets

    # For a cylindrical regenerator

    r1   = config.r1    # [m] internal radius of regenerator casing
    r2   = config.r2    # [m] external radius of regenerator casing
    r3   = config.r3    # [m] internal radius of cylindrical magnet
    rvs  = config.rvs   # [m] internal radius of void section named "void"
    rvs1 = config.rvs1  # [m] internal radius of void section named "void1"
    rvs2 = config.rvs2  # [m] internal radius of void section named "void2"

    # If two regenerators with either voids or other passive beds are considered in the species description

    L_add  = config.L_add   # [m] TODO: remove from code this param. This parameter is only for the PM1 device.

    # Variables common to all geometries

    Ac = config.Ac  # [m2] Area of cross sectional area of regenerator
    Pc = config.Pc  # [m] Perimeter of regenerator
    Nd = config.Nd  # [-] Demagnetizing factor

    species_discription = config.species_discription  # [-] Different species found in the axial direction
    x_discription       = config.x_discription        # [m] Position of the different species relative to a zero
    reduct_coeff        = config.reduct_coeff         # [-] Dictionary with reduction coefficients of MCMs

    # About the active beds of magnetocaloric material

    Dsp  = config.Dsp   # [m] Diameter of MCM spheres
    er   = config.er    # [-] Porosity of packed bed
    mK   = config.mK    # [W/(m*K)] Thermal conductivity of MCM
    mRho = config.mRho  # [kg/m3] Density of MCM

    # About passive bed of glass spheres

    Dspgs = config.Dspgs  # [m] Diameter of glass spheres
    egs   = config.egs    # [-] Porosity of packed bed of glass spheres
    gsCp  = config.gsCp   # [J / (kg*K)] Heat capacity of glass spheres
    gsK   = config.gsK    # [W/(m*K)] Thermal conductivity of glass spheres
    gsRho = config.gsRho  # [kg/m3] Density of glass spheres

    # About the passive bed of lead spheres

    Dspls = config.Dspls  # [m] Diameter of lead spheres
    els   = config.els    # [-] Porosity of packed bed of lead spheres
    lsCp  = config.lsCp   # [J / (kg*K)] Heat capacity of lead spheres
    lsK   = config.lsK    # [W/(m*K)] Thermal conductivity of lead spheres
    lsRho = config.lsRho  # [kg/m3] Density of lead spheres

    # About casing and insulation materials

    kair = config.kair  # [W/(m*K)] Thermal conductivity of air at Tamb
    kg10 = config.kg10  # [W/(m*K)] Thermal conductivity of glass reinforced epoxy G10 at Tamb
    kult = config.kult  # [W/(m*K)] Thermal conductivity of Ultem at Tamb

    # Other parameters

    percGly = config.percGly  # Percentage of Glycol in the water - glycol mixture
    CL_set  = config.CL_set   # Type of model for the calculation of heat leaks
    MOD_CL  = config.MOD_CL   # Switch for activation of usage of experimental data for heat leaks
    ch_fac  = config.ch_fac   # Factor for averaging heating and cooling properties of MCMs with hysteresis

    # Printing to .out file the input parameters for control purposes

    print("\nThe current case uses the following AMR configuration:\n",
          "\n{:<30}{:>10.3f}\t{:<6}".format("AMR length", L_reg1, "[m]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("AMR width", W_reg, "[m]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("AMR height", H_reg, "[m]"),
          "\n{:<30}{:>10.0e}\t{:<6}".format("Particle diameter", Dsp, "[m]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("AMR porosity", er, "[-]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("MCM thermal conductivity", mK, "[W/(m*K)]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("MCM density", mRho, "[kg/m3]"),
          "\n\n{:<30}{}".format("Species in AMR configuration", species_discription),
          "\n{:<30}{}".format("Layer positions", x_discription), flush = True)

    print("\nThe current case uses the following operating parameters:\n",
          "\n{:<30}{:>10.3f}\t{:<6}".format("Thot", Thot, "[K]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("Tcold", Tcold, "[K]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("Tspan", Thot-Tcold, "[K]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("Tamb", Tambset, "[K]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("Max. Vol flow rate", np.amax(vol_flow_profile) / 16.667e-6, "[Lpm]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("AMR frequency", ff, "[Hz]"),
          "\n{:<30}{:>10.3f}\t{:<6}".format("Max. applied magnetic field", np.amax(app_field), "[T]"), flush = True)

    print("\nNumerical simulation parameters:\n",
          "\n{:<30}{:>10d}\t{:<6}".format("Nodes", nodes, "[-]"),
          "\n{:<30}{:>10d}\t{:<6}".format("Time steps", timesteps, "[-]"),
          "\n{:<30}{:>10d}\t{:<6}".format("Maximum cycle iterations", max_cycle_iter, "[-]"),
          "\n{:<30}{:>10d}\t{:<6}".format("Maximum time step iterations", max_step_iter, "[-]"),
          "\n{:<30}{:>10d}\t{:<6}".format("Time limit", time_lim, "[min]"),
          "\n{:<30}{:>10.0e}\t{:<6}".format("Cycle tolerance", cycle_tol, "[-]"), flush = True)

    # Creating list of interpolating functions depending on selected materials

    materials = list(set([i for i in species_discription if i.startswith('reg')]))
    # First remove no reg elements, then remove repeated elements with set(), and make the resulting set a list

    cp_c_if_list  = []
    cp_h_if_list  = []
    Mag_c_if_list = []
    Mag_h_if_list = []
    S_c_if_list   = []
    S_h_if_list   = []

    for material in materials:
        cp_c_if_list.append(matCp_c(material_data(material.split('-')[1])[0]))
        cp_h_if_list.append(matCp_h(material_data(material.split('-')[1])[1]))
        Mag_c_if_list.append(matMag_c(material_data(material.split('-')[1])[2]))
        Mag_h_if_list.append(matMag_h(material_data(material.split('-')[1])[3]))
        S_c_if_list.append(matS_c(material_data(material.split('-')[1])[4]))
        S_h_if_list.append(matS_h(material_data(material.split('-')[1])[5]))

    # Note:
    # The function material_data loads .txt files containing experimental data and returns numpy arrays with these data.
    # The input of this function is a string like 'M0', which represents a specific material. The split method is used
    # to remove the 'reg' part coming from the species_description vector. The functions starting with mat, such as
    # matCp_c() return an interpolating function. So, this for loop creates a list of interpolating functions for the
    # materials listed in the list 'materials'.

    # Import the configuration (Old style)
    # if ConfName == "R1":
    #     from configurations.R1  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,\
    #         kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
    #         percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    # if ConfName == "R2":
    #     from configurations.R2  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
    #         percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    # if ConfName == "R3":
    #     from configurations.R3  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
    #         percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    # if ConfName == "R4":
    #     from configurations.R4  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
    #         percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    # if ConfName == "R5":
    #     from configurations.R5  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
    #         percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    # if ConfName == "R6":
    #     from configurations.R6  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
    #         percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    # if ConfName == "R7":
    #     from configurations.R7 import Ac, Nd, MOD_CL, Pc, kair, kg10, kult, mK, mRho, percGly, species_discription, x_discription, CL_set, ch_fac, casing_th, air_th, L_reg

    # Start Timer
    t0 = time.time()
    time_limit_reached = 0
    # The space discretization considers all layers in the regenerator assembly including voids and passive layers
    N     = nodes                  # [-] Spatial nodes in which the reg. assembly is splitted for sim. No ghost nodes.
    dx    = 1 / (N-1)              # [-] Molecule size
    L_tot = np.max(x_discription)  # [m] Total length of the domain (regenerator assembly)
    DX    = L_tot / (N-1)          # [m] Real element size

    freq  = ff                           # [Hz] Frequency of AMR cycle
    nt    = timesteps                    # [-] Number of time steps
    tau_c = 1/freq                       # [s] Period of AMR cycle
    dt    = 1 / (nt+1)                   # [-] The time step
    DT    = tau_c/(nt+1)                 # [s] Time step # DP comment: ok
    t     = np.linspace(0, tau_c, nt+1)  # [s] Time vector from beginning, 0 [s], to the end of AMR cycle, tau [s].

    # DP comment: not sure why the denominator of dt is nt+1. Maybe it is also because there a time node for zero.
    # TODO: test what happens if DX = L_tot/(N-1) because nodes N and 0 are ghost nodes.
    #  This is even the definition shown in equation B.1 of Theo's thesis

    # A tolerance is established per step and per cycle as criteria to stop the simulation

    # DP: Create a list (maxStepTol) containing the values of the tolerances used to finish the iterative calculation
    # process for every time step. The list is needed to avoid many iterations on the time step level when the tolerance
    # on the cycle level is still far from the criterion for convergence

    maxCycleTol = cycle_tol  # DP: this was originally 1e-6
    maxStepTol  = [1]  # DP: this was originally [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    a = 0
    while maxStepTol[-1] >= maxCycleTol:
        a = a+1
        maxStepTol.append(10**-a)

    maxSteps  = max_step_iter   # Maximum allowed number of time step iterations. Time step iteration breaks if reached.
    maxCycles = max_cycle_iter  # Maximum allowed number of cycle iterations. Simulation breaks if reached.

    # Build darcy velocity for all n steps
    # Sine wave (See notebook pg. 33 Theo Christiaanse Sept 2017)
    #vf = lambda at, Ac, Vd, sf: (Vd) * sf * np.pi * np.sin(2 * np.pi * sf * at) + np.sign(np.sin(2 * np.pi * sf * at))*sys.float_info.epsilon*2
    # DP comment: it seems that the second term of the equation that describes the velocity profile of an alternating piston in a crank-rod-piston mechanism was not considered. Instead a weird sign function is added
    # DP comment: also, vf becomes V, which is the input parameter Vd of the function solvefluid(...,Vd,...). Vd varies with time according to a sine function.
    # DP comment: However, the flow direction through the AMR bed is indicated by the sign of the derivative of velocity rather the sign of velocity itself
    # DP: sys.float_info.epsilon returns the smallest real number that added to 1.0 can be distinguished from 1.0 by
    # the binary representation of a floating point real number

    # block wave (See notebook pg. 111 Theo Christiaanse Sept 2017)
    #uf = lambda at, Ac, Vd, sf: (Vd*sf / (Ac)) * np.sign(1/(sf*2)-sys.float_info.epsilon-at)


    # Build all volumetric fluid speeds V[n] (m^3/s) dt time unit
    #V = vf(t, Ac, Vd, freq)

    # from sourcefiles.device import FAME_V_flow
    # V = FAME_V_flow.vol_flow_rate(nt, Vd, acc_period, max_flow_period, full_magn_ang, unbal_rat)  # DP: Vd for the FAME cooler is the maximum volumetric flow rate in m^3/s
    V = vol_flow_profile

    # ------------------------- Added on 28-09-2022 to fix violation of conservation of mass ---------------------------
    i = 0
    m_flow = np.zeros(np.shape(V))
    rho_f_in_ave = fRho((Tcold+Thot)/2, percGly)
    for v in V:
        # if v > 0:
        #     rho_f_in = fRho(Tcold, percGly)
        #     m_flow[i] = rho_f_in * v
        # elif v < 0:
        #     rho_f_in = fRho(Thot, percGly)
        #     m_flow[i] = rho_f_in * v
        # else:
        #     m_flow[i] = 0

        m_flow[i] = rho_f_in_ave * v
        # Note: by using a density calculated at the average temperature between Tcold and Thot, mass flow imbalance is
        # avoided.
        i = i+1
    # ------------------------------------------------------------------------------------------------------------------

    vol_disp = 0
    for i in range(int(np.floor((nt+1)/2))):
        v_disp = V[i]*DT  # Integration using the rectangle rule
        vol_disp = vol_disp + v_disp  # Volume displaced in one blowing process
    print("\nVolume displaced in one blowing process: {:.3e} [L]".format(vol_disp*1000), flush=True)

    pdrop = lambda at, dP, sf: dP * sf * np.pi * np.sin(2 * np.pi * sf * at) + np.sign(np.sin(2 * np.pi * sf * at)) * sys.float_info.epsilon * 2
    # DP comment: Not very clear what this function does
    dPreg    = 5.2 * 6894.7572931783/2
    Lreg_exp = 22.5e-3
    ddP      = pdrop(t, dPreg, freq)
    dPdz_exp = ddP/Lreg_exp
    #U = uf(t, 1, 1, freq)

    # Calculate the utilization as defined by Armando's paper
    #Uti     = (Vd * 1000 * 4200) / (1000 * Ac * (1  - er) * 6100 * (L_reg1+L_reg2))
    Uti = (vol_disp * 1000 * 4200) / (235 * Ac * (1 - er) * mRho * (L_reg1+L_reg2))
    # TODO: the heat capacity of the MCM should be read from an input file or calculated somehow
    # DP comment: 6100 is the density of the MCM. 4200 is the Cp of water-glycol mixture. 1000 is the density of water.
    # DP comment: 235 in the denominator is an average value of Cp of Gd.
    print('Utilization: {0:1.3f}'.format(Uti), flush = True)
    # print('Urms: {0:3.3f}'.format((Vd / Ac*er) * freq * np.pi*1/np.sqrt(2)))

    # Initial ch-factor
    ch_factor = np.ones(N + 1)*ch_fac  # ch_fac is defined in configuration file for averaging cooling and heating props

    # This is modification of the casing BC. DP comment: in the configuration file, CL_set = "grad" and MOD_CL=0
    if CL_set=="Tamb":
        # Ambiant Temperature non-diamentionilized
        yamb = np.ones(N + 1) * ((Tambset - Tcold) / (Thot - Tcold))
        if CL == 0:
            CL = np.zeros(N+1)
        elif CL == 1:
            CL = np.ones(N+1) # DP comment: CL is the switch in the fluid GE to turn on or off the heat leaks term
    if CL_set=="f292":
        yamb = np.ones(N + 1) * ((292 - Tcold) / (Thot - Tcold))
        CL = np.ones(N+1)
    if CL_set=="grad":
        yamb = (np.linspace(Tcold,Thot,num=N+1) - Tcold) / (Thot - Tcold)
        CL = np.ones(N+1)
    if CL_set=="adiabatic":
        yamb = (np.linspace(Tcold,Thot,num=N+1) - Tcold) / (Thot - Tcold)
        CL = np.zeros(N+1)
    if MOD_CL==1:
        # Outer component description
        #
        # insul
        # insulator this will set casing losses to zero
        # condu
        # This indicates a conductor, temperature will be set based on the temperature assumptions
        # air
        # This part is set to the temperature of the ambient
        # hothex
        # This will be the temperature set by the hot hex
        # coldhex
        # This will be the temperature set by the cold hex
        outer_dis = ['coldhex','condu','condu','condu','air','hothex']
        # Length left flange
        L_lf= 0.019
        # Length magnet with plates
        L_mp= 0.112
        # Length right flange
        L_rf= 0.047
        # Length air gap
        L_ag = 0.0146
        outer_x   = [0,
                    L_add,
                    L_add+L_lf,
                    L_add+L_lf+L_mp,
                    L_add+L_lf+L_mp+L_rf,
                    L_add+L_lf+L_mp+L_rf+L_ag,
                    L_add+0.209]
        # Loop through discription to make array CL and Tamb
        mm=0
        # We need to distribute the space identifiers along a matrix to use it later.
        # This funcion cycles through x_discription until it finds a new domain then sets acording
        # to the int_discription
        int_disc_outer      = np.zeros(N+1,dtype=np.int)
        outer_descriptor    = []
        xloc_outer          = np.zeros(N+1)
        # Set the rest of the nodes to id with geoDis(cription)
        for i in range(N+1): # sets 0->N
            xloc_outer[i] = (DX * i + DX / 2)  #modify i so 0->N DP: seems this is the position of the center of nodes
            if (xloc_outer[i] >= outer_x[mm + 1]):
                mm = mm + 1
            int_disc_outer[i] = mm
            outer_descriptor.append(outer_dis[mm])
            if outer_descriptor[i] == 'coldhex':
                yamb[i] = 0
                CL[i]   = 1
            if outer_descriptor[i] == 'hothex':
                yamb[i] = 1
                CL[i]   = 1
            if outer_descriptor[i] == 'insul':
                CL[i]   = 0
            if outer_descriptor[i] == 'condu':
                CL[i]   = 1
                OptVar = 20 # DP: this was added in order to avoid an error. OptVar was not defined
                yamb[i] = ((273+OptVar - Tcold) / (Thot - Tcold)) # DP: what is OptVar?? This is not defined. It is producing an error
            if outer_descriptor[i] == 'air':
                yamb[i] = ((Tambset - Tcold) / (Thot - Tcold))
                CL[i]   = 1


    ## Field settings for PM1
    nn = 0
    # We need to distribute the space identifiers along a matrix to use it later.
    # This funcion cycles through x_discription until it finds a new domain then sets acording
    # to the int_discription
    int_discription = np.zeros(N+1, dtype=np.int)
    species_descriptor = []
    xloc = np.zeros(N+1)
    # Set the rest of the nodes to id with geoDis(cription)

    for i in range(N+1): # sets 0->N
        xloc[i] = (DX * i + DX / 2)  #modify i so 0->N
        if (xloc[i] >= x_discription[nn + 1]):
            nn = nn + 1
        int_discription[i] = nn
        species_descriptor.append(species_discription[nn])
    # TODO: this approach includes a small error because x_discription starts from the node 1 and gives the positions
    #  of the boundaries of each node while xloc starts from node 0 and gives the position of the center of the node.

    ## Set Surface area and Porosity of solid and fluid

    # For the FAME cooler

    A_c = np.ones(N + 1) * Ac  # DP comment: transversal area of regenerator and glass sphere sections
    P_c = np.ones(N + 1) * Pc  # DP comment: perimeter of circular section of regenerator and glass sphere sections
    e_r = np.ones(N + 1)  # DP comment: porosity
    for i in range(N+1):
        if species_descriptor[i].startswith("reg"):
            e_r[i] = er
        elif species_descriptor[i].startswith("void"):
            e_r[i] = 1
    # TODO: if different descriptors are used then this for loop should include them
    # TODO: this change can generates problems if I do not eliminate the if statements for gs and ls in the rest of code

    # DP: the following lines have been replaced for the FAME cooler by the previous lines ##

    # A_c = np.ones(N + 1) # DP comment: transversal area of regenerator and glass sphere sections
    # e_r = np.ones(N + 1) # DP comment: porosity
    # P_c = np.ones(N + 1) # DP comment: perimeter of circular section of regenerator and glass sphere sections
    #
    # for i in range(N+1):
    #     if species_descriptor[i].startswith("reg"):
    #         A_c[i] = Ac
    #         e_r[i] = er
    #         P_c[i] = Pc
    #     elif species_descriptor[i]== 'gs': # In configuration R7 this is not necessary
    #         A_c[i] = Ac
    #         e_r[i] = egs
    #         P_c[i] = Pc
    #     elif species_descriptor[i]== 'ls': # In configuration R7 this is not necessary
    #         A_c[i] = Ac
    #         e_r[i] = els
    #         P_c[i] = Pc
    #     else:
    #         if species_descriptor[i]=='void': # In configuration R7 this is not necessary
    #             A_c[i] = rvs**2 * np.pi
    #             e_r[i] = 1
    #             P_c[i] = 2*rvs*np.pi
    #         if species_descriptor[i]=='void1': # In configuration R7 this is not necessary
    #             A_c[i] = rvs1**2 * np.pi
    #             e_r[i] = 1
    #             P_c[i] = 2*rvs1*np.pi
    #         if species_descriptor[i]=='void2': # In configuration R7 this is not necessary
    #             A_c[i] = rvs2**2 *np.pi
    #             e_r[i] = 1
    #             P_c[i] = 2*rvs2 *np.pi

    # This is the domain fraction, determining the algebraic
    # split between domains. It will ensure the variation of
    # conduction between domains is taken into account.
    # By default it will be 0.5 if domain species does not change.
    # Please review  notebook pg. 116-117
    fr = np.ones(N) * 0.5
    nn = 1
    for i in range(N):
        # If we are between boundaries fr will be set to 0.5
        if (xloc[i] < x_discription[nn] and x_discription[nn] < xloc[i+1]):
            fr[i] = (xloc[i+1]-x_discription[nn])/DX
            nn    = nn + 1
    # TODO: the problem with this is that x_discription starts at the node 1 while xloc starts at the node 0. So, this
    #  quantities are not based on the same reference and their difference could lead to problems.
    # DP: fr[i] is useful for the calculation of the kf_west Kf_east. It refers to the fraction of the control volume that corresponds to a particular material

    ############################# BUILD appliedField array ####################
    # Shift the rotation of the magnet so it aligns with the sin time
    #RotMag = lambda t, f: 360 * t * f + 270 - 360 * np.floor(t * f + 270 / 360)
    # Build all rotMag[n]
    #rotMag = np.copy(RotMag(t, freq))
    # This one exists everywhere.
   # appliedFieldm = np.ones((nt+1, N + 1))
   #  for i in range(N + 1):
   #      for n in range(0, nt+1):
   #          #Will only get the field if we find a regenerator
   #          if (species_descriptor[i].startswith("reg")):
   #              x_pos_w_respect_to_magnet = xloc[i] - magOffset
   #              appliedFieldm[n, i] = hapl.appliedField(x_pos_w_respect_to_magnet, rotMag[n])[0, 0]*CMCE
   #          else:
   #              appliedFieldm[n, i] = 0

    # For the POLO device
    # rotMag = np.linspace(0, 360, nt+1)
    # appliedFieldm = np.ones((nt+1, N + 1))
    # from sourcefiles.device import polo_mag_field
    # for i in range(N + 1):
    #     for n in range(0, nt+1):
    #         #Will only get the field if we find a regenerator
    #         if (species_descriptor[i].startswith("reg")):
    #             x_pos_w_respect_to_magnet = xloc[i] - 0.10532 / 2
    #             appliedFieldm[n, i] = polo_mag_field.appliedField(rotMag[n], x_pos_w_respect_to_magnet)[0, 0]*CMCE
    #         else:
    #             appliedFieldm[n, i] = 0

    # Applied field profile (Input of the model)
    #from sourcefiles.device import FAME_app_field
    #appliedFieldm = FAME_app_field.app_field(nt, N)
    # TODO decide what is the best approach for magnetic field. For now for POLO it is the one above
    appliedFieldm = app_field  # Matrix nt rows and N columns describing the magnetic field along the reg as a f(t)

    for i in range(N + 1):
        if (not species_descriptor[i].startswith("reg")):
            appliedFieldm[:, i] = 0


    ########################## START THE LOOP #########################

    # Initial temperature
    y1 = np.linspace(0, 1, N + 1)  # DP comment: Initial fluid temperature. Linear distribution from Tcold to Thot
    s1 = np.linspace(0, 1, N + 1)  # DP comment: Initial solid temperature. Linear distribution from Tcold to Thot

    # Check is there is some pickeled data
    PickleFileName = "./pickleddata/{0:}-{1:d}".format(jobName, int(caseNum))
    print("Pickle Data File: {}".format(PickleFileName), flush = True)
    try:
        # we open the file for reading
        fileObject = open(PickleFileName, 'rb')
        print("Loading the pickle file...\n", flush = True)
        # load the object from the file into var b
        bbb = pickle.load(fileObject)
        y = bbb[0]
        s = bbb[1]
        stepTolInt = bbb[2]
        iyCycle = bbb[3]
        isCycle = bbb[4]
    except FileNotFoundError:
        # Keep preset values
        print("Started normal!\n", flush = True)
        y = np.ones((nt+1, N+1))*y1 # DP: initial temperature distribution for every time step is set to a linear distribution from Tcold to Thot
        s = np.ones((nt+1, N+1))*s1
        stepTolInt = 0
        # Initial guess of the cycle values.
        iyCycle = np.copy(y)
        isCycle = np.copy(s)

    MFM = np.ones(N + 1)  # Magnetic Field Modifier
    int_field = np.zeros((nt+1, N+1))  # Matrix for storing internal magnetic field values for every time step and node
    htc_fs = np.zeros((nt+1, N+1))  # Matrix for storing heat transfer coefficients between fluid and solid
    fluid_dens = np.zeros((nt+1, N+1))  # Matrix for storing fluid density at every node and time step
    mass_flow = np.zeros((nt+1, N+1))  # Matrix for storing mass flow rate at every node and time step
    Vf = np.zeros((nt+1, N+1))  # Matrix for storing volume flow rate at every node and time step
    dPdx = np.zeros((nt+1, N+1))  # 15/09/22 Matrix for storing pressure drop per unit length at every node and time step
    k_stat = np.zeros((nt+1, N+1))  # 15/09/22 Matrix for storing kstat at every node and time step
    k_disp = np.zeros((nt+1, N+1))  # 15/09/22 Matrix for storing kdisp at every node and time step
    U_Pc_leaks = np.zeros((nt+1, N+1))  # 31/10/2022 Matrix for storing overall heat transfer coefficient for heat leaks

    cycleTol   = 0  # DP comment: this is equivalent to a boolean False
    cycleCount = 1  # DP comment: it was defined above that the maximum number of cycle iterations is 2000

    mu0 = 4 * 3.14e-7  # [Hm^-1] Vacuum permeability constant

    # %%%%%%%%%%%% DP: the iterative calculation process for the cycle starts here %%%%%%%%%%

    while (not cycleTol  and cycleCount <= maxCycles): # DP comment: "not cycleTol" evaluates if cycleTol is zero or False and return True if so...
        # Account for pressure every time step (restart every cycle)
        pt = np.zeros(nt + 1)  # DP: total pressure drop along the regenerator as function of time
        # DP comment: It seems these variables are not that relevant and that they are only created to keep a record
        minPrevHint = 0.5
        maxPrevHint = 0.5
        maxAplField = 0.5
        minAplField = 0.5
        maxMagTemp = Tcold
        minMagTemp = Thot
        maxCpPrev = 0
        minCpPrev = 3000
        maxSSprev = 0
        minSSprev = 3000
        #
        maxTemp = Tcold # DP comment: it is weird that the max temperature is the temperature of the cold side
        minTemp = Thot

        # 1) Calculation of magnetic field modifiers (MFM) at every position of the regenerator

        for i in range(N+1): # DP comment: This for loop intends to find a vector of Magnetic Field Modifiers for every position i along the flow path to account for the demagnetizing field in the MCM regenerators
            # Average Solid temperature
            Ts_ave=np.mean(s[:, i] * (Thot - Tcold) + Tcold) # DP comment: this returns the average value of a vector containing the solid temperatures at position i for all time steps
            # Maximum Applied Field
            maxApliedField = np.amax(appliedFieldm[:, i]) # DP comment: this returns the maximum value of a vector containing the applied field at position i for all time steps
            if maxApliedField == 0: # DP comment: for the PM1 there will be many positions with maxApliedField == 0 because they are out of the range of the magnet. For FAME cooler, maxApliedField is never zero
                MFM[i] = 0  # DP comment: MFM -> Magnetic Field Modifier, which was previously set to a matrix of ones
            else:
                # Maximum Magnetization at the maximum field
                mag_c = Mag_c_if_list[materials.index(species_descriptor[i])]
                mag_h = Mag_h_if_list[materials.index(species_descriptor[i])]
                # Old style [------
                # if   species_descriptor[i]== 'reg-si1': mag_c = si1.mMag_c; mag_h = si1.mMag_h  # DP comment: mMag_c and mMag_h interpolating functions are renamed
                # elif species_descriptor[i]== 'reg-si2': mag_c = si2.mMag_c; mag_h = si2.mMag_h
                # elif species_descriptor[i]== 'reg-si3': mag_c = si3.mMag_c; mag_h = si3.mMag_h
                # elif species_descriptor[i]== 'reg-si4': mag_c = si4.mMag_c; mag_h = si4.mMag_h
                # elif species_descriptor[i]== 'reg-si5': mag_c = si5.mMag_c; mag_h = si5.mMag_h
                # elif species_descriptor[i]== 'reg-Gd':  mag_c = Gd.mMag_c;  mag_h = Gd.mMag_h
                # -------]
                maxMagLoc= mag_c(Ts_ave,maxApliedField)[0, 0]*(1-ch_factor[i])+mag_h(Ts_ave,maxApliedField)[0, 0]*ch_factor[i] # DP comment: ch factors for all i positions were set to 0.5 previously
                # The resulting internal field
                Hint = maxApliedField - mRho * Nd * maxMagLoc * mu0 # DP comment: Nd comes from the configuration files. For the case of R1, Nd = 0.36
                # The decrease ratio of the applied field
                MFM[i] = Hint/maxApliedField # DP comment: this is for the accounting of the demagnetizing field in the calculation. It is updated at every
                # TODO: it seems that there is room for improvement here. The magnetization term in this equation is a function of internal field and temperature.
                #  Instead, it is calculated at the applied field. An iterative calculation process is required if Magnetization is considered function of Hint.
                #  Also, beyond this, the internal magnetic field should be calculated for every time step. Instead, an average MFM is calculated for every position
                #  along the regenerator and used to adjust the applied field at every time step and thus find an internal magnetic field. The internal magnetic field
                #  could be calculated using the temperature that is assumed at the beginning of every time step by implementing a small iterative loop given that
                #  magnetization is a function of the internal field.

        ###################### DP: The "for loop" to run over the time steps of a cycle starts here ###################

        Q_MCE = 0  # Added on 21/03/2023 to see the difference between Q_MCE and W_mag

        for n in range(1, nt+1):  # 1->nt

            # Initial
            stepTol = 0
            stepCount = 1
            # ch_factor[i]=0 coolingcurve selected
            # ch_factor[i]=1 heatingcurve seelected

            # Initial guess of the current step values.
            iynext  = np.copy(y[n-1, :])
            isnext  = np.copy(s[n-1, :])
            # DP: these are vectors containing the guessed values of the fluid and solid temperature distributions
            # for the current time step. As initial guess for the current time step, it is assumed that the temperature
            # distributions are equal to the final temp distribution of the previous time step. The guessed value will
            # be updated to the values obtained after each time step iteration.

            # current and previous temperature in [K]
            pfT = y[n-1, :] * (Thot - Tcold) + Tcold  # DP comment: Previous time step fluid temperature in [K]
            psT = s[n-1, :] * (Thot - Tcold) + Tcold  # DP comment: Previous time step solid temperature in [K]

            if max(pfT) > maxTemp:  # DP comment: given that maxTemp is Tcold, I assume max(pfT), which must be close to Thot, is greater than maxTemp
                maxTemp = max(pfT)
            if min(pfT) < minTemp:
                minTemp = min(pfT)

            # DP: Properties of fluid and solid initialized to zero for every time step.
            # DP: From the name of the variables, it seems to be related to the previous time step
            cpf_prev  = np.zeros(N+1)  # DP: heat capacity of fluid along the fluid path
            rhof_prev = np.zeros(N+1)  # DP: density of fluid along the fluid path
            muf_prev  = np.zeros(N+1)  # DP: viscosity of fluid along the fluid path
            kf_prev   = np.zeros(N+1)  # DP: thermal conductivity of fluid along the fluid path
            cps_prev  = np.zeros(N+1)  # DP: heat capacity of solid along the fluid path. This includes several types of solids such as glass spheres and MCM
            Ss_prev   = np.zeros(N+1)  # ??

            S_c_past  = np.zeros(N+1)  # DP: entropy of solid for a cooling protocol
            S_h_past  = np.zeros(N+1)  # DP: entropy of solid for a heating protocol
            Sirr_prev = np.zeros(N+1)  # DP: entropy of solid irreversible part?
            Sprev     = np.zeros(N+1)  # DP: Anhysteretic entropy of the MCM for the previous time step
            prevHintNew  = np.zeros(N+1) # ??

            for i in range(N+1):  # cps_prev Ss_prev are calculated in this for loop
                # DP: this goes from i=0 to i=N
                if species_descriptor[i].startswith("reg"):
                    # Internal field
                    prevHint       = appliedFieldm[n-1,i]*MFM[i]
                    prevHintNew[i] = appliedFieldm[n-1,i]*MFM[i]
                    # Heat capacity and entropy data of the MCM
                    cp_c = cp_c_if_list[materials.index(species_descriptor[i])]
                    cp_h = cp_h_if_list[materials.index(species_descriptor[i])]
                    ms_c = S_c_if_list[materials.index(species_descriptor[i])]
                    ms_h = S_h_if_list[materials.index(species_descriptor[i])]
                    # Old style [-------
                    # if   species_descriptor[i] == 'reg-si1': cp_c = si1.mCp_c; cp_h = si1.mCp_h; ms_c = si1.mS_c; ms_h = si1.mS_h
                    # elif species_descriptor[i] == 'reg-si2': cp_c = si2.mCp_c; cp_h = si2.mCp_h; ms_c = si2.mS_c; ms_h = si2.mS_h
                    # elif species_descriptor[i] == 'reg-si3': cp_c = si3.mCp_c; cp_h = si3.mCp_h; ms_c = si3.mS_c; ms_h = si3.mS_h
                    # elif species_descriptor[i] == 'reg-si4': cp_c = si4.mCp_c; cp_h = si4.mCp_h; ms_c = si4.mS_c; ms_h = si4.mS_h
                    # elif species_descriptor[i] == 'reg-si5': cp_c = si5.mCp_c; cp_h = si5.mCp_h; ms_c = si5.mS_c; ms_h = si5.mS_h
                    # elif species_descriptor[i] == 'reg-Gd':  cp_c = Gd.mCp_c;  cp_h = Gd.mCp_h;  ms_c = Gd.mS_c;  ms_h = Gd.mS_h
                    # -------]
                    # Previous specific heat
                    Tr = psT[i]
                    dT = .5  # DP: this could be any small value given that it is just for calculating the derivative
                    dsdT = (ms_c(Tr+dT, prevHint)[0, 0] * 0.5 + ms_h(Tr+dT, prevHint)[0, 0] * 0.5) - (ms_c(Tr-dT, prevHint)[0, 0] * 0.5 + ms_h(Tr-dT, prevHint)[0, 0] * 0.5)
                    cps_prev[i]  = psT[i]*(np.abs(dsdT)/(dT*2)) # DP: why not calculating the Cp from the available data?
                    # DP: 2 in the denominator obeys to the fact that the derivative is taken as [f(x+dx)-f(x-dx)]/(2*dx) instead of [f(x+dx)-f(x)]/(dx)
                    # Entropy position of the previous value
                    S_c_past[i]   = ms_c(Tr, prevHint)[0, 0]
                    S_h_past[i]   = ms_h(Tr, prevHint)[0, 0]
                    Sirr_prev[i]  = S_c_past[i] * (1-ch_factor[i]) - S_h_past[i] * ch_factor[i]  # DP: this corresponds to the irreversible part. It is not useful
                    Sprev[i]      = S_c_past[i] * (1-ch_factor[i]) + S_h_past[i] * ch_factor[i]  # DP: this is the anhysteretic entropy
                    # old code
                    Ss_prev[i]     = ms_c(psT[i], prevHint)[0, 0]*(1-ch_factor[i]) + ms_h(psT[i], prevHint)[0, 0]* ch_factor[i]  # DP: this is equivalent to Sprev[i]
                    if prevHint > maxPrevHint:
                        maxPrevHint = prevHint
                        maxAplField = appliedFieldm[n-1,i]
                        maxMagTemp  = psT[i]
                        maxCpPrev   = cps_prev[i]
                        maxSSprev   = Ss_prev[i]
                    if prevHint < minPrevHint:
                        minPrevHint = prevHint
                        minAplField = appliedFieldm[n-1,i]
                        minMagTemp  = psT[i]
                        minCpPrev   = cps_prev[i]
                        minSSprev   = Ss_prev[i]
                elif species_descriptor[i] == 'gs':  # This is where the gs stuff will go
                    cps_prev[i]    = gsCp
                    Ss_prev[i]     = 0
                elif species_descriptor[i] == 'ls':  # This is where the ls stuff will go
                    cps_prev[i]    = lsCp
                    Ss_prev[i]     = 0
                else: # This is where the void stuff will go
                    cps_prev[i]    = 0
                    Ss_prev[i]     = 0

                # liquid calculations

                cpf_prev[i]  = fCp(pfT[i], percGly)  # Fluid specific heat at position i at temp. of previous time step
                rhof_prev[i] = fRho(pfT[i], percGly)  # Fluid Density at position i at temp. of previous time step
                muf_prev[i]  = fMu(pfT[i], percGly)  # Fluid Dynamic Visc. at position i at temp. of previous time step
                kf_prev[i]   = fK(pfT[i], percGly)  # Fluid thermal conduct. at position i at temp. of prev. time step

            ##################### DP: here is where the iteration at every time step begins ######################

            while not stepTol and stepCount <= maxSteps:  # Loop until stepTol is found or maxSteps is hit.
                # Note:
                # iynext is the guess n Fluid
                # isnext is the guess n Solid
                # y[n-1,:] is the n-1 Fluid solution
                # s[n-1,:] is the n-1 Solid solution

                # Grab Current State properties
                fT = iynext * (Thot - Tcold) + Tcold # DP: guess fluid temperatures for the current time step
                sT = isnext * (Thot - Tcold) + Tcold # DP: guess solid temperatures for the current time step

                # DP: The following variables are used in the construction of the system of algebraic equations,
                # the solid and fluid tridiagonal matrices

                Cs     = np.zeros(N + 1)
                ks     = np.zeros(N + 1)
                Smce   = np.zeros(N + 1)
                k      = np.zeros(N + 1)
                Omegaf = np.zeros(N + 1)
                Spres  = np.zeros(N + 1)
                Lf     = np.zeros(N + 1)

                Cf          = np.zeros(N + 1)
                rhof_cf_ave = np.zeros(N + 1)
                rhos_cs_ave = np.zeros(N + 1)
                Ff          = np.zeros(N + 1)
                Sp          = np.zeros(N + 1)

                ww = 0.5  # Weighted guess value
                pt[n] = 0  # Int the pressure
                dP = 0
                for i in range(N + 1):  # Calculate coefficients of system of algebraic equations

                    # DP: properties are calculated at the temperatures of the previous and current time steps
                    # (assumed temperature, which changes at every iteration) and the average is taken. I said average
                    # because the weighting value is 0.5. There are actually three options here: one is to calculate
                    # the properties at the temperature of the current time step, the second is to calculate the
                    # properties at the temperature of the previous time step, and the third is to take the average of
                    # both as it was chosen here. There is even a fourth option, which is calculate the properties at
                    # the average temperature between current and previous time step.

                    # Properties of fluid at location i and temperature of current time step
                    cpf_ave = fCp(fT[i], percGly) * ww + cpf_prev[i] * (1 - ww)  # Specific heat
                    rhof_ave = fRho(fT[i], percGly) * ww + rhof_prev[i] * (1 - ww)  # Density
                    muf_ave = fMu(fT[i], percGly) * ww + muf_prev[i] * (1 - ww)  # Dynamic Viscosity
                    kf_ave = fK(fT[i], percGly) * ww + kf_prev[i] * (1 - ww)  # Thermal conductivity
                    rhof_cf_ave[i] = cpf_ave * rhof_ave  # Combined rhof cf

                    fluid_dens[n, i] = fRho(fT[i], percGly)  # Added on 03/01/2022
                    mass_flow[n, i] = fluid_dens[n, i] * V[n]  # Added on 03/01/2022
                    Vf[n, i] = m_flow[n] / rhof_ave  # Added on 28/09/2022
                    ### Fluid term
                    # Ff[i] = (rhof_cf_ave[i] * Vf[n, i]) / L_tot  # Modified on 30/03/2023. This is possibly the error
                    Ff[i] = cpf_ave * m_flow[n] / L_tot # Modified on 31/03/2023.
                    # DP: this is divided by L_tot because in the FluidSolver function Ff is divided by
                    # dx = 1/(N+1) instead of DX. So, it is necessary to include L_tot so that dx*L_tot = DX

                    if species_descriptor[i].startswith("reg"):
                        cp_c = cp_c_if_list[materials.index(species_descriptor[i])]
                        cp_h = cp_h_if_list[materials.index(species_descriptor[i])]
                        ms_c = S_c_if_list[materials.index(species_descriptor[i])]
                        ms_h = S_h_if_list[materials.index(species_descriptor[i])]
                        # Note: T_c and T_h are not used thus they are not included here.
                        Reduct = reduct_coeff[species_descriptor[i].split('-')[1]]
                        # Old style [-----------
                        # if   species_descriptor[i]== 'reg-si1': cp_c = si1.mCp_c; cp_h = si1.mCp_h; ms_c = si1.mS_c; ms_h = si1.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct = 0.55;
                        # elif species_descriptor[i]== 'reg-si2': cp_c = si2.mCp_c; cp_h = si2.mCp_h; ms_c = si2.mS_c; ms_h = si2.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct = 0.77;
                        # elif species_descriptor[i]== 'reg-si3': cp_c = si3.mCp_c; cp_h = si3.mCp_h; ms_c = si3.mS_c; ms_h = si3.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct = 0.73;
                        # elif species_descriptor[i]== 'reg-si4': cp_c = si4.mCp_c; cp_h = si4.mCp_h; ms_c = si4.mS_c; ms_h = si4.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct = 0.75;
                        # elif species_descriptor[i]== 'reg-si5': cp_c = si5.mCp_c; cp_h = si5.mCp_h; ms_c = si5.mS_c; ms_h = si5.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct = 0.72;
                        # elif species_descriptor[i]== 'reg-Gd':  cp_c = Gd.mCp_c;  cp_h = Gd.mCp_h;  ms_c = Gd.mS_c;  ms_h = Gd.mS_h;  T_h=Gd.mTemp_h;  T_c = Gd.mTemp_c;  Reduct = 1;
                        # ------------]
                        # --- Calculation of internal magnetic field
                        Hint = appliedFieldm[n, i] * MFM[i]
                        int_field[n, i] = Hint  # Added on 03/04/2022
                        # --- Calculation of rho*cs fluid
                        dT = 0.5  # DP (5-4-2022) This big dT value helps avoiding some noise in the backward calc of Cp. Returned to 0.5 on 21/12/2022
                        Tr = psT[i]
                        aveField = (Hint + prevHintNew[i]) / 2  # current and previous time step average
                        dsdT = (ms_c(sT[i]+dT, Hint)[0, 0] * 0.5 + ms_h(sT[i]+dT, Hint)[0, 0] * 0.5) - (ms_c(sT[i]-dT, Hint)[0, 0] * 0.5 + ms_h(sT[i]-dT, Hint)[0, 0] * 0.5)
                        # TODO: not clear why aveField is used instead of Hint (current time step).
                        cps_curr = sT[i] * (np.abs(dsdT) / (dT * 2))  # TODO: should not sT[i] be used instead of Tr?
                        cps_ave = cps_curr * ww + cps_prev[i] * (1 - ww)  # DP: this is equation B.9 of Theo's thesis
                        rhos_cs_ave[i] = cps_ave * mRho
                        # --- Calculation of Smce
                        # DP: the heat delivered due to the MCE during the time step is calculated as an isothermal
                        # entropy change. So, the entropy of current and previous steps are calculated at the temp
                        # of the previous step.
                        # The anhysteretic entropy of the current time step is calculated at the magnetic field of the
                        # current time step

                        S_c_curr   = ms_c(Tr, Hint)[0, 0]
                        S_h_curr   = ms_h(Tr, Hint)[0, 0]
                        Sirr_cur   = S_c_curr * (1 - ch_factor[i]) - S_h_curr * ch_factor[i]
                        # Anhysteretic entropy calculated from cooling high field and heating high field entropy curves
                        Scur       = S_c_curr * (1 - ch_factor[i]) + S_h_curr * ch_factor[i]
                        #Mod        = 0.5*(Sirr_cur+Sirr_prev[i])*np.abs((2*dT)/dsdT)
                        Smce[i] = (Reduct * A_c[i] * (1 - e_r[i]) * mRho * Tr * (Sprev[i] - Scur)) / (DT * (Thot - Tcold))
                        Q_MCE = Q_MCE + Smce[i]*(Thot - Tcold)*DX*DT*freq  # Added on 21/03/2023 to see dif between Q_MCE and W_mag
                        # Eq. B.20 of Theo's thesis states that the entropy difference should be Scur-Sprev. So, there
                        # is a error in the thesis cuz it is ignoring the minus sign in front of the Qmce expression.

                        # DP: the properties of the fluid in the following functions were calculated above as the
                        # average of the properties at the temperatures of current and previous time steps

                        # --- Calculation of effective thermal conductivity for fluid
                        k[i] = kDyn_P(Dsp, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(Vf[n, i] / (A_c[i])))
                        k_disp[n, i] = k[i]  # Added on 15/09/2022 in order to calculate entropy generation
                        # --- Calculation of coefficient of heat transfer by convection term east of the P node
                        Omegaf[i] = A_c[i] * htc.beHeff(Dsp, np.abs(Vf[n, i] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave, freq, cps_ave, mK, mRho, e_r[i])  # Beta Times Heff
                        htc_fs[n, i] = Omegaf[i] / A_c[i] / (6 * (1 - e_r[i]) / Dsp)  # Added on 03/04/2022
                        # --- Calculation of the coefficient of the viscous dissipation term and pressure drop
                        Spres[i], dP = predrop.SPresM(Dsp, np.abs(Vf[n, i] / (A_c[i])), np.abs(Vf[n-1, i] / (A_c[i])), DT, np.abs(Vf[n, i]), e_r[i], muf_ave, rhof_ave, A_c[i] * e_r[i])

                        # DP: for spherical particles the following correction is not needed

                        # dP = dP * 2.7
                        # Spres[i] = Spres[i]*2.7

                        # DP: this factor is to compensate for the additional pressure drop occurring in beds of
                        # irregular shaped particles given that Ergun's correlation is for beds of spherical particles

                        # --- Calculation of the coefficient of the heat leaks term
                        Lf[i] = P_c[i] * leaks.ThermalResistance(Dsp, np.abs(Vf[n, i] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave, kg10, r1, r2, r3, casing_th, freq, air_th)
                        U_Pc_leaks[n, i] = Lf[i]
                        # --- Calculation of the effective thermal conductivity for the solid
                        ks[i] = kStat(e_r[i], kf_ave, mK)
                        k_stat[n, i] = ks[i]
                        # --- Calculation of capacitance of solid
                        Cs[i] = rhos_cs_ave[i] * A_c[i] * (1 - e_r[i]) * freq
                        # DP: freq is used here because in the SolveSolid function Cs is divided by dt = 1/(nt+1).
                        # So, this way dt becomes DT
                        # TODO: for glass and lead spheres change the ThermalResistance function
                    elif species_descriptor[i] == 'gs':
                        # This is where the gs stuff will go
                        # Effective Conduction for solid
                        rhos_cs_ave[i] = gsCp * gsRho
                        # Effective Conduction for fluid
                        k[i] = kDyn_P(Dspgs, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(Vf[n, i] / (A_c[i])))
                        # Forced convection term east of the P node

                        Omegaf[i] = A_c[i] * htc.beHeff(Dspgs, np.abs(Vf[n, i] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave,
                                                      freq, gsCp, gsK, gsRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        Spres[i], dP = predrop.SPresM(Dspgs, np.abs(Vf[n, i] / (A_c[i])), np.abs(Vf[n-1, i] / (A_c[i])), DT, np.abs(Vf[n, i]), e_r[i], muf_ave, rhof_ave,
                                              A_c[i] * e_r[i])
                        # Loss term
                        Lf[i] = P_c[i] * ThermalResistance(Dspgs, np.abs(Vf[n, i] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave,
                                                       kg10, r1, r2, r3)
                        # Effective Conduction for solid
                        ks[i] = kStat(e_r[i], kf_ave, gsK)
                        #Smce
                        Smce[i] = 0
                        ### Capacitance solid
                        Cs[i] = rhos_cs_ave[i] * A_c[i] * (1 - e_r[i]) * freq
                    elif species_descriptor[i] == 'ls':
                        # This is where the ls stuff will go
                        # Effective Conduction for solid
                        rhos_cs_ave[i] = lsCp * lsRho
                        # Effective Conduction for fluid
                        k[i] = kDyn_P(Dspls, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(Vf[n, i] / (A_c[i])))
                        # Forced convection term east of the P node
                        Omegaf[i] = A_c[i] * htc.beHeff(Dspls, np.abs(Vf[n, i] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave,
                                                      freq, lsCp, lsK, lsRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        Spres[i], dP = predrop.SPresM(Dspls, np.abs(Vf[n, i] / (A_c[i])), np.abs(Vf[n-1, i] / (A_c[i])), DT, np.abs(Vf[n, i]), e_r[i], muf_ave, rhof_ave,
                                              A_c[i] * e_r[i])
                        # Loss term
                        Lf[i] = P_c[i] * ThermalResistance(Dspls, np.abs(Vf[n, i] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave,
                                                       kg10, r1, r2, r3)
                        # Effective Conduction for solid
                        ks[i] = kStat(e_r[i], kf_ave, lsK)
                        #Smce
                        Smce[i] = 0
                        ### Capacitance solid
                        Cs[i] = rhos_cs_ave[i] * A_c[i] * (1 - e_r[i]) * freq
                    else:

                        k[i] = kf_ave
                        if species_descriptor[i] == 'void':  # Used for PM1 device
                            Lf[i] = P_c[i] * leaks.ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs, r1, r2, r3)
                        elif species_descriptor[i] == 'void1':  # Used for PM1 device
                            Lf[i] = P_c[i] * leaks.ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs1, r1, r2, r3)
                        elif species_descriptor[i] == 'void2':  # Used for PM1 device
                            Lf[i] = P_c[i] * leaks.ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs2, r1, r2, r3)
                        elif species_descriptor[i] == 'void3':  # Used for FAME cooler and 8Mag device
                            Lf[i] = P_c[i] * leaks.ThermalResistanceVoid(kair, kf_ave, kg10, freq, np.abs(Vf[n, i] / (A_c[i])), A_c[i], P_c[i], casing_th, air_th)
                        elif species_descriptor[i] == 'void4':  # Used for POLO device
                            Lf[i] = P_c[i] * leaks.ThermalResistanceVoid(kair, kf_ave, kg10, r1, r2, r3)
                        # No solid in the void
                        ks[i] = 0
                        # No interaction between solid and fluid since there is no solid.
                        Omegaf[i] = 0  #
                        # This will just make the plots nicer by having the solid temperature be the fluid temperature.
                        Cs[i] = 1 * freq
                        Smce[i] = (iynext[i]-s[n-1, i])/DT
                        #neglect pressure term.
                        # TODO: implement pressure drop term in void sections of the casing
                        Spres[i] = 0
                        dP = 0
                        # This is where the void stuff will go
                    dPdx[n, i] = dP  # [Pa/m] dP is actually dP/dx
                    pt[n] = dP * DX + pt[n]  # DP: it seems that the term dP is actually dP/dx
                    # DP: pt[n] returns the pressure drop along the regenerator at the current time step

                ### Capacitance fluid
                Cf = rhof_cf_ave * A_c * e_r * freq  # Freq is required here as Cf is divided by dt not DT in the solver
                # DP: in Numpy A*B is an element wise multiplication, which means that both matrices must have same size
                # DP: Matrix multiplication in the row-by-column way is performed using np.matmul(A,B)
                # where A is an lxm matrix and B is an mxn matrix so that A*B is an lxn matrix

                Sp = Spres / (Thot - Tcold)

                Kfw = np.zeros(N - 1)
                Kfe = np.zeros(N - 1)
                Ksw = np.zeros(N - 1)
                Kse = np.zeros(N - 1)

                for i in range(N - 1):  # Calculates the coefficients of the conduction terms
                    # DP: this runs from 0 to N-2, ghost nodes are excluded, aligns with 1->N-1

                    # Fluid Conduction term west of the P node

                    Kfw[i] = ((1 - fr[i]) / (A_c[i] * e_r[i] * k[i])
                              + (fr[i]) / (A_c[i+1] * e_r[i+1] * k[i+1])) ** -1
                    # DP: the first element in the vector Kfw includes info about the ghost node 0 and the last term
                    # includes info of the last node of the regenerator and the one to its left. The ghost node N is
                    # excluded

                    # Fluid Conduction term east of the P node

                    Kfe[i] = ((1 - fr[i+1]) / (A_c[i+1] * e_r[i+1] * k[i+1])
                              + (fr[i+1]) / (A_c[i+2] * e_r[i+2] * k[i+2])) ** -1
                    # DP: the first element of vector Kfe (index 0) includes info of node 1 and node 2 (spatial domain)
                    # and the last element of vector Kfe (index N-2) includes info of node N-1, last node of the
                    # regenerator and the node to its right, ghost node N.

                    # Solid Conduction term west of the P node

                    if ks[i] == 0 or ks[i+1] == 0:  # DP: when the node corresponds to a void space
                        Ksw[i] = 0
                    else:
                        Ksw[i] = ((1 - fr[i]) / (A_c[i] * (1-e_r[i]) * ks[i])
                                + (fr[i]) / (A_c[i+1] * (1-e_r[i+1]) * ks[i+1])) ** -1

                    # Solid Conduction term east of the P node

                    if ks[i+1] == 0 or ks[i+2] == 0:  # DP: when the node corresponds to a void space
                        Kse[i] = 0
                    else:
                        Kse[i] = ((1 - fr[i+1]) / (A_c[i+1] * (1-e_r[i+1]) * ks[i+1])
                                + (fr[i+1]) / (A_c[i+2] * (1-e_r[i+2]) * ks[i+2])) ** -1
                # TODO: an apparent error in the solid conduction terms was corrected. e_r was used instead of (1-e_r)
                # TODO: check if fr is used correctly
                Omegas = np.copy(Omegaf) # DP: this is the coefficient of the convection term, which is equal for both fluid and solid
                Kfw = Kfw / (DX*L_tot)
                Kfe = Kfe / (DX*L_tot)
                Ksw = Ksw / (DX*L_tot)
                Kse = Kse / (DX*L_tot)
                # DP: L_tot is included in these terms because in solve functions they are multiplied by dx = 1/(N+1)

                ################################################################
                ####################### SOLVE FLUID EQ      ####################
                # Fluid Equation
                ynext = SolveFluid(iynext, isnext, y[n-1, :], s[n-1, :], V[n], Cf, Kfe, Kfw, Ff, Omegaf, Lf, Sp, CF, CL, CVD, N, dx, dt, yamb)  # V is only used in this function to get the direction of flow. That is why it was not changed to Vf on 28/09/2022
                ################################################################
                ####################### SOLVE SOLID EQ      ####################
                # Solid Equation
                snext = SolveSolid(ynext, isnext, y[n-1, :], s[n-1, :], V[n], Cs, Kse, Ksw, Omegas, Smce, CS, CMCE, N, dx, dt)  # V is only used in this function to get the direction of flow. That is why it was not changed to Vf on 28/09/2022
                ################################################################
                ####################### CHECK TOLLERANCE    ####################
                # Check the tolerance of the current time step
                stepTol = AbsTolFunc(ynext, iynext, maxStepTol[stepTolInt])[0] and AbsTolFunc(snext, isnext, maxStepTol[stepTolInt])[0]  # DP: only get the boolean part of the function,
                # which indicates whether or not the absolute difference between the assumed and calculated temperature distributions is less than the predefined tolerance.
                # The second element that returns from the function AbsTolFunc is not used or not relevant here
                ################################################################
                ####################### DO housekeeping     ####################
                # Add a new step to the step count
                stepCount = stepCount + 1
                # Check if we have hit the max steps
                if (stepCount == maxSteps):
                    print("Hit max step count", flush=True)
                    print(AbsTolFunc(ynext, iynext, maxStepTol[stepTolInt]), flush=True)
                    print(AbsTolFunc(snext, isnext, maxStepTol[stepTolInt]), flush=True)
                    print(stepTol, flush=True)
                # Copy current values to new guess and current step.
                s[n, :] = np.copy(snext)  # DP: updated the current time step solid Temp distrib in the matrix containing all time steps temperature distributions
                isnext  = np.copy(snext)  # DP: uses the solid temperature just calculated for new time step iteration
                y[n, :] = np.copy(ynext)
                iynext  = np.copy(ynext)
                if (np.any(np.isnan(y)) or np.any(np.isnan(s))):
                    # Sometimes the simulation hits a nan value. We can redo this point later.
                    print(y, flush=True)
                    print(s, flush=True)
                    break
                # Break the step calculation
                if ((time.time()-t0)/60) > time_lim: # DP: time.time() returns the number of seconds from January 1st 1970 00:00:00.
                    # So, the difference time.time()-t0 is in seconds, and dividing it by 60 returns a difference in minutes
                    break # DP: this breaks the while loop for every time step
                ################################################################
                ################### ELSE ITERATE AGAIN - WHILE LOOP FOR EVERY TIME STEP FINISHES HERE ##################
            # break the cycle calculation
            if ((time.time()-t0)/60) > time_lim:
                break # DP: This breaks the for loop over the time steps
        # Check if current cycle is close to previous cycle.
        [bool_y_check, max_val_y_diff] = AbsTolFunc2d(y, iyCycle, maxCycleTol)
        [bool_s_check, max_val_s_diff] = AbsTolFunc2d(s, isCycle, maxCycleTol)

        # DP: change the time step tolerance
        if (max_val_y_diff/10) < maxStepTol[stepTolInt]:
            stepTolInt = stepTolInt + 1

            # ----------------------------- printing some results when tolerance changes -------------------------------
            # DP: 9/1/2023 Write some outputs every time it changes tolerance. This allows to see influence of tolerance
            coolingpowersum = 0
            heatingpowersum = 0
            P_pump_AMR_tol = 0
            Q_leak_tol = 0
            P_mag_AMR_tol = 0
            startint=0
            for n in range(startint, nt):
                # DP: this for loop is for the numerical integration of the equation 3.33 of Theo's thesis (which misses the integrand, dt).
                tF = y[n, 0] * (Thot - Tcold) + Tcold  # DP: temperature in [K] at the cold end at time step n
                tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold  # DP: temperature in [K] at the cold end at time step n+1
                coolPn = freq * fCp((tF+tF1)/2, percGly) * m_flow[n] * DT * ((tF+tF1)/2 - Tcold)
                coolingpowersum = coolingpowersum + coolPn

                # DP: Heating power
                tF_h = y[n, -1]*(Thot - Tcold) + Tcold
                tF1_h = y[n+1, -1]*(Thot - Tcold) + Tcold
                heatPn = freq*fCp((tF_h+tF1_h)/2, percGly) * m_flow[n] * DT * ((tF_h+tF1_h)/2 - Thot)
                heatingpowersum = heatingpowersum + heatPn

            qc = num_reg * coolingpowersum  # DP: 2 changed by 7 to account for the number of regenerators of the device
            qh = num_reg * heatingpowersum
            Tf_tol = y * (Thot - Tcold) + Tcold
            Ts_tol = s * (Thot - Tcold) + Tcold

            for j in range(nt+1):
                for i in range(N+1):
                    Q_leak_tol = Q_leak_tol + freq * U_Pc_leaks[j, i] * (Tf_tol[j, i] - Tambset) * DX * DT * CL[i]
                    P_pump_AMR_tol = P_pump_AMR_tol + freq * np.abs(Vf[j, i]) * dPdx[j, i] * DX * DT

            for i in range(N+1):
                ms_h = S_h_if_list[materials.index(species_descriptor[i])]
                P_mag_node = 0
                for n in range(nt):  # Ghost nodes excluded from this calculation
                    s_current = ms_h(Ts_tol[n, i], int_field[n, i])[0, 0]
                    s_next = ms_h(Ts_tol[n+1, i], int_field[n+1, i])[0, 0]
                    P_mag_node = P_mag_node + freq * mRho * (W_reg*H_reg*DX*(1-e_r[i])) * 0.5 * (Ts_tol[n, i] + Ts_tol[n+1, i]) * (s_next - s_current)  # [W] Magnetic power over the full cycle for the current node
                P_mag_AMR_tol = P_mag_AMR_tol + P_mag_node  # [W] Magnetic power over the entire AMR for the full cycle

            print("{0:<15} {1:<29} {2:<29} {3:20} {4:20} {5:<20} {6:<29} {7:<29} {8:<29} {9:<29}"
                  .format("CycleCount {:d}".format(cycleCount),
                          "Cooling Power {:2.5e}".format(qc), "Heating Power {:2.5e}".format(qh),
                          "y-tol {:2.5e}".format(max_val_y_diff), "s-tol {:2.5e}".format(max_val_s_diff),
                          "Run time {:6.1f} [min]".format((time.time()-t0)/60), "Pump power {:2.5e}".format(P_pump_AMR_tol), "Mag power {:2.5e}".format(P_mag_AMR_tol), "Heat leak {:2.5e}".format(Q_leak_tol), "Pout-Pin {:2.5e}".format(qh+Q_leak_tol-qc-P_pump_AMR_tol+P_mag_AMR_tol)), flush=True)

            # ---------------------------- printing some results when tolerance changes -------------------------------

            if stepTolInt == len(maxStepTol):  # DP: len([3, 6, 1, 4, 9]) returns 5, the number of elements in the list
                stepTolInt=len(maxStepTol)-1

        # DP: it is useful to see on screen some results during the iterative calculation process
        if cycleCount % 10 == 1: # DP: this is true for cycleCount = 11 or 21 or 31 and so on. The operator % returns the modulus of the division
            coolingpowersum = 0
            heatingpowersum = 0
            startint=0
            for n in range(startint, nt):
                # DP: this for loop is for the numerical integration of the equation 3.33 of Theo's thesis (which misses the integrand, dt).
                tF = y[n, 0] * (Thot - Tcold) + Tcold  # DP: temperature in [K] at the cold end at time step n
                tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold  # DP: temperature in [K] at the cold end at time step n+1
                coolPn = freq * fCp((tF+tF1)/2, percGly) * m_flow[n] * DT * ((tF+tF1)/2 - Tcold)
                coolingpowersum = coolingpowersum + coolPn

                # DP: Heating power
                tF_h = y[n, -1]*(Thot - Tcold) + Tcold
                tF1_h = y[n+1, -1]*(Thot - Tcold) + Tcold
                heatPn = freq*fCp((tF_h+tF1_h)/2, percGly) * m_flow[n] * DT * ((tF_h+tF1_h)/2 - Thot)
                heatingpowersum = heatingpowersum + heatPn

            qc = num_reg * coolingpowersum  # DP: 2 changed by 7 to account for the number of regenerators of the device
            qh = num_reg * heatingpowersum

            print("{0:<15} {1:<29} {2:<29} {3:20} {4:20} {5:<20}"
                  .format("CycleCount {:d}".format(cycleCount),
                          "Cooling Power {:2.5e}".format(qc), "Heating Power {:2.5e}".format(qh),
                          "y-tol {:2.5e}".format(max_val_y_diff), "s-tol {:2.5e}".format(max_val_s_diff),
                          "Run time {:6.1f} [min]".format((time.time()-t0)/60)), flush=True)

        if ((time.time()-t0)/60) > time_lim:  # DP: if the for loop was broken above, then do...
            coolingpowersum=0
            heatingpowersum=0
            startint=0
            for n in range(startint, nt):
                tF = y[n, 0] * (Thot - Tcold) + Tcold
                tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold
                coolPn =  freq * fCp((tF+tF1)/2,percGly) * m_flow[n] * DT * ((tF+tF1)/2 - Tcold)
                coolingpowersum = coolingpowersum + coolPn

                # DP: Heating power
                tF_h = y[n, -1]*(Thot - Tcold) + Tcold
                tF1_h = y[n+1, -1]*(Thot - Tcold) + Tcold
                heatPn = freq*fCp((tF_h+tF1_h)/2, percGly) * m_flow[n]*DT*((tF_h+tF1_h)/2-Thot)
                heatingpowersum = heatingpowersum + heatPn

            qc = num_reg * coolingpowersum  # DP: changed from 2 to 7. Number of regenerators
            qh = num_reg * heatingpowersum  # Added by DP

            print("{0:<15} {1:<29} {2:<29} {3:20} {4:20} {5:<20}"
                  .format("CycleCount {:d}".format(cycleCount),
                          "Cooling Power {:2.5e}".format(qc), "Heating Power {:2.5e}".format(qh),
                          "y-tol {:2.5e}".format(max_val_y_diff), "s-tol {:2.5e}".format(max_val_s_diff),
                          "Run time {:6.1f} [min]".format((time.time()-t0)/60)), flush=True)

            print('\nA pickle file will be saved')
            time_limit_reached = 1

            # # Pickle data
            # aaa = (y, s, stepTolInt, iyCycle, isCycle)
            # # open the file for writing
            # fileObject = open(PickleFileName, 'wb')
            # # this writes the object a to the
            # # file named 'testfile'
            # pickle.dump(aaa, fileObject)
            # # here we close the fileObject
            # fileObject.close()
            # print("Saving pickle file...", flush=True)
            # # Quit Program
            # sys.exit()

        Ts_last = s * (Thot - Tcold) + Tcold  # Last cycle solid temperature before updating initial condition for new cycle
        Tf_last = y * (Thot - Tcold) + Tcold  # Last cycle fluid temperature before updating initial condition for new cycle

        # Copy last value to the first of the next cycle.
        if cycleCount % 2 == 0:  # Convergence accelation implemented. Reference: Int J Refrig. 65 (2016) 250-257
            cycleTol = bool_y_check and bool_s_check  # DP comment: this will return True if both arguments are True, otherwise it returns False, which will keep the cycle while loop running
            s[0, :] = np.copy(s[-1, :]) + gain * (s[-1, :] - s[0, :])
            y[0, :] = np.copy(y[-1, :]) + gain * (y[-1, :] - y[0, :])
        else:
            if gain == 0:  # This is necessary in order to check tolerance every cycle when convergence acceleration option is not used
                cycleTol = bool_y_check and bool_s_check
            s[0, :] = np.copy(s[-1, :])
            y[0, :] = np.copy(y[-1, :])
        # Add Cycle
        cycleCount = cycleCount + 1
        # Did we hit the maximum number of cycles
        if cycleCount == maxCycles:
            print("Hit max cycle count\n", flush=True)
        # Copy current cycle to the stored value
        isCycle = np.copy(s)
        iyCycle = np.copy(y)
        if np.any(np.isnan(y)):
            # Sometimes the simulation hits a nan value. We can redo this point later.
            print(y, flush=True)
            break  # DP: this breaks the while loop for the cycle calculation
        # End Cycle
    t1 = time.time()
    ########################## END THE LOOP #########################

    # DP: the iterative calculation process on the cycle level ends up here

    if time_limit_reached == 0:

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DP: this section is not useful in my case %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        quart = int(nt/4)
        halft = int(nt/2)
        tquat = int(nt*3/4)
        # This is a modifier so we can modify the boundary at which we calculate the
        # effectiveness
        cold_end_node = np.min(np.argwhere([val.startswith("reg") for val in species_descriptor])) # 0
        # DP: the following argument: [val.startswith("reg") for val in species_descriptor] returns an array of False and True values after evaluating the condition
        # startwith("reg"). This array is of the same length as species_descriptor
        # DP: np.argwhere returns an array with the positions of the True values obtained from [val.startswith("reg") for val in species_descriptor].
        # DP: Finally, np.min gives the position of the first node at which the word reg is found, i.e. where the regenerator starts along the flow path
        hot_end_node   = np.max(np.argwhere([val.startswith("reg") for val in species_descriptor]))  # -1
        eff_HB_CE = np.trapz((1-y[halft:,  cold_end_node]), x=t[halft:]) / (tau_c/2) # DP: this is in agreement with equation 8-373 of the book of Nellis and Klein
        # DP: Effectiveness over the hot blow, which apparently occurs during the second half of the period
        # DP: this function integrates the first element over the domain given by the second element.
        eff_CB_HE = np.trapz(y[:halft+1,  hot_end_node], x=t[:halft+1]) / (tau_c/2)  # DP: Effectiveness over the cold blow
        # TODO: the blow periods in the FAME cooler are shorter than tau_c/2 because the volumetric flow rate profile is not sinusoidal
        tFce = np.zeros(nt+1)
        tFhe = np.zeros(nt+1)
        yEndBlow = np.zeros(N+1)
        yHalfBlow = np.zeros(N+1)
        sEndBlow = np.zeros(N+1)
        sHalfBlow = np.zeros(N+1)

        yMaxCBlow = np.zeros(N+1)
        yMaxHBlow = np.zeros(N+1)
        sMaxCBlow = np.zeros(N+1)
        sMaxHBlow = np.zeros(N+1)

        tFce = y[:,  cold_end_node] * (Thot - Tcold) + Tcold # DP: temperature in [K] of the fluid at the cold end of the regenerator during the cycle
        tFhe = y[:, hot_end_node] * (Thot - Tcold) + Tcold # DP: temperature in [K] of the fluid at the hot end of the regenerator during the cycle

        # DP: probably, given the sine shape of the volumetric flow with respect to time, the max values of the blow processes occur at 1/4 and 3/4 of the period
        yMaxCBlow  = y[quart, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the fluid in [K] at the instant tau/4
        yMaxHBlow = y[tquat, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the fluid in [K] at the instant tau*3/4

        yEndBlow  = y[-1, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the fluid in [K] at the end of the cycle
        yHalfBlow = y[halft, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the fluid in [K] at half time the cycle period

        sMaxCBlow  = s[quart, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the solid in [K] at the instant tau/4
        sMaxHBlow = s[tquat, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the solid in [K] at the instant tau*3/4

        sEndBlow  = s[-1, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the solid in [K] at the end of the cycle
        sHalfBlow = s[halft, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the solid in [K] at half time the cycle period

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        coolingpowersum=0
        power_in_out_cold_side = 0
        Qc_var_cp = 0
        startint=0
        # DP: this is the numerical integration of freq*integral(m*Cp*(Tf,cold_end-Tcold)*dt) from 0 to tau, equation 3.33 of Theo's thesis.
        # It seems that it is performed following a rectangle rule. https://en.wikipedia.org/wiki/Numerical_integration
        for n in range(startint, nt):
            tF = Tf_last[n, 0]
            tF1 = Tf_last[n+1, 0]
            coolPn = freq * fCp((tF+tF1)/2, percGly) * m_flow[n] * DT * ((tF+tF1)/2 - Tcold)
            coolingpowersum = coolingpowersum + coolPn
            power_in_out_cold_side = power_in_out_cold_side + freq * ((fCp(tF, percGly) + fCp(tF1, percGly)) / 2) * m_flow[n] * DT * (tF - Tcold)

            Trange = np.linspace(tF, Tcold, 500)
            # dT = (tF-Thot)/(500-1)
            for i in range(500-1):
                Qc_var_cp = Qc_var_cp + freq * np.abs(m_flow[n]) * (fCp(Trange[i], percGly) + fCp(Trange[i+1], percGly)) / 2 * (Trange[i+1]-Trange[i]) * DT

        qc = num_reg * coolingpowersum  # [W] Gross cooling power of the device

        heatingpowersum=0
        power_in_out_hot_side = 0
        Qh_var_cp = 0
        startint=0
        for n in range(startint, nt):
            tF = Tf_last[n, -1]
            tF1 = Tf_last[n+1, -1]
            heatPn = freq * fCp((tF+tF1)/2, percGly) * m_flow[n] * DT * ((tF+tF1)/2-Thot)
            heatingpowersum = heatingpowersum + heatPn
            power_in_out_hot_side = power_in_out_hot_side + freq * ((fCp(tF, percGly) + fCp(tF1, percGly)) / 2) * m_flow[n] * DT * (tF-Thot)

            Trange = np.linspace(Thot, tF, 500)
            # dT = (tF-Thot)/(100-1)
            for i in range(500-1):
                Qh_var_cp = Qh_var_cp + freq * m_flow[n] * (fCp(Trange[i], percGly) + fCp(Trange[i+1], percGly)) / 2 * (Trange[i+1]-Trange[i]) * DT

        qh = num_reg * heatingpowersum  # [W] Heating power of the device
        print('Power in out cold side = {} [W]'.format(power_in_out_cold_side))
        print('Power in out hot side = {} [W]'.format(power_in_out_hot_side))

        # Cooling power of FAME cooler is 7 times the cooling power of one regenerator.
        # Demonstrated in the file Cooling_capacity_calc.py

        Kamb = 2.5  # DP: This has to be measured experimentally for the
        qccor = qc - Kamb * (Tambset-Tcold)  # DP: this corresponds to the net power output, equation 3.34 of Theo's thesis.
        # It includes a correction to account for additional heat leaks in the CHEX

        # Calculation of average pressure drop across the regenerator

        pave = np.trapz(pt[halft:], x=t[halft:]) / (tau_c/2)  # DP: average pressure drop across the regenerator

        print("\n{:<15} {:<29} {:<29} {:20} {:20} {:<20}"
              .format("CycleCount {:d}".format(cycleCount-1),
                      "Cooling Power {:2.5e}".format(qc), "Heating Power {:2.5e}".format(qh),
                      "y-tol {:2.5e}".format(max_val_y_diff), "s-tol {:2.5e}".format(max_val_s_diff),
                      "Run time {:4.1f} [min]".format((t1-t0)/60)), flush=True)

        # print('Effectiveness HB-CE {} CB-HE {}'.format(eff_HB_CE, eff_CB_HE)) # TODO check whether effectiveness is useful
        print('\nMaximum pressure drop: {:.3f} (kPa)'.format(np.amax(pt)/1000), flush=True)
        print('Average pressure drop: {:.3f} (kPa)'.format(pave/1000), flush=True)

        print('\nValues found at minimal field', flush=True)
        print('min Applied Field: {:.3f}'.format(minAplField), flush=True)
        print('min Internal Field: {:.3f}'.format(minPrevHint), flush=True)
        print('min Magnetic Temperature: {:.3f}'.format(minMagTemp), flush=True)
        print('min Cp: {:.3f}'.format(minCpPrev), flush=True)
        print('min SS:{:.3f}'.format(minSSprev), flush=True)
        print('Lowest Temperature found in the SS cycle: {:.3f}'.format(minTemp), flush=True)

        print('\nValues found at maximum field', flush = True)
        print('max Applied Field: {:.3f}'.format(maxAplField), flush=True)
        print('max Internal Field: {:.3f}'.format(maxPrevHint), flush=True)
        print('max Magnetic Temperature: {:.3f}'.format(maxMagTemp), flush=True)  # DP: this probably refers to Max MCM Temperature
        print('max Cp: {:.3f}'.format(maxCpPrev), flush = True)
        print('max SS:{:.3f}'.format(maxSSprev), flush = True)
        print('highest Temperature found in the SS cycle: {:.3f}'.format(maxTemp), flush=True)

        # Remove Pickle
        try:
            os.remove(PickleFileName)
            print("\nWe removed the pickle file", flush=True)
        except FileNotFoundError:
            print('\nThe calculation converged!', flush=True)

        # -------------------------------------- Entropy generation calculations --------------------------------------

        # Note: feature added on 27/09/2022
        # Equations are based on: T. Lei et al. / Applied Thermal Engineering 111 (2017) 1232–1243

        S_ht_hot = 0
        S_ht_cold = 0
        S_ht_fs = 0
        S_ht_amb = 0
        S_vd = 0
        S_condu_stat = 0
        S_condu_disp = 0

        P_pump_AMR = 0
        Q_leak = 0


        #Ts = s * (Thot - Tcold) + Tcold
        beta = 6 * (1 - er) / Dsp

        for j in range(nt+1):
            cp_f_hot_ave = fCp((Tf_last[j, -1] + Thot) / 2, percGly)
            cp_f_cold_ave = fCp((Tf_last[j, 0] + Tcold) / 2, percGly)
            S_ht_hot = S_ht_hot + freq * np.abs(m_flow[j]) * cp_f_hot_ave * (np.log(Thot / Tf_last[j, -1]) + (Tf_last[j, -1] - Thot) / Thot) * DT
            S_ht_cold = S_ht_cold + freq * np.abs(m_flow[j]) * cp_f_cold_ave * (np.log(Tcold / Tf_last[j, 0]) + (Tf_last[j, 0] - Tcold) / Tcold) * DT
            for i in range(N+1):
                S_ht_amb = S_ht_amb + freq * U_Pc_leaks[j, i] * (Tambset - Tf_last[j, i])**2 * DX * DT / (Tambset * Tf_last[j, i])
                Q_leak = Q_leak + freq * U_Pc_leaks[j, i] * (Tf_last[j, i] - Tambset) * DX * DT * CL[i]
                S_ht_fs = S_ht_fs + freq * htc_fs[j, i] * beta * Ac * (Tf_last[j, i] - Ts_last[j, i])**2 * DX * DT / (Tf_last[j, i] * Ts_last[j, i])
                P_pump_AMR = P_pump_AMR + freq * np.abs(Vf[j, i]) * dPdx[j, i] * DX * DT
                S_vd = S_vd + freq * np.abs(Vf[j, i]) * dPdx[j, i] * DX * DT / Tf_last[j, i]
                if i==0:
                    dTsdx = (Ts_last[j, i+1] - Ts_last[j, i]) / DX
                    dTfdx = (Tf_last[j, i+1] - Tf_last[j, i]) / DX
                elif i==nodes:
                    dTsdx = (Ts_last[j, i] - Ts_last[j, i-1]) / DX
                    dTfdx = (Tf_last[j, i] - Tf_last[j, i-1]) / DX
                else:
                    dTsdx = (Ts_last[j, i+1] - Ts_last[j, i-1]) / (2 * DX)
                    dTfdx = (Tf_last[j, i+1] - Tf_last[j, i-1]) / (2 * DX)
                S_condu_stat = S_condu_stat + freq * k_stat[j, i] * Ac * (1 - e_r[i]) * dTsdx**2 * DX * DT / Ts_last[j, i]**2
                S_condu_disp = S_condu_disp + freq * k_disp[j, i] * Ac * e_r[i] * dTfdx**2 * DX * DT / Tf_last[j, i]**2

        # ----------------------------- 27/10/2022 Calculation of magnetic power input -------------------------------------
        P_mag_AMR = 0
        W_mag = 0
        for i in range(1, N, 1):  # Ghost nodes excluded
            ms_h = S_h_if_list[materials.index(species_descriptor[i])]
            P_mag_node = 0
            for n in range(nt+1):
                s_current = ms_h(Ts_last[n, i], int_field[n, i])[0, 0]
                if n == nt:
                    s_next = ms_h(Ts_last[0, i], int_field[0, i])[0, 0]
                    P_mag_node = P_mag_node + freq * mRho * (W_reg*H_reg*DX*(1-e_r[i])) * 0.5 * (Ts_last[n, i] + Ts_last[0, i]) * (s_next - s_current)  # [W] Magnetic power over the full cycle for the current node
                    W_mag = W_mag + freq * mRho * (W_reg*H_reg*DX*(1-e_r[i])) * ((0.5 * Ts_last[n, i] * (ms_h(Ts_last[n, i]+0.5, int_field[n, i])[0, 0] - ms_h(Ts_last[n, i]-0.5, int_field[n, i])[0, 0]) + 0.5 * Ts_last[0, i] * (ms_h(Ts_last[0, i]+0.5, int_field[0, i])[0, 0] - ms_h(Ts_last[0, i]-0.5, int_field[0, i])[0, 0])) * (Ts_last[0, i] - Ts_last[n, i])
                                                                                 + (Ts_last[n, i] * (ms_h(Ts_last[n, i], int_field[0, i])[0, 0] - ms_h(Ts_last[n, i], int_field[n, i])[0, 0])))
                else:
                    s_next = ms_h(Ts_last[n+1, i], int_field[n+1, i])[0, 0]
                    P_mag_node = P_mag_node + freq * mRho * (W_reg*H_reg*DX*(1-e_r[i])) * 0.5 * (Ts_last[n, i] + Ts_last[n+1, i]) * (s_next - s_current)  # [W] Magnetic power over the full cycle for the current node
                    W_mag = W_mag + freq * mRho * (W_reg*H_reg*DX*(1-e_r[i])) * ((0.5 * Ts_last[n, i] * (ms_h(Ts_last[n, i]+0.5, int_field[n, i])[0, 0] - ms_h(Ts_last[n, i]-0.5, int_field[n, i])[0, 0]) + 0.5 * Ts_last[n+1, i] * (ms_h(Ts_last[n+1, i]+0.5, int_field[n+1, i])[0, 0] - ms_h(Ts_last[n+1, i]-0.5, int_field[n+1, i])[0, 0])) * (Ts_last[n+1, i] - Ts_last[n, i])
                                                                            + (Ts_last[n, i] * (ms_h(Ts_last[n, i], int_field[n+1, i])[0, 0] - ms_h(Ts_last[n, i], int_field[n, i])[0, 0])))
            P_mag_AMR = P_mag_AMR + P_mag_node  # [W] Magnetic power over the entire AMR for the full cycle
        # ------------------------------------------------------------------------------------------------------------------
        error1 = abs((qh+Q_leak-qc)-(P_pump_AMR-P_mag_AMR))*100/(P_pump_AMR-P_mag_AMR)
        error2 = abs((Qh_var_cp+Q_leak-Qc_var_cp)-(P_pump_AMR-P_mag_AMR))*100/(P_pump_AMR-P_mag_AMR)
        print('Enthalpy in out cold side = {} [W]'.format(Qc_var_cp), flush=True)
        print('Enthalpy in out hot side = {} [W]'.format(Qh_var_cp), flush=True)
        print('Cycle average cooling capacity = {} [W]'.format(qc), flush=True)
        print('Cycle average heating capacity = {} [W]'.format(qh), flush=True)
        print('Cycle average heat leaks = {} [W]'.format(Q_leak), flush=True)
        print('Cycle average pumping power = {} [W]'.format(P_pump_AMR), flush=True)
        print('Cycle average magnetic power = {} [W]'.format(P_mag_AMR), flush=True)
        print('error in power input 1 = {} [%]'.format(error1), flush=True)
        print('error in power input 2 = {} [%]'.format(error2), flush=True)
        print('outputs,{},{},{},{},{},{},{},{},{}'.format(qc, qh, Q_leak, P_pump_AMR, P_mag_AMR, error1, power_in_out_cold_side, power_in_out_hot_side, error2), flush=True)
        print('Q_MCE = {}'.format(Q_MCE), flush=True)

        return Thot, Tcold, qc, qccor, (t1-t0)/60, pave, eff_HB_CE, eff_CB_HE, tFce, tFhe, yHalfBlow, yEndBlow, sHalfBlow, \
               sEndBlow, y, s, pt, np.max(pt), Uti, freq, t, xloc, yMaxCBlow, yMaxHBlow, sMaxCBlow, sMaxHBlow, qh, \
               cycleCount, int_field, htc_fs, fluid_dens, mass_flow, dPdx, k_stat, k_disp, S_ht_hot, S_ht_cold, S_ht_fs, \
               S_vd, S_condu_stat, S_condu_disp, S_ht_amb, P_pump_AMR, P_mag_AMR, Q_leak, Qc_var_cp, Qh_var_cp
        # TODO remove from return the input parameters such as Thot, Tcold, freq, xloc

    elif time_limit_reached == 1:
        return y, s, stepTolInt, iyCycle, isCycle
# ------------------ DP: the function "Run_Active" ends here ----------------------------
