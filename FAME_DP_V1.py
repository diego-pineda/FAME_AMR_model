# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 16:50:31 2017

@author: Theo
"""
# mpi4py
# from mpi4py import MPI
# multiprocessing
#from multiprocessing import Pool
#from multifuntest import sqrt
#from patankarreggd_one_void import runActive
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


################################# Si Material properties  ####################################
#
# Naming convention
#
# Specific heat function
#
# mCp_c/h
#
# Magnetization
#
# mMag_c/h
#
# Entropy
#
# mS_c/h
#
# Temp
#
# mTemp_c/h


from sourcefiles.mat import si1
from sourcefiles.mat import si2
from sourcefiles.mat import si3
from sourcefiles.mat import si4
from sourcefiles.mat import si5
from sourcefiles.mat import Gd

######################################## Fluid properties  ##########################################

# Density
from sourcefiles.fluid.density import fRho
# Dynamic Viscosity
from sourcefiles.fluid.dynamic_viscosity import fMu
# Specific Heat
from sourcefiles.fluid.specific_heat import fCp
# Conduction
from sourcefiles.fluid.conduction import fK

######################################## CLOSURE RELATIONSHIPS ######################################


# Dynamic conduction
from closure.dynamic_conduction import kDyn_P
# Static conduction
from closure.static_conduction import kStat
# Internal Heat transfer coefficient * Specific surface area
from closure.inter_heat import beHeff_I, beHeff_E
# pressure Drop
from closure.pressure_drop import SPresM
# Resistance Term in the Regenerator and void
from closure.resistance import ThermalResistance,ThermalResistanceVoid
from closure.FAME_resistance import FAME_ThermalResistance, FAME_ThermalResistance2, FAME_ThermalResistanceVoid


############################################## SOLVER ##############################################
############################################## TDMA   ##############################################
from core.tdma_solver import TDMAsolver


######################################## EXPONENTIAL SCHEME #########################################

@jit
def alpha_exp(Pe):
    # Exponential Scheme
    val = np.abs(Pe)/(np.expm1(np.abs(Pe)))
    return np.max([0,val])



######################################## SOLID SOLVER ##############################################
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

@jit(f8(f8),nopython=True)
def alpha_pow(Pe):
    # Powerlaw
    val = (1-0.1*abs(Pe))**5
    return max(0,val)


######################################## FLUID SOLVER ##############################################
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
    ynext = np.zeros(N+1) # DP: this creates a position for the temperature of every node including the ghost nodes.


    if Vd>0:
        # Add a value at the start
        # Add bc, value to the start of the flow
        # Dirichlet ghost node
        ynext[0]=0
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N.
            # DP: this actually loops from indices 0 to N-2. The system of algebraic equations to solve has the form:
            # DP: a[j]y[i-1]+b[j]y[i]+c[j]y[i+1]=d[j]
            # This is easier to understand when considering j and i ranging from index 1 to N-1. This way it is
            # clear that N-1 equations are created with its coefficients and the temperatures of all nodes from 0 to N
            # are considered in the system of equations. Note that the index of the nodes are relevant in the right hand
            # side terms of the following equations.
            # Build tridagonal matrix coefficients
            # pg 112-113 Theo Christiaanse 2017
            Aw=alpha_pow(Ff[j]/Kfw[j]) # TODO: determine why using this?
            Ae=alpha_pow(Ff[j+1]/Kfe[j])
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
    elif Vd<0:
        # Add a value at the end
        ynext[-1]=1
        for j in range(N-1):  # This will loop through 1 to N+1 which aligns with 0->N
            # Build tridagonal matrix coefficients
            # pg 112-113 Theo Christiaanse 2017
            Aw=alpha_pow(Ff[j+1]/Kfw[j])
            Ae=alpha_pow(Ff[j+2]/Kfe[j])
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
        ynext[0]=ynext[1]
        return ynext
    else:

        for j in range(N-1):  # DP: this loops from indices 0 to N-2.

            # Build tridiagonal matrix coefficients
            # DP: centered discretization implemented

            Aw=alpha_pow(Ff[j]/Kfw[j]) # TODO: determine why using this?
            Ae=alpha_pow(Ff[j+1]/Kfe[j])
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



####################################### LOOP FUNC ##############################################
@jit(nb.types.Tuple((b1,f8))(f8[:],f8[:],f8))
def AbsTolFunc(var1,var2,Tol):
    maximum_val=np.max(np.abs(var1-var2))
    return maximum_val<=Tol,maximum_val

@jit(nb.types.Tuple((b1,f8))(f8[:,:],f8[:,:],f8))
def AbsTolFunc2d(var1,var2,Tol):
    maximum_val=np.max(np.abs(var1-var2))
    return maximum_val<=Tol,maximum_val

######################################### RUN ACTIVE ############################################

def runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName,time_lim,cycle_tol,max_step_iter,max_cycle_iter):
    '''
    # runActive : Runs a AMR simulation of a pre-setup geometry
    # Arguments :
    # caseNum        <- caseNum number
    # Thot           <- Hot side heat exchanger
    # Tcold          <- Cold side heat exchanger
    # cen_loc        <- offset of the regenerator to the magnet
    # Tambset        <- Set ambient temperature
    # dispV          <- Displaced volume [m^3]. DP: for the FAME cooler this represents the maximum volumetric flow rate
    # ff             <- frequency [Hz]
    # CF             <- Enable/Disable Conduction term in Fluid
    # CS             <- Enable/Disable Conduction term in Solid
    # CL             <- Enable/Disable Heat leaks term in the Fluid GE
    # CVD            <- Enable/Disable Viscous Dissipation Term in the Fluid GE
    # CMCE           <- Enable/Disable MCE
    # nodes          <- Number of Spacial nodes used
    # timesteps      <- Number of Timesteps per cycle
    # ConfName       <- Load a certain configuration file
    # jobName        <- The name of the job
    # time_lim       <- Simulation time limit in minutes (Added by DP)
    # cycle_tol      <- Maximum cycle tolerance: criterion for the end of the iterative calculation process
    # max_step_iter  <- Maximum time step iterations the simulation is allowed to take
    # max_cycle_iter <- Maximum cycle iterations the simulation is allowed to take

    ########### 14-9-2017 16:39
    code has been check on NTU and U correctness. Some interesting spike was found
    when fluid speed hit zero. This can be solved by using the eps method which worked
    in COMSOL as well.
        - Moving forward implementing fluid and solid properties.
    ########### 18-9-2017 08:43
    Fluid and Solid properties have been inplemented.
    Code has been change to a function.
    Pressure drop term has been implemented however, need to redo the math on
    term.
    ###########    ''     09:20
    Implemented pressure and leak terms. Math checks out. Should be good to go
    and do some spatial and temporal resolution tests.
         - Need to implement Glass spheres and Void space options. Should not
           be difficult as I've already implemented the distretization of the
           discription and build placer functions to implement the different
           closure functions.
    ###########     ''      13:33
    Cleaning up the functions so only what is changing per time step is taken
    as an imput to the function. This makes the code a lot more readable.
    ########### 14-11-2017 14:17
    This version of the code is ported from the V4 version before the gradient
    porosity was added. The code has modified to activate the field again.
    '''

    # Import the configuration
    if ConfName == "R1":
        from configurations.R1  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
            percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R2":
        from configurations.R2  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
            percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R3":
        from configurations.R3  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
            percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R4":
        from configurations.R4  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
            percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R5":
        from configurations.R5  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
            percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R6":
        from configurations.R6  import Ac,Dspgs,Dspls,L_add,L_reg1, L_reg2, MOD_CL,Nd,Pc,egs,els,er,gsCp,gsK,gsRho,kair,kg10,kult,lsCp,lsK,lsRho,mK,mRho, \
            percGly,r1,r2,r3, rvs,rvs1,rvs2,species_discription,x_discription,CL_set,ch_fac
    if ConfName == "R7":
        from configurations.R7 import Ac, Nd, MOD_CL, Pc, er, kair, kg10, kult, mK, mRho, percGly, species_discription, x_discription, CL_set, ch_fac, casing_th, air_th

    # TODO: check if the variables left out in configuration R7 are necessary or not.
    print("Hot Side: {} Cold Side: {}".format(Thot,Tcold))
    # Start Timer
    t0 = time.time()
    # Volume displacement in [m3]
    # Small displacer 2.53cm^2
    # Medium displacer 6.95cm^2
    # 1inch = 2.54cm
    Vd      = dispV # DP: for the FAME cooler this is the maximum volumetric flow rate
    freq    = ff
    tau_c   = 1/freq # DP comment: this is the period of the cycle

    # Number of spatial nodes
    N = nodes
    # Number of time steps
    nt = timesteps
    # Molecule size
    dx = 1 / (N+1) # DP comment: I think the denominator in this case is N+1 because there is a node zero
    # The time step is:
    dt = 1 / (nt+1) # DP comment: I am not sure why in this case the denominator is nt+1.  Maybe it is also because there a time node for zero.
    # To prevent the simulation running forever we limit the simulation to
    # find a tollerance setting per step and per cycle.

    # DP: Create a list (maxStepTol) containing the values of the tolerances used to finish the iterative calculation
    # process for every time step. The list is needed to avoid many iterations on the time step level when the tolerance
    # on the cycle level is still far from the criterion for convergence

    maxCycleTol = cycle_tol  # DP: this was originally 1e-6
    maxStepTol  = [1]  # DP: this was originally [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    a = 0
    while maxStepTol[-1] >= maxCycleTol:
        a = a+1
        maxStepTol.append(10**-a)

    # We also limit the number of steps and cycles the simulation can take. If
    # the simulation runs over these there might be an issue with the code.
    maxSteps  = max_step_iter  # DP: this was 2000 in Theo's code. I set this to 200
    maxCycles = max_cycle_iter  # DP: this was 2000 in Theo's code. I set this to 300

    print("Number of cycle iterations: {}\nNumber of time step iterations: {}".format(maxCycles, maxSteps))
    # Cycle period
    t = np.linspace(0, tau_c, nt+1)
    # Total length of the domain
    L_tot   = np.max(x_discription)  # [m]
    # Real element size
    # Element size
    DX = L_tot/ (N+1) # DP comment: this means that the discretization takes into account not only the AMR bed but also the voids and glass-sphere beds
    # TODO: verify why DX is defined this way. In my opinion it should be DX = L_tot/(N-1) because nodes N and 0 are ghost nodes.
    #  That is even the definition shown in equation B.1 of Theo's thesis
    # Time step
    DT = tau_c/(nt+1) # DP comment: ok

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

    from sourcefiles.device import FAME_V_flow
    V = FAME_V_flow.vol_flow_rate(nt, Vd)  # DP: Vd for the FAME cooler is the maximum volumetric flow rate in m^3/s

    pdrop = lambda at, dP, sf: (dP) * sf * np.pi * np.sin(2 * np.pi * sf * at) + np.sign(np.sin(2 * np.pi * sf * at)) * sys.float_info.epsilon * 2
    # DP comment: Not very clear what this function does
    dPreg    = 5.2 * 6894.7572931783/2
    Lreg_exp = 22.5e-3
    ddP      = pdrop(t, dPreg, freq)
    dPdz_exp = ddP/Lreg_exp
    #U = uf(t, 1, 1, freq)

    # Calculate the utilization as defined by Armando's paper
    #Uti     = (Vd * 1000 * 4200) / (1000 * Ac * (1  - er) * 6100 * (L_reg1+L_reg2))# TODO by DP: redefine utilization for the FAME cooler
    Uti = (Vd * 1000 * 4200) / (1000 * Ac * (1 - er) * 6100 * (L_tot-0.012)) # DP: 0.012 is the length of the voids
    # DP comment: 6100 is the density of the MCM. 4200 is the Cp of water-glycol mixture. 1000 in the numerator is the density of water.
    # DP comment: 1000 in the denominator is an approximate value of Cp of MCM.
    print('Utilization: {0:1.3f} Frequency: {1:1.2f} [Hz]'.format(Uti,freq))
    print('Urms: {0:3.3f}'.format((Vd / Ac*er) * freq * np.pi*1/np.sqrt(2)))

    # Initial ch-factor
    ch_factor = np.ones(N + 1)*ch_fac # DP comment: ch_fac = 0.5 is set in the configuration file. This is the averaging cooling and heating factor

    # This is modification of the casing BC. DP comment: in the configuration file, CL_set = "grad" and MOD_CL=0
    if CL_set=="Tamb":
        # Ambiant Temperature non-diamentionilized
        yamb = np.ones(N + 1) * ((Tambset - Tcold) / (Thot - Tcold))
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
    int_discription = np.zeros(N+1,dtype=np.int)
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
        elif species_descriptor[i]=='void':
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
   #     for n in range(0, nt+1):
   #          #Will only get the field if we find a regenerator
   #         if (species_descriptor[i].startswith("reg")):
   #             x_pos_w_respect_to_magnet = xloc[i] - magOffset
   #             appliedFieldm[n, i] = hapl.appliedField(x_pos_w_respect_to_magnet, rotMag[n])[0, 0]*CMCE
   #         else:
   #             appliedFieldm[n, i] = 0

    # Applied field FAME cooler
    from sourcefiles.device import FAME_app_field
    appliedFieldm = FAME_app_field.app_field(nt, N)

    for i in range(N + 1):
        if (not species_descriptor[i].startswith("reg")):
            appliedFieldm[:, i] = 0


    ########################## START THE LOOP #########################

    # Some housekeeping to make this looping work

    #
    # Initial temperature
    y1 = np.linspace(0,1, N + 1) # DP comment: Initial fluid temperature. Linear distribution from Tcold to Thot
    s1 = np.linspace(0, 1, N + 1) # DP comment: Initial solid temperature. Linear distribution from Tcold to Thot

    #
    # Check is there is some pickeled data
    PickleFileName = "./pickleddata/{0:}-{1:d}".format(jobName,int(caseNum))
    print("Pickle Data File: {}".format(PickleFileName))
    try:
        # we open the file for reading
        fileObject = open(PickleFileName,'rb')
        print("we are loading the pickle file!")
        # load the object from the file into var b
        bbb = pickle.load(fileObject)
        y   = bbb[0]
        s   = bbb[1]
        stepTolInt = bbb[2]
        iyCycle = bbb[3]
        isCycle = bbb[4]
    except FileNotFoundError:
        # Keep preset values
        print("started normal")
        y = np.ones((nt+1, N + 1))*y1 # DP comment: initial temperature distribution for every time step is set to a linear distribution from Tcold to Thot
        s = np.ones((nt+1, N + 1))*s1
        stepTolInt = 0
        # Initial guess of the cycle values.
        iyCycle = np.copy(y)
        isCycle = np.copy(s)

    # Magnetic Field Modifier
    MFM = np.ones(N + 1)
    #
    cycleTol   = 0 # DP comment: this is equivalent to a boolean False
    cycleCount = 1 # DP comment: it was defined above that the maximum number of cycle iterations is 2000

    ########### DP comment: the iterative calculation process for the cycle starts here ###########

    while (not cycleTol  and cycleCount <= maxCycles): # DP comment: "not cycleTol" evaluates if cycleTol is zero or False and return True if so...
        # Account for pressure every time step (restart every cycle)
        pt = np.zeros(nt + 1) # DP: it seems this refers to a pressure drop along the regenerator
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

        # Vacuum permeability constant
        mu0     = 4 * 3.14e-7  # [Hm^-1] # TODO: it seems that this constant can be defined outside the loop

        for i in range(N+1): # DP comment: This for loop intends to find a vector of Magnetic Field Modifiers for every position i along the flow path to account for the demagnetizing field in the MCM regenerators
            # Average Solid temperature
            Ts_ave=np.mean(s[:,i]* (Thot - Tcold) + Tcold) # DP comment: this returns the average value of a vector containing the solid temperatures at position i for all time steps
            # Maximum Applied Field
            maxApliedField = np.amax(appliedFieldm[:,i]) # DP comment: this returns the maximum value of a vector containing the applied field at position i for all time steps
            if maxApliedField==0: # DP comment: for the PM1 there will be many positions with maxApliedField == 0 because they are out of the range of the magnet. For FAME cooler, maxApliedField is never zero
                MFM[i] = 0 # DP comment: MFM -> Magnetic Field Modifier, which was previously set to a matrix of ones
            else:
                # Maximum Magnetization at the maximum field
                if   species_descriptor[i]== 'reg-si1': mag_c = si1.mMag_c; mag_h = si1.mMag_h # DP comment: mMag_c and mMag_h interpolating functions are renamed
                elif species_descriptor[i]== 'reg-si2': mag_c = si2.mMag_c; mag_h = si2.mMag_h
                elif species_descriptor[i]== 'reg-si3': mag_c = si3.mMag_c; mag_h = si3.mMag_h
                elif species_descriptor[i]== 'reg-si4': mag_c = si4.mMag_c; mag_h = si4.mMag_h
                elif species_descriptor[i]== 'reg-si5': mag_c = si5.mMag_c; mag_h = si5.mMag_h
                elif species_descriptor[i]== 'reg-Gd':  mag_c = Gd.mMag_c;  mag_h = Gd.mMag_h
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

                # iteration of the cycle loop

        ###################### DP: The "for loop" to run over the time steps of a cycle starts here ###################

        for n in range(1, nt+1):  # 1->nt
            # Run every timestep
            # Initial
            stepTol = 0
            stepCount = 1
            # ch_factor[i]=0 coolingcurve selected
            # ch_factor[i]=1 heatingcurve seelected
            # Initial guess of the current step values.
            iynext  = np.copy(y[n-1, :]) # DP: this is a vector containing the guess fluid temperature distribution along the flowing path for the current time step
            # DP: as initial guess for the current time step, it is assumed that the temperature distribution is equal to the final temp distribution in the previous time step
            isnext  = np.copy(s[n-1, :])
            # current and previous temperature in [K]
            pfT     = y[n-1, :]  * (Thot - Tcold) + Tcold # DP comment: Previous time step fluid temperature in [K]
            psT     = s[n-1, :]  * (Thot - Tcold) + Tcold # DP comment: Previous time step solid temperature in [K]

            if max(pfT)>maxTemp: # DP comment: given that maxTemp is Tcold, I assume max(pfT), which must be close to Thot, is greater than maxTemp
                maxTemp = max(pfT)
            if min(pfT)<minTemp:
                minTemp = min(pfT)

            # DP: Properties of fluid and solid initialized to zero for every time step.
            # DP: From the name of the variables, it seems to be related to the previous time step
            cpf_prev  = np.zeros(N+1) # DP: heat capacity of fluid along the fluid path
            rhof_prev = np.zeros(N+1) # DP: density of fluid along the fluid path
            muf_prev  = np.zeros(N+1) # DP: viscosity of fluid along the fluid path
            kf_prev   = np.zeros(N+1) # DP: thermal conductivity of fluid along the fluid path
            cps_prev  = np.zeros(N+1) # DP: heat capacity of solid along the fluid path. This includes several types of solids such as glass spheres and MCM
            Ss_prev   = np.zeros(N+1) # ??
            S_c_past  = np.zeros(N+1) # DP: entropy of solid for a cooling protocol
            S_h_past  = np.zeros(N+1) # DP: entropy of solid for a heating protocol
            Sirr_prev = np.zeros(N+1) # DP: entropy of solid irreversible part?
            Sprev     = np.zeros(N+1) # DP: Anhysteretic entropy of the MCM for the previous time step
            prevHintNew  = np.zeros(N+1) # ??


            for i in range(N+1): # DP: this goes from i=0 to i=N
                if species_descriptor[i].startswith("reg"):
                    # Internal field
                    prevHint        = appliedFieldm[n-1,i]*MFM[i]
                    prevHintNew[i]  = appliedFieldm[n-1,i]*MFM[i]
                    # DP: loading heat capacity and entropy data of the MCM
                    if   species_descriptor[i]== 'reg-si1': cp_c = si1.mCp_c; cp_h = si1.mCp_h; ms_c = si1.mS_c; ms_h = si1.mS_h
                    elif species_descriptor[i]== 'reg-si2': cp_c = si2.mCp_c; cp_h = si2.mCp_h; ms_c = si2.mS_c; ms_h = si2.mS_h
                    elif species_descriptor[i]== 'reg-si3': cp_c = si3.mCp_c; cp_h = si3.mCp_h; ms_c = si3.mS_c; ms_h = si3.mS_h
                    elif species_descriptor[i]== 'reg-si4': cp_c = si4.mCp_c; cp_h = si4.mCp_h; ms_c = si4.mS_c; ms_h = si4.mS_h
                    elif species_descriptor[i]== 'reg-si5': cp_c = si5.mCp_c; cp_h = si5.mCp_h; ms_c = si5.mS_c; ms_h = si5.mS_h
                    elif species_descriptor[i]== 'reg-Gd':  cp_c = Gd.mCp_c;  cp_h = Gd.mCp_h;  ms_c = Gd.mS_c;  ms_h = Gd.mS_h
                    # Previous specific heat
                    Tr=psT[i]
                    dT=.5 # DP: this could be any small value given that it is just for calculating the derivative
                    dsdT = (ms_c(Tr+dT, prevHint)[0, 0]*(.5)  +  ms_h(Tr+dT, prevHint)[0, 0]*(.5)) - (ms_c(Tr-dT, prevHint)[0, 0]*(.5)  +  ms_h(Tr-dT, prevHint)[0, 0]*(.5))
                    cps_prev[i]  = psT[i]*(np.abs(dsdT)/(dT*2)) # DP: why not calculating the Cp from the available data?
                    # DP: 2 in the denominator obeys to the fact that the derivative is taken as [f(x+dx)-f(x-dx)]/(2*dx) instead of [f(x+dx)-f(x)]/(dx)
                    # Entropy position of the previous value
                    S_c_past[i]   = ms_c(Tr, prevHint)[0, 0]
                    S_h_past[i]   = ms_h(Tr, prevHint)[0, 0]
                    Sirr_prev[i]  = S_c_past[i] *(1-ch_factor[i])  -  S_h_past[i] *ch_factor[i] # DP: this corresponds to the irreversible part. It is not useful
                    Sprev[i]      = S_c_past[i] *(1-ch_factor[i])  +  S_h_past[i] *ch_factor[i] # DP: this is the anhysteretic entropy
                    # old code
                    Ss_prev[i]     = ms_c(psT[i], prevHint)[0, 0]*(1-ch_factor[i]) + ms_h(psT[i], prevHint)[0, 0]*ch_factor[i] # DP: this is equivalent to Sprev[i]
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
                elif species_descriptor[i]== 'gs':
                    cps_prev[i]    = gsCp
                    Ss_prev[i]     = 0
                    # This is where the gs stuff will go
                elif species_descriptor[i]== 'ls':
                    cps_prev[i]    = lsCp
                    Ss_prev[i]     = 0
                    # This is where the gs stuff will go
                else:
                    cps_prev[i]    = 0
                    Ss_prev[i]     = 0
                    # This is where the void stuff will go
                # liquid calculations
                # Calculate Specific heat
                cpf_prev[i]    = fCp(pfT[i],percGly)
                # Calculate Density
                rhof_prev[i]   = fRho(pfT[i],percGly)
                # Calculate Dynamic Viscosity
                muf_prev[i]    = fMu(pfT[i],percGly)
                # Calculate Conduction
                kf_prev[i]     = fK(pfT[i],percGly)
            ##################### DP: here is where the iteration at every time step begins ######################
            # Loop until stepTol is found or maxSteps is hit.
            while ( not stepTol and stepCount <= maxSteps):

                # iynext is the guess n Fluid
                # isnext is the guess n Solid
                # y[n-1,:] is the n-1 Fluid solution
                # s[n-1,:] is the n-1 Solid solution
                ################################################################
                # Grab Current State properties
                fT = iynext * (Thot - Tcold) + Tcold # DP: this is a vector containing the guess fluid temperatures for all positions along the flow path for the current time step
                sT = isnext * (Thot - Tcold) + Tcold # DP: this is a vector containing the guess solid temperatures for all positions along the flow path for the current time step

                # DP: The following variables are used in the construction of the system of algebraic equations, the solid and fluid tridiagonal matrices

                Cs = np.zeros(N + 1)
                ks = np.zeros(N + 1)
                Smce = np.zeros(N + 1)
                k = np.zeros(N + 1)
                Omegaf = np.zeros(N + 1)
                Spres = np.zeros(N + 1)
                Lf = np.zeros(N + 1)

                Cf = np.zeros(N + 1)
                rhof_cf_ave = np.zeros(N + 1)
                rhos_cs_ave = np.zeros(N + 1)
                Ff = np.zeros(N + 1)
                Sp = np.zeros(N + 1)

                # Weighted guess value
                ww = 0.5
                # Int the pressure
                pt[n] = 0
                dP = 0
                for i in range(N + 1):

                    # DP: properties are calculated at the temperatures of the previous and current (assumed, which changes at every iteration) time steps
                    # and the average is taken. I said average because the weighting value is 0.5. There are actually three options here: one is to calculate
                    # the properties at the temperature of the current time step, the second is to calculate the properties at the temperature of the previous
                    # time step, and the third is to take the average of both as it was chosen here.

                    # Calculate Specific heat fluid
                    cpf_ave  = fCp(fT[i], percGly) * ww + cpf_prev[i] * (1 - ww)
                    # Calculate Density fluid
                    rhof_ave = fRho(fT[i], percGly) * ww + rhof_prev[i] * (1 - ww)
                    # Calculate Dynamic Viscosity fluid
                    muf_ave  = fMu(fT[i], percGly) * ww + muf_prev[i] * (1 - ww)
                    # Calculate Conduction fluid
                    kf_ave   = fK(fT[i], percGly) * ww + kf_prev[i] * (1 - ww)
                    # Combined rhof cf
                    rhof_cf_ave[i] = cpf_ave * rhof_ave
                    if species_descriptor[i].startswith("reg"):
                        if   species_descriptor[i]== 'reg-si1': cp_c = si1.mCp_c; cp_h = si1.mCp_h; ms_c = si1.mS_c; ms_h = si1.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.55;
                        elif species_descriptor[i]== 'reg-si2': cp_c = si2.mCp_c; cp_h = si2.mCp_h; ms_c = si2.mS_c; ms_h = si2.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.77;
                        elif species_descriptor[i]== 'reg-si3': cp_c = si3.mCp_c; cp_h = si3.mCp_h; ms_c = si3.mS_c; ms_h = si3.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.73;
                        elif species_descriptor[i]== 'reg-si4': cp_c = si4.mCp_c; cp_h = si4.mCp_h; ms_c = si4.mS_c; ms_h = si4.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.75;
                        elif species_descriptor[i]== 'reg-si5': cp_c = si5.mCp_c; cp_h = si5.mCp_h; ms_c = si5.mS_c; ms_h = si5.mS_h; T_h=si1.mTemp_h; T_c = si1.mTemp_c; Reduct     = 0.72;
                        elif species_descriptor[i]== 'reg-Gd':  cp_c = Gd.mCp_c;  cp_h = Gd.mCp_h;  ms_c = Gd.mS_c;  ms_h = Gd.mS_h;  T_h=Gd.mTemp_h;  T_c = Gd.mTemp_c;  Reduct     = 1;
                        # Field
                        Hint = appliedFieldm[n, i]*MFM[i]
                        # rho*cs fluid
                        dT=1
                        Tr         = psT[i] # TODO: should not this be sT[i]?
                        aveField = (Hint+prevHintNew[i])/2 # DP: average between the internal field of the previous and current time steps
                        dsdT = (ms_c(sT[i]+dT, aveField)[0, 0]*(.5)  +  ms_h(sT[i]+dT, aveField)[0, 0]*(.5)) - (ms_c(sT[i]-dT, aveField)[0, 0]*(.5)  +  ms_h(sT[i]-dT, aveField)[0, 0]*(.5))
                        # TODO: not clear why aveField is used instead of Hint (current time step). Conversely, temperature is for the current time step.
                        cps_curr   = Tr*(np.abs(dsdT)/(dT*2)) # DP: it is not clear why it uses the temperature of the previous time step instead of that of the current time step
                        cps_ave = cps_curr  * ww + cps_prev[i] * (1 - ww) # DP: this is equation B.9 of Theo's thesis
                        rhos_cs_ave[i] = cps_ave * mRho
                        # Smce DP: it seems that the MCE is calculated as an isothermal entropy change because the entropy of current and previous steps
                        # are calculated at the temperature of the previous step.
                        # The anhysteretic entropy of the current time step is calculated at the magnetic field of the current time step

                        S_c_curr   = ms_c(Tr, Hint)[0, 0] # DP: for the calculation of the MCE is clear that the initial temperature must be that of the previous time step
                        S_h_curr   = ms_h(Tr, Hint)[0, 0] # DP: there are 3 optns: to use T° of the current time step (assumed), T° of previous time step, or a combination
                        Sirr_cur   = S_c_curr *(1-ch_factor[i])  -  S_h_curr *ch_factor[i]
                        Scur       = S_c_curr *(1-ch_factor[i])  +  S_h_curr *ch_factor[i] # DP: this is the anhysteretic entropy calculated from cooling high field and heating high field entropy curves
                        #Mod        = 0.5*(Sirr_cur+Sirr_prev[i])*np.abs((2*dT)/dsdT)
                        Smce[i]    = (Reduct*A_c[i] * (1 - e_r[i]) * mRho * Tr * (Sprev[i]-Scur))/ (DT* (Thot - Tcold))
                        # TODO: eq. B.20 of Theo's thesis states that the entropy difference should be Scur-Sprev. So, the question is: is there an inconsistency here?
                        # DP: the properties of the fluid in the following functions were calculated above as the average of the properties at the
                        # temperatures of current and previous time steps
                        # Effective Conduction for fluid
                        k[i] = kDyn_P(Dsp, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(V[n] / (A_c[i])))
                        # Forced convection term east of the P node
                        Omegaf[i] = A_c[i] * beHeff_E(Dsp, np.abs(V[n] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave, freq, cps_ave, mK, mRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        Spres[i], dP = SPresM(Dsp, np.abs(V[n] / (A_c[i])), np.abs(V[n]), e_r[i], muf_ave, rhof_ave,A_c[i] * e_r[i])
                        dP = dP * 2.7 # DP: from the paper, this factor is to compensate for the additional pressure drop occurring in beds of irregular
                        # shaped particles given that Ergun's correlation is for beds of spherical particles
                        Spres[i] = Spres[i]*2.7
                        # Loss term
                        if ConfName == "R7":
                            Lf[i] = P_c[i] * FAME_ThermalResistance(Dsp, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave, kg10, casing_th, air_th)
                            #Lf[i] = P_c[i] * FAME_ThermalResistance2(Dsp, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave, kg10, casing_th, freq)
                        else:
                            Lf[i] = P_c[i] * ThermalResistance(Dspls, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave, kg10, r1, r2, r3)
                        # TODO: this is not necessary for the FAME cooler. It can be deleted.


                        # Effective Conduction for solid
                        ks[i] = kStat(e_r[i], kf_ave, mK)
                        ### Capacitance solid
                        Cs[i] = rhos_cs_ave[i] * A_c[i] * (1 - e_r[i]) * freq  # DP: freq is used here because in the..
                        # SolveSolid function Cs is divided by dt = 1/(nt+1). So, this way dt becomes DT
                    elif species_descriptor[i] == 'gs':
                        # This is where the gs stuff will go
                        # Effective Conduction for solid
                        rhos_cs_ave[i] = gsCp * gsRho
                        # Effective Conduction for fluid
                        k[i] = kDyn_P(Dspgs, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(V[n] / (A_c[i])))
                        # Forced convection term east of the P node
                        # TODO: heat transfer coefficient is based on crushed particles rather than spherical
                        Omegaf[i] = A_c[i] * beHeff_I(Dspgs, np.abs(V[n] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave,
                                                      freq, gsCp, gsK, gsRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        Spres[i], dP = SPresM(Dspgs, np.abs(V[n] / (A_c[i])), np.abs(V[n]), e_r[i], muf_ave, rhof_ave,
                                              A_c[i] * e_r[i])
                        # Loss term
                        Lf[i] = P_c[i] * ThermalResistance(Dspgs, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave,
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
                        k[i] = kDyn_P(Dspls, e_r[i], cpf_ave, kf_ave, rhof_ave, np.abs(V[n] / (A_c[i])))
                        # Forced convection term east of the P node
                        Omegaf[i] = A_c[i] * beHeff_I(Dspls, np.abs(V[n] / (A_c[i])), cpf_ave, kf_ave, muf_ave, rhof_ave,
                                                      freq, lsCp, lsK, lsRho, e_r[i])  # Beta Times Heff
                        # Pressure drop
                        Spres[i], dP = SPresM(Dspls, np.abs(V[n] / (A_c[i])), np.abs(V[n]), e_r[i], muf_ave, rhof_ave,
                                              A_c[i] * e_r[i])
                        # Loss term
                        Lf[i] = P_c[i] * ThermalResistance(Dspls, np.abs(V[n] / (A_c[i])), muf_ave, rhof_ave, kair, kf_ave,
                                                       kg10, r1, r2, r3)
                        # Effective Conduction for solid
                        ks[i] = kStat(e_r[i], kf_ave, lsK)
                        #Smce
                        Smce[i] = 0
                        ### Capacitance solid
                        Cs[i] = rhos_cs_ave[i] * A_c[i] * (1 - e_r[i]) * freq
                    else:
                        k[i] = kf_ave
                        if species_descriptor[i] == 'void' and ConfName != "R7":
                            Lf[i] = P_c[i] * ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs, r1, r2, r3)
                        elif species_descriptor[i] == 'void' and ConfName == "R7":
                            Lf[i] = P_c[i] * FAME_ThermalResistanceVoid(kair, kf_ave, kg10, A_c[i], P_c[i], casing_th, air_th)
                        elif species_descriptor[i] == 'void1':
                            Lf[i] = P_c[i] * ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs1, r1, r2, r3)
                        elif species_descriptor[i] == 'void2':
                            Lf[i] = P_c[i] * ThermalResistanceVoid(kair, kf_ave, kg10, kult, rvs2, r1, r2, r3)
                        # No solid in the void
                        ks[i] = 0
                        # No interaction between solid and fluid since there is no solid.
                        Omegaf[i] = 0 #
                        # This will just make the plots nicer by having the temperature of the solid be the fluid temperature.
                        Cs[i] = 1
                        Smce[i] = (iynext[i]-s[n-1,i])/DT
                        #neglect pressure term.
                        Spres[i]= 0
                        dP = 0
                        # This is where the void stuff will go
                    pt[n] = dP * DX + pt[n] # DP: it seems that the term dP is actually dP/dx
                    # DP: pt[n] returns the pressure drop along the regenerator at the current time step

                # DP: In the original file, the following variables were defined before the for loop that ended a few lines above
                Kfw = np.zeros(N - 1)
                Kfe = np.zeros(N - 1)
                Ksw = np.zeros(N - 1)
                Kse = np.zeros(N - 1)

                ### Capacitance fluid
                Cf = rhof_cf_ave * A_c * e_r * freq # DP: in Numpy the A*B is an element wise multiplication, which means that both matrices must be of the same size.
                # DP: Matrix multiplication in the row-by-column way is performed using np.matmul(A,B) and A is lxm and B is mxn so that A*B is lxn

                ### Fluid term
                Ff = (rhof_cf_ave * V[n]) / L_tot # DP: this is divided by L_tot because in the FluidSolver function Ff
                # is divided by dx = 1/(N+1) instead of DX. So, it is necessary to include L_tot so that dx*L_tot = DX
                Sp = Spres / (Thot - Tcold)

                for i in range(N - 1): # DP: this runs from 0 to N-2, ghost nodes are excluded, aligns with 1->N-1
                    # Fluid Conduction term west of the P node
                    Kfw[i] = ((1 - fr[i]) / (A_c[i] * e_r[i] * k[i])
                              + (fr[i]) / (A_c[i+1] * e_r[i+1] * k[i+1])) ** -1
                    # DP: the first element in the vector Kfw includes info about the ghost node 0 and the last term
                    # includes info of the last node of the regenerator and the one to its left. The ghost node N is excluded
                    # Fluid Conduction term east of the P node
                    Kfe[i] = ((1 - fr[i+1]) / (A_c[i+1] * e_r[i+1] * k[i+1])
                              + (fr[i+1]) / (A_c[i+2] * e_r[i+2] * k[i+2])) ** -1
                    # DP: the first element of vector Kfe (index 0) includes info of node 1 and node 2 (spatial domain)
                    # and the last element of vector Kfe (index N-2) includes info of node N-1, last node of the regenerator
                    # and the node to its right, ghost node N.
                    # Solid Conduction term
                    if ks[i]==0 or ks[i+1]==0: # DP: when the node corresponds to a void space
                        Ksw[i] =0
                    else:
                        # Conduction term west of the P node
                        Ksw[i] = ((1 - fr[i]) / (A_c[i] * (1-e_r[i]) * ks[i])
                                + (fr[i]) / (A_c[i+1] * (1-e_r[i+1]) * ks[i+1])) ** -1
                    if ks[i+1]==0 or ks[i+2]==0:
                        Kse[i] =0
                    else:
                        # Conduction term east of the P node
                        Kse[i] = ((1 - fr[i+1]) / (A_c[i+1] * (1-e_r[i+1]) * ks[i+1])
                                + (fr[i+1]) / (A_c[i+2] * (1-e_r[i+2]) * ks[i+2])) ** -1
                # TODO: an apparent error in the solid conduction terms was corrected because e_r was used instead of (1-e_r)
                # TODO: check if fr is used correctly
                Omegas = np.copy(Omegaf) # DP: this is the coefficient of the convection term, which is equal for both fluid and solid
                Kfw = Kfw /(DX*L_tot) # DP: L_tot is included here because in SolveFluid function Kfw is multiplied by dx = 1/(N+1)
                Kfe = Kfe /(DX*L_tot)
                Ksw = Ksw /(DX*L_tot)
                Kse = Kse /(DX*L_tot)


                ################################################################
                ####################### SOLVE FLUID EQ      ####################
                # Fluid Equation
                ynext = SolveFluid(iynext, isnext, y[n-1,:], s[n-1,:],V[n],Cf,Kfe,Kfw,Ff,Omegaf,Lf,Sp,CF,CL,CVD,N,dx,dt,yamb)
                ################################################################
                ####################### SOLVE SOLID EQ      ####################
                # Solid Equation
                snext = SolveSolid(ynext, isnext, y[n-1,:], s[n-1,:],V[n],Cs,Kse,Ksw,Omegas,Smce,CS,CMCE,N,dx,dt)
                ################################################################
                ####################### CHECK TOLLERANCE    ####################
                # Check the tolerance of the current time step
                stepTol = AbsTolFunc(ynext,iynext,maxStepTol[stepTolInt])[0] and AbsTolFunc(snext,isnext,maxStepTol[stepTolInt])[0] # DP: only get the boolean part of the function,
                # which indicates whether or not the absolute difference between the assumed and calculated temperature distributions is less than the predefined tolerance.
                # The second element that returns from the function AbsTolFunc is not used or not relevant here
                ################################################################
                ####################### DO housekeeping     ####################
                # Add a new step to the step count
                stepCount = stepCount + 1
                # Check if we have hit the max steps
                if (stepCount == maxSteps):
                    print("Hit max step count")
                    print(AbsTolFunc(ynext,iynext,maxStepTol[stepTolInt]))
                    print(AbsTolFunc(snext,isnext,maxStepTol[stepTolInt]))
                    print(stepTol)
                # Copy current values to new guess and current step.
                s[n, :] = np.copy(snext)  # DP: updated the current time step solid Temp distrib in the matrix containing all time steps temperature distributions
                isnext  = np.copy(snext)  # DP: uses the solid temperature just calculated for new time step iteration
                y[n, :] = np.copy(ynext)
                iynext  = np.copy(ynext)
                if (np.any(np.isnan(y)) or np.any(np.isnan(s))):
                    # Sometimes the simulation hits a nan value. We can redo this point later.
                    print(y)
                    print(s)
                    break
                # Break the step calculation
                if ((time.time()-t0)/60)>time_lim: # DP: time.time() returns the number of seconds from January 1st 1970 00:00:00.
                    # So, the difference time.time()-t0 is in seconds, and dividing it by 60 returns a difference in minutes
                    break # DP: this breaks the while loop for every time step
                ################################################################
                ################### ELSE ITERATE AGAIN - WHILE LOOP FOR EVERY TIME STEP FINISHES HERE ###################
            # break the cycle calculation
            if ((time.time()-t0)/60)>time_lim:
                break # DP: This breaks the for loop over the time steps
        # Check if current cycle is close to previous cycle.
        [bool_y_check,max_val_y_diff]=AbsTolFunc2d(y,iyCycle,maxCycleTol)
        [bool_s_check,max_val_s_diff]=AbsTolFunc2d(s,isCycle,maxCycleTol)
        cycleTol = bool_y_check and bool_s_check # DP comment: this will return True if both arguments are True, otherwise it returns False, which will keep the cycle while loop running
        # DP: change the time step tolerance
        if (max_val_y_diff/10)<maxStepTol[stepTolInt]:
            stepTolInt = stepTolInt + 1
            if stepTolInt == len(maxStepTol): # DP: len([3, 6, 1, 4, 9]) returns 5, the number of elements in the list
                stepTolInt=len(maxStepTol)-1

        # DP: it is useful to see on screen some results during the iterative calculation process
        if cycleCount % 10 == 1: # DP: this is true for cycleCount = 11 or 21 or 31 and so on. The operator % returns the modulus of the division
            coolingpowersum=0
            heatingpowersum=0
            startint=0
            for n in range(startint, nt):
                # DP: this for loop is for the numerical integration of the equation 3.33 of Theo's thesis (which misses the integrand, dt).
                tF = y[n, 0] * (Thot - Tcold) + Tcold # DP: temperature in [K] at the cold end at time step n
                tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold # DP: temperature in [K] at the cold end at time step n+1
                coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * ((tF+tF1)/2 - Tcold)
                coolingpowersum = coolingpowersum + coolPn

                # DP: Heating power
                tF_h = y[n,-1]*(Thot - Tcold) + Tcold
                tF1_h = y[n+1,-1]*(Thot - Tcold) + Tcold
                heatPn = freq*fCp((tF_h+tF1_h)/2,percGly)*fRho((tF_h+tF1_h)/2,percGly)*V[n]*DT*((tF_h+tF1_h)/2-Thot)
                heatingpowersum = heatingpowersum + heatPn

            qc = coolingpowersum # DP: 2 changed by 7 to account for the number of regenerators of the device
            qh = heatingpowersum
            print("Case num {0:d} CycleCount {1:d} Cooling Power {2:2.5e} Heating Power {3:2.5e} y-tol {4:2.5e} s-tol {5:2.5e} run time {6:4.1f} [min]".format(int(caseNum),cycleCount,qc,qh,max_val_y_diff,max_val_s_diff,(time.time()-t0)/60 ))

        if ((time.time()-t0)/60)>time_lim: # DP: if the for loop was broken above, then do...
            coolingpowersum=0
            heatingpowersum=0
            startint=0
            for n in range(startint, nt):
                tF = y[n, 0] * (Thot - Tcold) + Tcold
                tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold
                coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * ((tF+tF1)/2 - Tcold)
                coolingpowersum = coolingpowersum + coolPn

                # DP: Heating power
                tF_h = y[n,-1]*(Thot - Tcold) + Tcold
                tF1_h = y[n+1,-1]*(Thot - Tcold) + Tcold
                heatPn = freq*fCp((tF_h+tF1_h)/2,percGly)*fRho((tF_h+tF1_h)/2,percGly)*V[n]*DT*((tF_h+tF1_h)/2-Thot)
                heatingpowersum = heatingpowersum + heatPn

            qc = coolingpowersum # DP: changed from 2 to 7. Number of regenerators
            qh = heatingpowersum # Added by DP
            print("Case num {0:d} CycleCount {1:d} Cooling Power {2:2.5e} Heating Power {6:2.5e} y-tol {3:2.5e} s-tol {4:2.5e} run time {5:4.1f} [min]".format(int(caseNum),cycleCount,qc,max_val_y_diff,max_val_s_diff,(time.time()-t0)/60,qh ))
            # Pickle data
            aaa = (y, s, stepTolInt, iyCycle, isCycle)
            # open the file for writing
            fileObject = open(PickleFileName,'wb')
            # this writes the object a to the
            # file named 'testfile'
            pickle.dump(aaa,fileObject)
            # here we close the fileObject
            fileObject.close()
            print("saving pickle file")
            # Quit Program
            sys.exit()
        # Copy last value to the first of the next cycle.
        s[0, :] = np.copy(s[-1, :])
        y[0, :] = np.copy(y[-1, :])
        # Add Cycle
        cycleCount = cycleCount + 1
        # Did we hit the maximum number of cycles
        if (cycleCount == maxCycles):
            print("Hit max cycle count\n")
        # Copy current cycle to the stored value
        isCycle = np.copy(s)
        iyCycle = np.copy(y)
        if np.any(np.isnan(y)):
            # Sometimes the simulation hits a nan value. We can redo this point later.
            print(y)
            break # DP: this breaks the while loop for the cycle calculation
        # End Cycle
    t1 = time.time()
    ########################## END THE LOOP #########################

    # DP: the iterative calculation process on the cycle level ends up here

    quart=int(nt/4)
    halft=int(nt/2)
    tquat=int(nt*3/4)
    # This is a modifier so we can modify the boundary at which we calculate the
    # effectiveness
    cold_end_node  = np.min(np.argwhere([val.startswith("reg") for val in species_descriptor])) # 0
    # DP: the following argument: [val.startswith("reg") for val in species_descriptor] returns an array of False and True values after evaluating the condition
    # startwith("reg"). This array is of the same length as species_descriptor
    # DP: np.argwhere returns an array with the positions of the True values obtained from [val.startswith("reg") for val in species_descriptor].
    # DP: Finally, np.min gives the position of the first node at which the word reg is found, i.e. where the regenerator starts along the flow path
    hot_end_node   = np.max(np.argwhere([val.startswith("reg") for val in species_descriptor])) # -1
    eff_HB_CE = np.trapz((1-y[halft:,  cold_end_node]),x=t[halft:]) /(tau_c/2) # DP: this is in agreement with equation 8-373 of the book of Nellis and Klein
    # DP: Effectiveness over the hot blow, which apparently occurs during the second half of the period
    # DP: this function integrates the first element over the domain given by the second element.
    eff_CB_HE = np.trapz(y[:halft+1,  hot_end_node],x=t[:halft+1])/ (tau_c/2) # DP: Effectiveness over the cold blow
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
    yMaxCBlow  = y[quart,  :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the fluid in [K] at the instant tau/4
    yMaxHBlow = y[tquat, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the fluid in [K] at the instant tau*3/4

    yEndBlow  = y[-1,  :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the fluid in [K] at the end of the cycle
    yHalfBlow = y[halft, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the fluid in [K] at half time the cycle period

    sMaxCBlow  = s[quart,  :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the solid in [K] at the instant tau/4
    sMaxHBlow = s[tquat, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the solid in [K] at the instant tau*3/4

    sEndBlow  = s[-1,  :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the solid in [K] at the end of the cycle
    sHalfBlow = s[halft, :] * (Thot - Tcold) + Tcold # DP: temperature distribution of the solid in [K] at half time the cycle period

    coolingpowersum=0
    startint=0
    # DP: this is the numerical integration of freq*integral(m*Cp*(Tf,cold_end-Tcold)*dt) from 0 to tau, equation 3.33 of Theo's thesis.
    # It seems that it is performed following a rectangle rule. https://en.wikipedia.org/wiki/Numerical_integration
    for n in range(startint, nt):
        tF = y[n, 0] * (Thot - Tcold) + Tcold
        tF1 = y[n+1, 0] * (Thot - Tcold) + Tcold
        coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * ((tF+tF1)/2 - Tcold)
        coolingpowersum = coolingpowersum + coolPn
    qc = coolingpowersum # DP: cooling power of the device

    coolingpowersum=0 # DP: to avoid confusions, this variable should have been named heatingpowersum
    startint=0
    for n in range(startint, nt):
        tF = y[n, -1] * (Thot - Tcold) + Tcold
        tF1 = y[n+1, -1] * (Thot - Tcold) + Tcold
        coolPn =  freq * fCp((tF+tF1)/2,percGly) * fRho((tF+tF1)/2,percGly) * V[n] * DT * (Thot-(tF+tF1)/2)
        coolingpowersum = coolingpowersum + coolPn
    qh = coolingpowersum # DP: heating power of the device
    # DP: cooling power of FAME device is 7 times the cooling power of one regenerator.
    # Demonstrated in the file Cooling_capacity_calc.py

    Kamb = 2.5 # DP: This has to be measured experimentally for the
    qccor = 7*qc - Kamb * (Tambset-Tcold)  # DP: this corresponds to the net power output, equation 3.34 of Theo's thesis.
    # It includes a correction to account for additional heat leaks in the CHEX
    pave = np.trapz(pt[halft:], x=t[halft:]) / (tau_c/2)  # DP: average pressure drop across the regenerator
    print("{0:3.1f} {1:3.1f} {2:1.2f} {3:1.2f} Cycle Count: {4:d} Tol-y: {5:1.4e} Tol-s {6:1.4e}".format(float(Thot), float(Tcold), float(Uti), float(freq), int(cycleCount), float(max_val_y_diff), float(max_val_s_diff)))
    print('Utilization: {0:1.3f} Frequency: {1:1.2f}'.format(Uti, freq))
    print("Run time: {0:3.2f} [min]".format((t1 - t0) / 60))
    print("Hot Side: {} Cold Side: {}".format(Thot, Tcold))
    print('Effectiveness HB-CE {} CB-HE {}'.format(eff_HB_CE, eff_CB_HE))
    print('Cooling power {}'.format(qc))
    print('Corrected Cooling Power {}'.format(qccor))
    print('Pressure drop {} (kPa)'.format(pave/1000))

    print('Values found at minimal field')
    print('min Applied Field {}'.format(minAplField))
    print('min Internal Field {}'.format(minPrevHint))
    print('min Magnetic Temperature {}'.format(minMagTemp))
    print('min Cp: {}'.format(minCpPrev))
    print('min SS:{}'.format(minSSprev))
    print('Lowest Temperature found in the SS cycle: {}'.format(minTemp))


    print('Values found at maximum field')
    print('max Applied Field {}'.format(maxAplField))
    print('max Internal Field {}'.format(maxPrevHint))
    print('max Magnetic Temperature {}'.format(maxMagTemp)) # DP: this probably refers to Max MCM Temperature
    print('max Cp: {}'.format(maxCpPrev))
    print('max SS:{}'.format(maxSSprev))
    print('highest Temperature found in the SS cycle: {}'.format(maxTemp))

    # Remove Pickle
    try:
        os.remove(PickleFileName)
        print("We removed the pickle file")
    except FileNotFoundError:
        print('Hey! It was done very fast.')

    return Thot, Tcold, qc, qccor, (t1-t0)/60, pave, eff_HB_CE, eff_CB_HE, tFce, tFhe, yHalfBlow, yEndBlow, sHalfBlow, sEndBlow, y, s, pt, np.max(pt), Uti, freq, t, xloc, yMaxCBlow, yMaxHBlow, sMaxCBlow, sMaxHBlow, qh, cycleCount
 #          0     1   2    3        4       5
################# DP: the function "Run_Active" ends here ####################

if __name__ == '__main__':

    # Some useful functions for storing data.
    def FileSave(filename, content):
        with open(filename, "a") as myfile:
            myfile.write(content)

    def FileSaveMatrix(filename, content):
        with open(filename, "a") as f:
            for line in content:
                f.write(" ".join("{:9.6f}\t".format(x) for x in line))
                f.write("\n")

    def RunCaseThotTcold(case, jobName):  # DP: this is necessary for running arrays of tasks in the cluster
        numCases       = 4
        hotResolution  = 1
        coldResolution = 8

        maxcase = numCases * hotResolution * coldResolution
        Thotarr = np.linspace(273+22, 273+22, hotResolution)

        casenum=int(np.floor(case/(hotResolution*coldResolution))) # DP: I don't understand why making things complicated like this...

        if (casenum==0):
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 24
            cen_loc       = 0
            Tambset       = 298
            dispV         = 30.52e-6
            ff            = 1.7
            Dsp           = 600e-6
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 600
            cName         = "R7"
            time_limit    = 600  # [min] Time limit for the simulation in minutes
            cycle_toler   = 1e-3  # Maximum cycle tolerance: criterion for ending the iterative calculation process
            maxStepIter   = 300  # Maximum time step iterations the simulation is allowed to take
            maxCycleIter  = 300  # Maximum cycle iterations the simulation is allowed to take
        if (casenum==1):
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 24
            cen_loc       = 0
            Tambset       = 298
            dispV         = 30.52e-6
            ff            = 1.7
            Dsp           = 600e-6
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 600
            cName         = "R7"
            time_limit    = 700  # [min] Time limit for the simulation in minutes
            cycle_toler   = 1e-5  # Maximum cycle tolerance: criterion for ending the iterative calculation process
            maxStepIter   = 500  # Maximum time step iterations the simulation is allowed to take
            maxCycleIter  = 500  # Maximum cycle iterations the simulation is allowed to take
        if (casenum==2):
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 24
            cen_loc       = 0
            Tambset       = 298
            dispV         = 30.52e-6
            ff            = 1.7
            Dsp           = 600e-6
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 600
            cName         = "R7"
            time_limit    = 800  # [min] Time limit for the simulation in minutes
            cycle_toler   = 1e-6  # Maximum cycle tolerance: criterion for ending the iterative calculation process
            maxStepIter   = 700  # Maximum time step iterations the simulation is allowed to take
            maxCycleIter  = 700  # Maximum cycle iterations the simulation is allowed to take
        if (casenum==3):
            #RunTest("test_128_20_ALL.txt", 6.4e-6, 2, CF, CS, CL, CVD,CMCE, Thot, 35, num_processors, 200, 400, [0,20,40],300e-6)
            fileName      = "{}.txt".format(jobName)
            MaxTSpan      = 24
            cen_loc       = 0
            Tambset       = 298
            dispV         = 30.52e-6
            ff            = 1.7
            Dsp           = 600e-6
            CF            = 1
            CS            = 1
            CL            = 0
            CVD           = 1
            CMCE          = 1
            nodes         = 400
            timesteps     = 600
            time_limit    = 900  # [min] Time limit for the simulation in minutes
            cycle_toler   = 1e-7  # Maximum cycle tolerance: criterion for ending the iterative calculation process
            maxStepIter   = 900  # Maximum time step iterations the simulation is allowed to take
            maxCycleIter  = 900  # Maximum cycle iterations the simulation is allowed to take

        # DP: I don't get why this is defined in this way.
        Thot = Thotarr[int(np.floor(case/coldResolution)%hotResolution)]
        Tcold = Thot - MaxTSpan*(case%(coldResolution))/(coldResolution)-0.1

        print("iteration: {}/{} Case number: {} Thot: {} Tcold: {}".format(case, maxcase, casenum, Thot, Tcold))

        results = runActive(case,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,cName,jobName,time_limit,cycle_toler,maxStepIter,maxCycleIter)
        # Get result roots variable is broken down in:
        #  0     1    2   3      4        5
        # Thot,Tcold,qc,qccor,(t1-t0)/60,pave,
        #           6               7
        # integral_eff_HB_CE,integral_eff_CB_HE,
        #  8    9      10        11       12       13
        # tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,sEndBlow,
        # 14 15 16      17     18   19  20 21
        # y, s, pt, np.max(pt),Uti,freq,t,xloc
        # 22           23       24         25
        #yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow

        fileNameSave        = './Ends/' + str(case) + fileName
        FileSave(fileNameSave, "{},{},{},{},{},{} \n".format('Tspan [K]', 'Qc_corr [W]', 'Qc [W]', 'Cycles [-]', 'run time [min]', 'Max. Pressure drop [Pa]'))
        FileSave(fileNameSave, "{},{:4.2f},{:4.2f},{},{:4.2f},{:4.2f} \n".format(results[0]-results[1], results[3], results[2], results[27], results[4], results[17]))
        FileSave(fileNameSave, "Fluid temperatures \n")
        FileSaveMatrix(fileNameSave, results[14])
        FileSave(fileNameSave, "\n")
        FileSave(fileNameSave, "Solid temperatures \n")
        FileSaveMatrix(fileNameSave, results[15])
        FileSave(fileNameSave, "\n")


        # fileNameSave        = './' + fileName # DP: ./ is for specifying that the file is save to the working directory
        # fileNameEndTemp     = './Ends/{:3.0f}-{:3.0f}-PysicalEnd'.format(Thot,Tcold)+fileName
        # fileNameSliceTemp   = './Blow/{:3.0f}-{:3.0f}-BlowSlice'.format(Thot,Tcold)+fileName
        # FileSave(fileNameSave,"{},{},{},{},{},{},{} \n".format(results[0],results[1],results[2],results[3],results[4],results[5],results[26]) )
        # #FileSave(fileNameEndTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]','Uti [-]', 'freq [Hz]', 'run time [min]','Eff CE-HB [-]', 'Eff HE-CB [-]') )
        # #FileSave(fileNameEndTemp,"{},{},{},{},{} \n".format(results[0],results[1],results[18],results[19], results[4],results[6],results[7]) )
        # #EndTemperatures = np.stack((results[20], results[8],results[9]), axis=-1)
        # #FileSaveMatrix(fileNameEndTemp,EndTemperatures)
        # FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]','Uti [-]', 'freq [Hz]', 'run time [min]') )
        # FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format(results[0],results[1],results[18],results[19], results[4]) )
        # BlowSliceTemperatures = np.stack((results[21],results[10],results[11],results[12],results[13],results[22],results[23],results[24],results[25]), axis=-1)
        # FileSaveMatrix(fileNameSliceTemp,BlowSliceTemperatures)

    RunCaseThotTcold(float(sys.argv[1]),sys.argv[2])
    #RunCaseThotTcold(1)

    # ---------------------------------- Calculation of just one case --------------------------------------

    # #runActive(caseNum,Thot,Tcold,cen_loc,Tambset,dispV,ff,CF,CS,CL,CVD,CMCE,nodes,timesteps,Dsp,ConfName,jobName,time_limit,cycle_toler,maxStepIter,maxCycleIter)
    # #MaxTSpan      = 10
    # caseNumber    = 2
    # Thot          = 295
    # Tcold         = 292
    # cen_loc       = 0
    # Tambset       = 294
    # dispV         = 15.33e-6  # [m3/s] DP: device vol. flow rate = 1.84 L/min, 2 regenerators with simultaneous flow.
    # ff            = 1.2  # [Hz] DP: frequency of AMR cycle
    # CF            = 1
    # CS            = 1
    # CL            = 0
    # CVD           = 1
    # CMCE          = 1
    # nodes         = 400
    # timesteps     = 600
    # Dsp           = 425e-6
    # cName         = "R7"
    # jName         = "First_trial" # DP: It is better to use underline to connect words because this is used as file name
    # time_limit    = 600  # [min] Time limit for the simulation in minutes
    # cycle_toler   = 1e-4  # Maximum cycle tolerance: criterion for ending the iterative calculation process
    # maxStepIter   = 200  # Maximum time step iterations the simulation is allowed to take
    # maxCycleIter  = 300  # Maximum cycle iterations the simulation is allowed to take
    #
    # results = runActive(caseNumber, Thot, Tcold, cen_loc, Tambset, dispV, ff, CF, CS, CL, CVD, CMCE, nodes, timesteps, Dsp, cName, jName, time_limit,cycle_toler, maxStepIter, maxCycleIter)
    #
    # # Some useful functions for storing data.
    # def FileSave(filename, content):
    #     with open(filename, "a") as myfile:
    #         myfile.write(content)
    #
    # def FileSaveMatrix(filename, content):
    #     with open(filename, "a") as f:
    #         for line in content:
    #             f.write(" ".join("{:9.6f}\t".format(x) for x in line))
    #             f.write("\n")
    #
    # #  runActive():  return Thot,Tcold,qc,qccor,(t1-t0)/60,pave,eff_HB_CE,eff_CB_HE,tFce,tFhe,yHalfBlow,yEndBlow,sHalfBlow,sEndBlow,y, s, pt, np.max(pt),Uti,freq,t,xloc,yMaxCBlow,yMaxHBlow,sMaxCBlow,sMaxHBlow,qh
    # #                       0       1   2   3     4         5     6           7      8    9      10        11       12       13     14 15 16    17       18  19   20 21    22         23       24         25      26
    #
    # fileName = "Testing_functionality2.txt"
    # fileNameSave = './' + fileName
    # fileNameSliceTemp = './Blow/{:3.0f}-{:3.0f}-BlowSlice'.format(Thot, Tcold) + fileName
    # FileSave(fileNameSave,"{},{},{},{},{},{},{} \n".format(results[0], results[1], results[2], results[3], results[4], results[5],results[26]))
    # FileSave(fileNameSliceTemp,"{},{},{},{},{} \n".format('Thot [K]', 'Tcold [K]', 'Uti [-]', 'freq [Hz]', 'run time [min]'))
    # FileSave(fileNameSliceTemp,"{},{},{:4.2f},{},{:4.2f} \n".format(results[0], results[1], results[18], results[19], results[4]))
    # BlowSliceTemperatures = np.stack((results[21], results[10], results[11], results[12], results[13], results[22], results[23], results[24], results[25]), axis=-1)
    # FileSaveMatrix(fileNameSliceTemp, BlowSliceTemperatures)
    #
    # fluidtemperature = './' + "Fluid_Temperature2.txt"
    # fluidtemperatures = results[14]
    # FileSaveMatrix(fluidtemperature,fluidtemperatures)
    #
    # solidtemperature = './' + "Solid_Temperature2.txt"
    # solidtemperatures = results[15]
    # FileSaveMatrix(solidtemperature,solidtemperatures)