##### FAME cooler regenerator geometry #####

import numpy as np

W_reg = 0.045
H_reg = 0.013
Ac = H_reg * W_reg # [m^2] Cross sectional area of one regenerator
Pc = W_reg * 2 + H_reg * 2 # [m] Perimeter of the cross section of the regenerator
casing_th = 0.0035 # [m] Thickness of the casing material. G10 glass fiber reinforced epoxy in this case.
air_th = 0.001 # [m] Thickness of the air layer in between regenerator casing and magnets
L_reg1 = 0.060 # [m] length of one regenerator. New variable, it seems that it is not necessary
#species_discription = ['void', 'reg-si5', 'reg-si4', 'reg-si3', 'reg-si2', 'reg-si1', 'void'] # This is to identify the layers of MCM
#species_discription = ['void', 'reg-si3', 'reg-si2', 'reg-si1', 'void'] # This is to identify the layers of MCM
species_discription = ['void3', 'reg-Gd', 'void3']  # Case with some voids on the ends of the regenerator
#species_discription = ['reg-Gd'] # This is to identify the layers of MCM
# If the number of layers increases to 30, which is a realistic number, this list should have 30 elements to identify
# each layer.
#x_discription = [0, 0.006, 0.018, 0.030, 0.042, 0.054, 0.066, 0.072] # This needs to be defined prior to the simulation
#x_discription = [0, 0.006, 0.026, 0.046, 0.066, 0.072]
x_discription = [0, 0.006, 0.066, 0.072]
#x_discription = [0, 0.060]
# This needs to be defined prior to the simulation and adjusted everytime the geometry changes

# TODO. It could be interesting to see how the length of the layers influence the performance of the regenerator.
# One way to study this is by considering all layer of the same length and varying that length. A second perspective is
# by using layers of different length. This could have an impact in the shape of the temperature profiles
# in the regenerator.

# TODO. Write a test statement to check if the number of elements of x_discription equals the number of elements of species_discription plus one

#er      = 0.53 # [-] Porosity of regenerator
er      = 0.36 # [-] Porosity of regenerator. er=0.36 in Bowei's paper


percGly = 20 # [%] Percentage of glycol in the water glycol mixture used as HTF

# Radius information of the assembly

# r1      = 16e-3 / 2  # Inner Radius [m]
# r2      = 19.05e-3 / 2  # Outer Radius [m]
# r3      = 22e-3 / 2  # Bore Radius [m]
#
# Vvoid1  = 1.598e-6 # [m^3] 1 [cm^3]
# Vvoid2  = 1169.86e-9 # [m^3]
# Lvoid1  = 29.27e-3 #[m]
# Lvoid2  = 51.435e-3 #[m]
#
# L_add   = 18e-3
# Lvoid   = 14.6e-3 + L_add
# rv      = 1.9e-3  # Cold side void to the check valve radius [m]
# Vvoid   = 458.24e-9+rv**2*3.14*L_add # [m^3]
# rvs     = np.sqrt(Vvoid/(3.14*Lvoid))  # Cold side void radius [m]
# rvs1    = np.sqrt(Vvoid1/(3.14*Lvoid1))  # Hot  side void radius [m]
# rvs2    = np.sqrt(Vvoid2/(3.14*Lvoid2))  # Hot  side void radius [m]

################### WITH DV
# You can build any geometry based on glass spheres, regenerator(in this case Gd), and void space
#species_discription = ['void','gs','ls','void1','reg-si1','gs','void2']
#print("the geometry looks like: {}".format(species_discription))

mRho  = 7900.  # [kg/m^3]... DP: This was 6100 and is an assumed MCM density to calculate porosity of bed
mK    = 14  # [W/(m K)]...DP: this was 6 and is an assumed thermal conductivity of the MCM

#L_reg1 = 0.0225
#L_reg2 = 0.023
# Locations 0<- cold hex, hot hex -> end
# x_discription = [0,
#                     L_add+14.6e-3,
#                     L_add+76.1e-3+1.27e-3/2-Lvoid1/2-L_reg2,
#                     L_add+76.1e-3+1.27e-3/2-Lvoid1/2,
#                     L_add+76.1e-3+1.27e-3/2+Lvoid1/2,
#                     L_add+76.1e-3+1.27e-3/2+Lvoid1/2+L_reg1,
#                     L_add+76.1e-3+76.1e-3+1.27e-3-14.605e-3,
#                     L_add+76.1e-3+76.1e-3+1.27e-3-14.605e-3+51.435e-3] #[m]

# Distance between base of the cold heat exchanger to magnet center
#magOffset = L_add+ 76.1e-3+1.27e-3/2 # DP: this is used to create the matrix h_aplied[n,i], but it is not imported..?


# Surface Area
# Regenerator and glass spheres
#Ac      = np.pi * r1 ** 2
# Regenerator and glass spheres
#Pc      = 2 * np.pi * r1

# Set Material & Fluid Properties
# (BOROSILICATE) GLASS spheres
#gsCp  = 800.  # [J/(kg K)]
#gsRho = 2230.  # [kg/m^3]
#gsK   = 1.2  # [W/(m K)]
# http://www.scientificglass.co.uk/contents/en-uk/d115_Physical_Properties_of_Borosilicate_Glass.html
# http://www.schott.com/borofloat/english/attribute/thermic/index.html
# http://www.schott.com/d/tubing/9a0f5126-6e35-43bd-bf2a-349912caf9f2/schott-algae-brochure-borosilicate.pdf

# Lead spheres
#lsCp  = 830.  # [J/(kg K)]
#lsRho = 2230.  # [kg/m^3]
#lsK   = 1.005  # [W/(m K)]

kult  =  0.122   # [W/(m K)] Thermal conductivity Ultem
# https://www.plasticsintl.com/datasheets/ULTEM_GF30.pdf

kg10  =  0.608  # [W/(m K)] Thermal conductivity g10 material
# http://cryogenics.nist.gov/MPropsMAY/G-10%20CR%20Fiberglass%20Epoxy/G10CRFiberglassEpoxy_rev.htm

kair  = 0.0255  # [W/(m K)] air material


# Transport booklet

#er      = 0.53 # [-] Porosity of regenerator
#els     = 0.36  # [-] Porosity of lead spheres
#egs     = 0.43  # [-] Porosity of glass spheres
Dsp     = 600e-6  # [m] Diameter of MCM particles (spheres)
#Dspgs   = 0.003175  # [m] Diameter of glass spheres
#Dspls   = 300e-6  # [m] Diameter of lead spheres

Nd      = 0.36  # [-] Demagnetization coefficient

CL_set="Tamb" # Casing BC assumption DP: changed from grad to Tamb. It does make more sense to have a constant
# ambient temperature along the regenerator.

MOD_CL=0 # Do we wish to use experimental modication of the BC

ch_fac=0.5 # Averaging heating and cooling properties

print(CL_set)