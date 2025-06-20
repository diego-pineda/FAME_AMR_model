##### Regenerator geometry #####

import numpy as np

# ---- Overall shape: single regenerator rectangular cross section in the gap of a C-shaped rotational magnet

# Note: set to zero all parameters if not applicable

W_reg     = 0    # [m] Width of regenerator bed
H_reg     = 0    # [m] Height of regenerator bed
L_reg1    = 0.1  # [m] Length of regenerator bed 1
L_reg2    = 0      # [m] Length of regenerator bed 2
casing_th = 0.0016  # [m] Thickness of the casing material. G10 glass fiber reinforced epoxy in this case.
air_gap   = 0.026  # [m] Air gap between magnets: space for placing the AMRs

air_th    = air_gap - H_reg - 2 * casing_th  # [m] Thickness of the air layer in between regenerator casing and magnets

# Note: When working in a dynamic way, e.g. using a GUI for the inputs, the following can be implemented
# if air_th < 0:
#     print("Regenerator plus casing dimensions exceed air gap")

# ---- Overal shape: series regenerators of cylindrical shape in the core of a halbach cilynder

# NOTE!: set to zero all parameters if not applicable

r1      = 0.0111       # [m] Inner Radius of regenerator casing
r2      = 0.0127       # [m] Outer Radius of regenerator casing
r3      = 0.0135       # [m] Bore Radius of magnet

voids_same_radius = "yes"

if voids_same_radius == "yes":
    rvs = 0
    rvs1 = 0
    rvs2 = 0
    L_add = 0
else:
    # TODO: eliminate these particularities of the PM1 device and make it more general

    L_add   = 0                         # [m] Lenght of pipe from cold side void to the check valve
    rv      = 0                       # [m] Radius of cold side void to the check valve pipe
    Vvoid   = 458.24e-9 + rv**2*3.14*L_add  # [m^3]
    Lvoid   = 14.6e-3 + L_add               # [m] Length of void volume and connecting pipe together

    Vvoid1  = 0     # [m^3] Void volume between the cold side of the regenerators and the CHEX
    Lvoid1  = 0     # [m] Length of void volume 1

    Vvoid2  = 0     # [m^3] Void volume between the two regenerators located in the bore of the magnet
    Lvoid2  = 0     # [m] Length of void volume 2

    rvs = np.sqrt(Vvoid/(3.14*Lvoid))    # [m] Cold side void equivalent radius
    rvs1 = np.sqrt(Vvoid1/(3.14*Lvoid1))  # [m] Hot  side void equivalent radius
    rvs2 = np.sqrt(Vvoid2/(3.14*Lvoid2))  # [m] Intermediate void equivalent radius

# Parameters common to all geometries

if r1 != 0 and W_reg == 0:
    Ac = np.pi * r1 ** 2  # [m^2] Cross sectional area of one regenerator
    Pc = 2 * np.pi * r1   # [m] Perimeter of the cross section of the regenerator
elif r1 == 0 and W_reg != 0:
    Ac = H_reg * W_reg          # [m^2] Cross sectional area of one regenerator
    Pc = W_reg * 2 + H_reg * 2  # [m] Perimeter of the cross section of the regenerator

Nd = 0.36  # [-] Demagnetization coefficient

# ------- Description of layers: materials and lengths -------

# Note: this is very device specific. The following species can be used.

# void: for voids of cylindrical shape surrounded by two annular layers of materials,
# ultem and G11, before the air gap with the magnet

# void1: for voids of cylindrical shape surrounded by two annular layers of materials,
# ultem and G11, before the air gap with the magnet. Generally used in the middle of two beds.
# Useful if a different geometry than "void" is needed.

# void2: for voids of cylindrical shape surrounded by two annular layers of materials,
# # ultem and G11, before the air gap with the magnet. Generally used in the middle of two beds.
# # Useful if a different geometry than "void" and "void1" is needed.

# void3: for voids of cuboid shape surrounded by one layer of insulating material,
# G11, before the air gap with the magnet. The casing has the same dimensions around the void and around a bed

# void4: for voids of cylindrical shape surrounded by casing of G11 material, air gap, and magnet. Voids simulated
# as an empty space adjacent to the regenerator bed with the same diameter as the bed.

# gs: packed bed of glass spheres

# ls: packed bed of lead spheres

# reg-MCM: packed bed of MCM

species_discription = ['void4', 'reg-Gd', 'void4']  # Different species found along the axis of the regenerator assembly
x_discription       = [0, 0.0026, 0.1026, 0.1052]    # [m] position of each species in the assembly

# Active bed of MCM

Dsp  = 550e-6  # [m] Diameter of MCM spheres
er   = 0.362    # [-]
mK   = 10.5      # [W/(m K)]...DP: this was 6 and is an assumed thermal conductivity of the MCM
mRho = 7900    # [kg/m^3]... DP: This was 6100 and is an assumed MCM density to calculate porosity of bed

# (BOROSILICATE) Glass spheres

Dspgs = 0  # [m]
egs   = 0  # [-]
gsCp  = 0  # [J/(kg K)]
gsK   = 0  # [W/(m K)]
gsRho = 0  # [kg/m^3]
# http://www.scientificglass.co.uk/contents/en-uk/d115_Physical_Properties_of_Borosilicate_Glass.html
# http://www.schott.com/borofloat/english/attribute/thermic/index.html
# http://www.schott.com/d/tubing/9a0f5126-6e35-43bd-bf2a-349912caf9f2/schott-algae-brochure-borosilicate.pdf

# Lead spheres

Dspls = 0  # [m]
els   = 0  # [-]
lsCp  = 0  # [J/(kg K)]
lsK   = 0  # [W/(m K)]
lsRho = 0  # [kg/m^3]

# Casing and insulating materials

kair  = 0.0255  # [W/(m K)] air material
kg10  =  0.608  # [W/(m K)] Thermal conductivity g10 material (http://cryogenics.nist.gov/MPropsMAY/G-10%20CR%20Fiberglass%20Epoxy/G10CRFiberglassEpoxy_rev.htm)
kult  =  0.122  # [W/(m K)] Thermal conductivity Ultem (https://www.plasticsintl.com/datasheets/ULTEM_GF30.pdf)

# Other parameters

percGly = 20 # [%] Percentage of glycol in the water glycol mixture used as HTF
CL_set  = "Tamb"  # Casing BC assumption
MOD_CL  = 0  # Do we wish to use experimental modication of the BC
ch_fac  = 0.5  # Averaging heating and cooling properties

