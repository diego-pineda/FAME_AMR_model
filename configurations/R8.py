##### Regenerator geometry #####

import numpy as np

# ---- Overall shape: single regenerator rectangular cross section in the gap of a C-shaped rotational magnet

W_reg     = 0.045  # [m] Width of regenerator bed
H_reg     = 0.013  # [m] Height of regenerator bed
L_reg1    = 0.060  # [m] Length of regenerator bed 1
L_reg2    = 0      # [m] Length of regenerator bed 2
casing_th = 0.005  # [m] Thickness of the casing material. G10 glass fiber reinforced epoxy in this case.
air_gap   = 0.026  # [m] Air gap between magnets: space for placing the AMRs

air_th    = air_gap - H_reg - 2 * casing_th  # [m] Thickness of the air layer in between regenerator casing and magnets

# TODO: When working in a dynamic way, e.g. using a GUI for the inputs, the following can be implemented
# if air_th < 0:
#     print("Regenerator plus casing dimensions exceed air gap")

# ---- Overal shape: series regenerators of cylindrical shape in the core of a Halbach cylinder

r1      = 0       # [m] Inner Radius of regenerator casing
r2      = 0       # [m] Outer Radius of regenerator casing
r3      = 0       # [m] Bore Radius of magnet

# TODO: eliminate these particularities of the PM1 device and make it more general

L_add   = 18e-3                         # [m] Lenght of pipe from cold side void to the check valve
rv      = 1.9e-3                        # [m] Radius of cold side void to the check valve pipe
Vvoid   = 458.24e-9 + rv**2*3.14*L_add  # [m^3]
Lvoid   = 14.6e-3 + L_add               # [m] Length of void volume and connecting pipe together

Vvoid1  = 0     # [m^3] Void volume between the cold side of the regenerators and the CHEX
Lvoid1  = 0     # [m] Length of void volume 1

Vvoid2  = 0     # [m^3] Void volume between the two regenerators located in the bore of the magnet
Lvoid2  = 0     # [m] Length of void volume 2

rvs     = np.sqrt(Vvoid/(3.14*Lvoid))    # [m] Cold side void equivalent radius
rvs1    = np.sqrt(Vvoid1/(3.14*Lvoid1))  # [m] Hot  side void equivalent radius
rvs2    = np.sqrt(Vvoid2/(3.14*Lvoid2))  # [m] Intermediate void equivalent radius

# ------- Cross sectional area and perimeter. Parameters common to all geometries

if r1 != 0 and W_reg == 0:
    Ac = np.pi * r1 ** 2  # [m^2] Cross sectional area of one regenerator
    Pc = 2 * np.pi * r1   # [m] Perimeter of the cross section of the regenerator
elif r1 == 0 and W_reg != 0:
    Ac = H_reg * W_reg          # [m^2] Cross sectional area of one regenerator
    Pc = W_reg * 2 + H_reg * 2  # [m] Perimeter of the cross section of the regenerator

Nd     = 0.36  # [-] Demagnetization coefficient

# ------- Description of layers: materials and lengths -------

species_discription = ['void', 'reg-M0', 'void']  # Different species found along the axis of the regenerator assembly
x_discription       = [0, 0.006, 0.066, 0.072]    # [m] position of each species in the assembly
reduct_coeff        = dict(M0=1)  # [-] Reduction coefficients used for taking into account thermal hysteresis

# Active bed of MCM

Dsp  = 600e-6  # [m] Diameter of MCM spheres
er   = 0.53    # [-]
mK   = 10.5      # [W/(m K)]...DP: this was 6 and is an assumed thermal conductivity of the MCM
mRho = 7900    # [kg/m^3]... DP: This was 6100 and is an assumed MCM density to calculate porosity of bed

# (BOROSILICATE) Glass spheres

Dspgs = 0.003175  # [m]
egs     = 0.43    # [-]
gsCp  = 800       # [J/(kg K)]
gsK   = 1.2       # [W/(m K)]
gsRho = 2230      # [kg/m^3]
# http://www.scientificglass.co.uk/contents/en-uk/d115_Physical_Properties_of_Borosilicate_Glass.html
# http://www.schott.com/borofloat/english/attribute/thermic/index.html
# http://www.schott.com/d/tubing/9a0f5126-6e35-43bd-bf2a-349912caf9f2/schott-algae-brochure-borosilicate.pdf

# Lead spheres

Dspls = 300e-6  # [m]
els   = 0.36    # [-]
lsCp  = 830     # [J/(kg K)]
lsK   = 1.005   # [W/(m K)]
lsRho = 2230    # [kg/m^3]

# Casing and insulating materials

kair  = 0.0255  # [W/(m K)] air material
kg10  =  0.608  # [W/(m K)] Thermal conductivity g10 material (http://cryogenics.nist.gov/MPropsMAY/G-10%20CR%20Fiberglass%20Epoxy/G10CRFiberglassEpoxy_rev.htm)
kult  =  0.122  # [W/(m K)] Thermal conductivity Ultem (https://www.plasticsintl.com/datasheets/ULTEM_GF30.pdf)

# Other parameters

percGly = 20 # [%] Percentage of glycol in the water glycol mixture used as HTF
CL_set  = "Tamb"  # Casing BC assumption
MOD_CL  = 0  # Do we wish to use experimental modication of the BC
ch_fac  = 0.5  # Averaging heating and cooling properties

