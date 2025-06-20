# %%%%%%%%% FAME Cooler applied field function %%%%%%%%%%

# Assumptions
# 1) The whole regenerator is magnetized simultaneously. This is necessary because of the 1D nature of the model
# even though the real situation is that the regenerator is swept by the magnetic field so that one edge of the
# regenerator is magnetized first.
# 2) The magnetic field intensity in the air gap between the magnets is uniform in the portion occupied by the
# regenerator in the radial direction.
# 3) The magnetic field intensity in the air gap only varies in the azimuthal direction.
# 4) The maximum magnetic field intensity is 0.875 [T] and the minimum is 0 [T]
# 5) The magnetic field increases linearly from 0 [T] to 0.875 [T] while the magnet is swept an angle of 30°
# 6) This code tries to emulate the magnetic field produced by the magnet assembly of the FAME cooler developed at TU Delft
# See: B. Huang, J. Lai, D. Zeng, Z. Zheng, B. Harrison, A. Oort, N. van Dijk, and E. Brück, “Development of an experimental
# rotary magnetic refrigerator prototype,” Int. J. Refrig., vol. 104, Aug. 2019.
# 7) The frequency of the magnetic refrigeration cycle is twice as large as the frequency of rotation. So, one magnetic refrigeration cycle
# is completed in a rotation angle of 180°

import numpy as np
# max_app_field = 0.875


def app_field(nt, N, max_app_field):
    ap_fld = np.zeros((nt+1, N+1))
    angle = 0
    full_mag_ang = 30
    max_field_period = 45

    for n in range(0, nt+1):
        if angle < full_mag_ang:
            ap_fld[n, :] = (max_app_field/full_mag_ang)*(180*(n/nt))  # [T]. Applied field as a function of time during the magnetization ramp
        elif full_mag_ang <= angle < (full_mag_ang + max_field_period):
            ap_fld[n, :] = max_app_field # [T] Applied field as a function of time during the cold to hot blow process
        elif (full_mag_ang + max_field_period) <= angle < (2 * full_mag_ang + max_field_period):
            ap_fld[n, :] = (max_app_field/full_mag_ang)*((2 * full_mag_ang + max_field_period)-180*n/nt) # [T] Applied field during the demagnetization ramp
        else:
            ap_fld[n, :] = 0 # [T] Applied field during the hot to cold blow process
        angle = 180*(n+1)/nt
    return ap_fld


if __name__ == '__main__':

    # %%%%%%%%%%%%%%%% The following lines are for testing the function %%%%%%%%%%%%%%%%

    nt = 600  # number of nodes in the time domain
    N = 200  # number of nodes in the spatial domain
    mag_field = 1.4

    ap_field = app_field(nt, N, mag_field)
    # print(ap_field)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

    fig, plot1 = plt.subplots()
    plt.plot(np.linspace(0, 180, nt+1), ap_field[:, 0])
    plt.xlabel("Angle [°]")
    plt.ylabel("Applied field [T]")
    plt.title("Applied magnetic field as a function of rotation angle")
    plt.grid(which='both', axis='both')
    plot1.xaxis.set_major_locator(MultipleLocator(15))
    plt.show()

    # %%%%%%%%% Ploting the results of Bowei's simulation about magnetic field %%%%%%%%%%

    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # from matplotlib.ticker import LinearLocator, FormatStrFormatter
    # import numpy as np
    #
    # # Import the data
    # mag_field_data = np.loadtxt('Magfield_profile.txt')
    # X = mag_field_data[1:, 0]
    # Y = mag_field_data[0, 1:]
    # X, Y = np.meshgrid(X, Y)
    # Z = np.matrix.transpose(mag_field_data[1:, 1:])
    # print(Z)
    #
    # # Plot the surface.
    # fig = plt.figure(1)
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #
    # # Customize the z axis.
    # ax.set_zlim(0, 1)
    # ax.zaxis.set_major_locator(LinearLocator(11))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()

