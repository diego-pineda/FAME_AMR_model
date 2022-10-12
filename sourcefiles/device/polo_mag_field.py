import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d

# Creating the data for the interpolation function

rot_angle = np.linspace(0, 360, 361)

B0 = 1.688 * np.abs(np.sin(rot_angle / 2 * np.pi / 180 + np.pi / 4))
B1 = 1.688 * np.abs(np.sin(rot_angle / 2 * np.pi / 180 + np.pi / 4))
B2 = 1.677 * np.abs(np.sin(rot_angle / 2 * np.pi / 180 + np.pi / 4))
B3 = 1.629 * np.abs(np.sin(rot_angle / 2 * np.pi / 180 + np.pi / 4))
B4 = 1.533 * np.abs(np.sin(rot_angle / 2 * np.pi / 180 + np.pi / 4))
B5 = 1.296 * np.abs(np.sin(rot_angle / 2 * np.pi / 180 + np.pi / 4))

B = np.array([B5, B4, B3, B2, B1, B0, B1, B2, B3, B4, B5])
x = np.array([-0.050, -0.040, -0.030, -0.020, -0.010, 0, 0.010, 0.020, 0.030, 0.040, 0.050])

# Create the interpolation function

Rotation = rot_angle
xPosition = x
Z = np.matrix.transpose(B)

appliedField = RectBivariateSpline(Rotation, xPosition, Z, kx=1, ky=1)

# Plot Magnetic flux density as function of rotating angle and position in a 2D plot.

# A = np.matrix.transpose(np.array([B0, B1, B2, B3, B4, B5]))
# plt.figure(1)
# plt.plot(np.transpose(rot_angle), A)
# plt.xlabel("Rotating angle [deg]")
# plt.ylabel("Magnetic flux density [T]")
# plt.grid(which='both')
#
# # Plot the surface.
#
# R, X = np.meshgrid(rot_angle, x)
#
# fig = plt.figure(2)
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(R, X, B, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax.set_zlabel("Applied field [T]")
# plt.xlabel("Rotating angle [deg]")
# plt.ylabel("Axial direction [m]")
#
# ax.set_xlim(0, 360)
# ax.xaxis.set_major_locator(LinearLocator(7))
# ax.set_ylim(-0.05, 0.05)
# ax.yaxis.set_major_locator(LinearLocator(11))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()


def polo_app_field(nt, N, L_reg):

    ap_fld = np.zeros((nt+1, N+1))
    angles = np.linspace(0, 360, nt+1)
    x_positions = np.linspace(-L_reg/2, L_reg/2, N+1)

    i = 0
    for angle in angles:
        j = 0
        for x_position in x_positions:
            ap_fld[i, j] = appliedField(angle, x_position)[0][0]
            j = j + 1
        i = i + 1

    return ap_fld

# The following lines are just for testing the function


if __name__ == '__main__':

    a = polo_app_field(8,10,0.1)
    angles = np.linspace(0, 360, 8+1)

    plt.plot(np.transpose(angles),a)
    plt.show()
