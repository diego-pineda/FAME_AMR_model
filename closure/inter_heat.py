import numba as nb
from numba import jit, f8, int32,b1
# Numpy library
import numpy as np


# Engelbrechts Correlation for packed beds suggeted used by Lei.
@jit(f8     (f8 , f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8),nopython=True)
def beHeff_E(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):
    hefff = (0.7*(fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** 0.23) * fK / Dsp
    beta  = 6 * (1 - er) / Dsp
    return hefff*beta


# Beta*Heff based on the work of Iman.
@jit(f8     (f8 , f8,  f8, f8,  f8,   f8,   f8,  f8, f8,   f8, f8),nopython=True)
def beHeff_I(Dsp, Ud, fCp, fK, fMu, fRho, freq, mCp, mK, mRho, er):
    # Iman uses the DF factor, and a Wakao Nusselt.
    #
    # The decomposition of the hefff is as follows:
    # Nu_sp = 2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)
    # Biot = Nu_sp * fK / (2 * mK)
    # Fourier = 4 * mK / (mRho * mCp * freq * Dsp ** 2)
    # phi_H = 1 - 4 / (35 * Fourier)
    # DF = 1 / (1 + Biot / 5 * phi_H)
    # hefff = Nu_sp * fK / Dsp * DF
    # DP: I think Theo avoids creating as many variables as above to save computing memory

    hefff = (2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)) * fK / Dsp / (1 + (2 + 0.11e1 * (fRho * Ud * Dsp / fMu) ** 0.6e0 * (fMu * fCp / fK) ** (0.1e1 / 0.3e1)) * fK / mK / 10 * (1 - mRho * mCp * freq * Dsp ** 2 / mK / 35))
    beta = 6 * (1 - er) / Dsp  # [m2/m3] specific surface area of a packed bed of spheres
    return hefff*beta


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from sourcefiles.fluid.density import fRho
    from sourcefiles.fluid.dynamic_viscosity import fMu
    from sourcefiles.fluid.conduction import fK
    from sourcefiles.fluid.specific_heat import fCp



    T_fluid = 290  # [K] Temperature of fluid
    perc_gly = 20  # [-] Percentage of glycol in the mixture
    Vol_flow_rate = 30.52e-6  # [m3/s] Volumetric flow rate
    Area_cross = 0.045*0.013  # [m2] Cross sectional area of bed
    fAMR = 1.7  # [Hz]
    Gd_K = 14  # [W/(m*K)]
    Gd_rho = 7900  # [kg/m3]
    Gd_cp = 240  # [J/(kg*K)]

    Dsphere = [400e-6, 500e-6, 600e-6, 700e-6, 800e-6]  # [m] Diameter of sphere
    Porosity = [0.3, 0.35, 0.4, 0.45, 0.5]   # [] Porosity of bed
    hI_beta = np.ones((len(Porosity), len(Dsphere)))
    hE_beta = np.ones((len(Porosity), len(Dsphere)))
    hM_beta = np.ones((len(Porosity), len(Dsphere)))
    hM_beta_DF = np.ones((len(Porosity), len(Dsphere)))

    rho_fluid = fRho(T_fluid, perc_gly)
    mu_fluid = fMu(T_fluid, perc_gly)
    cp_fluid = fCp(T_fluid,perc_gly)
    k_fluid = fK(T_fluid,perc_gly)
    Ud = Vol_flow_rate/Area_cross

    x = 0

    for er in Porosity:
        y = 0
        for Dsp in Dsphere:

            beta = 6 * (1 - er) / Dsp

            hI_beta[x, y] = beHeff_I(Dsp, Ud, cp_fluid, k_fluid, mu_fluid, rho_fluid, fAMR, Gd_cp, Gd_K, Gd_rho, er)
            hE_beta[x, y] = beHeff_E(Dsp, Ud, cp_fluid, k_fluid, mu_fluid, rho_fluid, fAMR, Gd_cp, Gd_K, Gd_rho, er)

            # Heat transfer coefficient calculated by using the correlation presented in the book of Mills

            Vel_char = Ud / er
            Len_char = Dsp * (er / (1 - er))
            Re = Vel_char * Len_char * rho_fluid / mu_fluid
            Pr = cp_fluid * mu_fluid / k_fluid
            Nu = (0.5 * Re ** 0.5 + 0.2 * Re ** (0.2e1/0.3e1)) * Pr ** (0.1e1 / 0.3e1)
            hM_beta[x, y] = (Nu * k_fluid / Len_char) * beta

            Biot = Nu * k_fluid / (2 * Gd_K)
            print("Bi = {}".format(Biot))
            Fourier = 4 * Gd_K / (Gd_rho * Gd_cp * fAMR * Dsp ** 2)
            print("Fo = {}".format(Fourier))
            phi_H = 1 - 4 / (35 * Fourier)
            DF = 1 / (1 + Biot / 5 * phi_H)
            print("DF = {}".format(DF))
            hM_beta_DF[x, y] = (Nu * k_fluid / Len_char) * DF * beta

            y = y + 1

        x = x + 1

    #print("hE = {} [W/(m2*K)], hI = {} [W/(m2*K)], hM = {} [W/(m2*K)]".format(hE, hI, hM))
    # print(hI)
    # print(hM)
    fig1 = plt.figure(1)
    plt.plot(Dsphere, hI_beta)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Porosity = 0.3', 'Porosity = 0.35', 'Porosity = 0.4', 'Porosity = 0.45', 'Porosity = 0.5'])
    plt.xlabel("Diameter of spheres [m]")
    plt.ylabel("h*beta [W/($m^3$*K)]")
    plt.title("Wakao & Kagei (1982) Correlation")
    plt.grid(which='major',axis='both')

    fig2 = plt.figure(2)
    plt.plot(Dsphere, hM_beta)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Porosity = 0.3', 'Porosity = 0.35', 'Porosity = 0.4', 'Porosity = 0.45', 'Porosity = 0.5'])
    plt.xlabel("Diameter of spheres [m]")
    plt.ylabel("h*beta [W/($m^3$*K)]")
    plt.title("Whitaker (1972) Correlation")
    plt.grid(which='major',axis='both')

    fig3 = plt.figure(3)
    plt.plot(Dsphere, hM_beta_DF)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Porosity = 0.3', 'Porosity = 0.35', 'Porosity = 0.4', 'Porosity = 0.45', 'Porosity = 0.5'])
    plt.xlabel("Diameter of spheres [m]")
    plt.ylabel("h*beta [W/($m^3$*K)]")
    plt.title("Correlation in Mills's book with Degradation Factor (DF)")
    plt.grid(which='major',axis='both')

    #plt.show()

    # Part II: Comparison of heat transfer coefficients as function of velocity for given porosity and particle diameter

    flow_rates = np.array([10e-6, 15e-6, 20e-6, 25e-6, 30e-6, 35e-6, 40e-6, 45e-6, 50e-6])  # [m3/s]
    Area_cross = 0.045*0.013  # [m2] Cross sectional area of bed
    velocities = flow_rates / Area_cross
    er = 0.36  # [-]
    Dsp = 600e-6  # [m]
    x = 0
    y = 0
    hI_beta = np.ones((1, len(flow_rates)))  # Wakao and Kaguei (1982). Re based on Darcean velocity and diam. of sphere
    hE_beta = np.ones((1, len(flow_rates)))  # Engelbrecht (2008). Re based on Darcean velocity and diam. of sphere
    hPG_beta = np.ones((1, len(flow_rates)))  # Pallares and Grau (2010). Re based on Darcean vel. and diam. of sphere
    hPG_beta_DF = np.ones((1, len(flow_rates)))  # Pallares and Grau (2010). Re based on Darcean vel. and diam. of sphere

    hW72_beta = np.ones((1, len(flow_rates)))  # Whitaker (1972). Re based on pore vel. and hydraulic diameter
    hW72_beta_DF = np.ones((1, len(flow_rates)))  # Whitaker (1972). Re based on pore vel. and hydraulic diameter
    hW83_beta = np.ones((1, len(flow_rates)))  # Whitaker (1983). Re based on pore vel. and hydraulic diameter
    hW83_beta_DF = np.ones((1, len(flow_rates)))  # Whitaker (1983). Re based on pore vel. and hydraulic diameter
    hMM91_beta = np.ones((1, len(flow_rates)))  # Macias-Machin (1991). Re based on pore vel. and hydraulic diameter

    hKL91_beta = np.ones((1, len(flow_rates)))  # Kunii and Levenspiel (1991). Re based on pore vel. and diam. of sph
    hKL91_beta_DF = np.ones((1, len(flow_rates)))  # Kunii and Levenspiel (1991). Re based on pore vel. and diam. of sph
    hMM91_2_beta_DF = np.ones((1, len(flow_rates)))  # Macias-Machin (1991). Re based on pore vel. and diam. of sphere
    hMM91_2_beta = np.ones((1, len(flow_rates)))  # Macias-Machin (1991). Re based on pore vel. and diam. of sphere

    for Ud in velocities:

        beta = 6 * (1 - er) / Dsp

        # Heat transfer coefficient correlations based on Darcean velocity and diameter of spheres

        # Wakao and Kaguei (1982) with DF

        hI_beta[x, y] = beHeff_I(Dsp, Ud, cp_fluid, k_fluid, mu_fluid, rho_fluid, fAMR, Gd_cp, Gd_K, Gd_rho, er)

        # Engelbrecht (2008)

        hE_beta[x, y] = beHeff_E(Dsp, Ud, cp_fluid, k_fluid, mu_fluid, rho_fluid, fAMR, Gd_cp, Gd_K, Gd_rho, er)

        # Pallares and Grau (2010) with DF

        Nu_PG = 2 * (1 + 4 * (1 - er) / er) + (1 - er) ** 0.5e0 * (rho_fluid * Ud * Dsp / mu_fluid) ** 0.6e0 * (mu_fluid * cp_fluid / k_fluid) ** (0.1e1 / 0.3e1)
        hPG_beta[x, y] = beta * Nu_PG * k_fluid / Dsp
        hPG_beta_DF[x, y] = beta * (Nu_PG * k_fluid / Dsp / (1 + Nu_PG * k_fluid / Gd_K / 10 * (1 - Gd_rho * Gd_cp * fAMR * Dsp ** 2 / Gd_K / 35)))

        # Macias-Machin (1991)

        Nu_MM91_2 = 1.27 + 2.66 * (rho_fluid * Ud * Dsp / mu_fluid / er) ** 0.56 * (mu_fluid * cp_fluid / k_fluid) ** -0.41 * ((1 - er) / er) ** 0.29
        hMM91_2_beta[x, y] = beta * Nu_MM91_2 * k_fluid / Dsp
        hMM91_2_beta_DF[x, y] = beta * (Nu_MM91_2 * k_fluid / Dsp / (1 + Nu_MM91_2 * k_fluid / Gd_K / 10 * (1 - Gd_rho * Gd_cp * fAMR * Dsp ** 2 / Gd_K / 35)))

        # Kunii and Levenspiel (1991)

        Nu_KL91 = 2 + 1.8 * (rho_fluid * Ud * Dsp / mu_fluid / er) ** 0.5 * (mu_fluid * cp_fluid / k_fluid) ** 0.33
        hKL91_beta[x, y] = beta * Nu_KL91 * k_fluid / Dsp
        hKL91_beta_DF[x, y] = beta * (Nu_KL91 * k_fluid / Dsp / (1 + Nu_KL91 * k_fluid / Gd_K / 10 * (1 - Gd_rho * Gd_cp * fAMR * Dsp ** 2 / Gd_K / 35)))

        # Heat transfer coefficients based on pore velocity and hydraulic diameter

        Vel_char = Ud / er
        Len_char = Dsp * (er / (1 - er))
        Re = Vel_char * Len_char * rho_fluid / mu_fluid
        Pr = cp_fluid * mu_fluid / k_fluid
        print("Re = {}, Re^0.6Pr^1/3 = {}".format(Re, Re ** 0.6 * Pr ** (1/3)))

        # Whitaker (1972) presented in Mills

        Nu_W72 = (0.5 * Re ** 0.5 + 0.2 * Re ** (0.2e1/0.3e1)) * Pr ** (0.1e1 / 0.3e1)
        hW72_beta[x, y] = (Nu_W72 * k_fluid / Len_char) * beta

        Biot = Nu_W72 * k_fluid / (2 * Gd_K)
        print("Bi = {}".format(Biot))
        Fourier = 4 * Gd_K / (Gd_rho * Gd_cp * fAMR * Dsp ** 2)
        print("Fo = {}".format(Fourier))
        phi_H = 1 - 4 / (35 * Fourier)
        DF = 1 / (1 + Biot / 5 * phi_H)
        print("DF = {}".format(DF))
        hW72_beta_DF[x, y] = (Nu_W72 * k_fluid / Len_char) * DF * beta

        # Whitaker (1983) presented in Kaviany

        Nu_W83 = 2 + (0.4 * Re ** 0.5 + 0.2 * Re ** (0.2e1/0.3e1)) * Pr ** 0.4  # Whitaker (1983), in Kaviani's book.
        hW83_beta[x, y] = (Nu_W83 * k_fluid / Len_char) * beta

        Biot = Nu_W83 * k_fluid / (2 * Gd_K)
        print("Bi = {}".format(Biot))
        Fourier = 4 * Gd_K / (Gd_rho * Gd_cp * fAMR * Dsp ** 2)
        print("Fo = {}".format(Fourier))
        phi_H = 1 - 4 / (35 * Fourier)
        DF = 1 / (1 + Biot / 5 * phi_H)
        print("DF = {}".format(DF))
        hW83_beta_DF[x, y] = (Nu_W83 * k_fluid / Len_char) * DF * beta

        Nu_MM91 = 1.27 + 2.66 * Re ** 0.56 * Pr ** -0.41 * ((1 - er) / er) ** 0.29
        hMM91_beta[x, y] = (Nu_MM91 * k_fluid / Len_char) * beta

        y = y + 1

    fig4 = plt.figure(4)
    # plt.plot(flow_rates,np.transpose(hKL91_beta))
    plt.plot(flow_rates,np.transpose(hKL91_beta_DF))
    # plt.plot(flow_rates, np.transpose(hPG_beta))
    plt.plot(flow_rates, np.transpose(hPG_beta_DF))
    # plt.plot(flow_rates, np.transpose(hW83_beta))
    plt.plot(flow_rates, np.transpose(hW83_beta_DF))
    # plt.plot(flow_rates, np.transpose(hW72_beta))
    plt.plot(flow_rates, np.transpose(hW72_beta_DF))
    plt.plot(flow_rates, np.transpose(hI_beta))
    # plt.plot(flow_rates,np.transpose(hMM91_2_beta))
    plt.plot(flow_rates,np.transpose(hMM91_2_beta_DF))
    plt.plot(flow_rates,np.transpose(hE_beta))

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Kunii and Levenspiel (1991)', 'Pallares and Grau (2010)',
                'Whitaker (1983)', 'Whitaker (1972)',
                'Wakao and Kaguei (1982)', 'Macias-Machin (1991)', 'Engelbrecht (2008)'])


    # plt.legend(['Kunii and Levenspiel (1991)', 'Kunii and Levenspiel (1991) with DF', 'Pallares and Grau (2010)', 'Pallares and Grau (2010) with DF',
    #             'Whitaker (1983)', 'Whitaker (1983) with DF', 'Whitaker (1972)', 'Whitaker (1972) with DF',
    #             'Wakao and Kaguei (1982)', 'Macias-Machin (1991)', 'Macias-Machin (1991) with DF','Engelbrecht (2008)'])
    plt.xlabel("Flow rates [m$^3$/s]")
    plt.ylabel("h*beta [W/($m^3$*K)]")
    plt.title("Comparison of heat transfer coefficient correlations")
    plt.grid(which='major',axis='both')
    # plt.show()

    # Part III: Variation of Prandtl number with temperature for a mixture of water and glycol

    temperatures = [275, 280, 285, 290, 295, 300]
    i = 0
    Prandtl = np.ones(len(temperatures))
    heat_cap = np.ones(len(temperatures))
    visco = np.ones(len(temperatures))
    therm_cond = np.ones(len(temperatures))

    for Temp in temperatures:
        Prandtl[i] = fCp(Temp,perc_gly) * fMu(Temp, perc_gly) / fK(Temp,perc_gly)
        heat_cap[i] = fCp(Temp,perc_gly)
        visco[i] = fMu(Temp, perc_gly)
        therm_cond[i] = fK(Temp,perc_gly)
        i = i + 1

    fig5 = plt.figure(5)
    plt.plot(temperatures, Prandtl)
    plt.plot(temperatures,heat_cap / 1000)
    plt.plot(temperatures,visco * 1e3)
    plt.plot(temperatures,therm_cond)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(['Prandtl', 'Heat capacity', 'Viscosity', 'Thermal conductivity'])
    plt.xlabel("Fluid temperature [K]")
    plt.ylabel("Prandtl number [-]")
    plt.title("Prandtl of water / glycol mixture vs T")
    plt.grid(which='major',axis='both')
    plt.show()

    # Conclusion: the variation of Prandtl number with temperature is mainly due to the variation of the viscosity of
    # the mixture with temperature. Heat capacity and thermal conductivity of the mixture remain fairly constant in the
    # temperature range of interest.
