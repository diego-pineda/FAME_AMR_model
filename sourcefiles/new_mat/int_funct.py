import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d, interp2d
import pandas as pd


def material_data(mat_name):

    # Note: this function returns numpy arrays with material data and receives as only input an string with the name of
    # the material, which according to the adopted convention must start with M followed by a consecutive number.
    try:
        cpdat_c_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_cp_c.txt', sep='\t', lineterminator='\n', header=None)
        cpdat_c    = pd.DataFrame(cpdat_c_df).to_numpy()
        cpdat_h_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_cp_h.txt', sep='\t', lineterminator='\n', header=None)
        cpdat_h    = pd.DataFrame(cpdat_h_df).to_numpy()
        # Magnetization
        magdata_c_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_Mag_c.txt', sep='\t', lineterminator='\n', header=None)
        magdata_c    = pd.DataFrame(magdata_c_df).to_numpy()
        magdata_h_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_Mag_h.txt', sep='\t', lineterminator='\n', header=None)
        magdata_h    = pd.DataFrame(magdata_h_df).to_numpy()
        # Entropy
        datstot_c_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_S_c.txt', sep='\t', lineterminator='\n', header=None)
        datstot_c    = pd.DataFrame(datstot_c_df).to_numpy()
        datstot_h_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_S_h.txt', sep='\t', lineterminator='\n', header=None)
        datstot_h    = pd.DataFrame(datstot_h_df).to_numpy()
    except FileNotFoundError:
        cpdat_c_df   = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_cp_c.csv', sep=';', header=None)
        cpdat_c = pd.DataFrame(cpdat_c_df).to_numpy()
        cpdat_h_df   = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_cp_h.csv', sep=';', header=None)
        cpdat_h  = pd.DataFrame(cpdat_h_df).to_numpy()
        # Magnetization
        magdata_c_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_Mag_c.csv', sep=';', header=None)
        magdata_c = pd.DataFrame(magdata_c_df).to_numpy()
        magdata_h_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_Mag_h.csv', sep=';', header=None)
        magdata_h = pd.DataFrame(magdata_h_df).to_numpy()
        # Entropy
        datstot_c_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_S_c.csv', sep=';', header=None)
        datstot_c = pd.DataFrame(datstot_c_df).to_numpy()
        datstot_h_df = pd.read_csv('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_S_h.csv', sep=';', header=None)
        datstot_h = pd.DataFrame(datstot_h_df).to_numpy()
    # # Specifc Heat
    # cpdat_c   = np.loadtxt('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_cp_c.txt')
    # cpdat_h   = np.loadtxt('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_cp_h.txt')
    # # Magnetization
    # magdata_c = np.loadtxt('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_Mag_c.txt')
    # magdata_h = np.loadtxt('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_Mag_h.txt')
    # # Entropy
    # datstot_c = np.loadtxt('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_S_c.txt')
    # datstot_h = np.loadtxt('sourcefiles/new_mat/'+mat_name+'/'+mat_name+'_S_h.txt')

    return cpdat_c, cpdat_h, magdata_c, magdata_h, datstot_c, datstot_h


# Defining functions that return "Interpolation functions" for every material property based on exp. data as input.


def matCp_c(cpdat_c):
    # Heat capacity cooling
    HintCp_c = cpdat_c[0, 1:]
    TempCp_c = cpdat_c[1:, 0]
    mCp_c = RectBivariateSpline(TempCp_c, HintCp_c, cpdat_c[1:, 1:], ky=1, kx=1)
    return mCp_c

def matCp_h(cpdat_h):
    # Heat capacity heating
    HintCp_h = cpdat_h[0, 1:]
    TempCp_h = cpdat_h[1:, 0]
    mCp_h = RectBivariateSpline(TempCp_h, HintCp_h, cpdat_h[1:, 1:], ky=1, kx=1)
    return mCp_h

def matMag_c(magdata_c):
    # Magnetisation cooling
    HintMag_c = magdata_c[0, 1:]
    TempMag_c = magdata_c[1:, 0]
    mMag_c = RectBivariateSpline(TempMag_c, HintMag_c, magdata_c[1:, 1:], ky=1, kx=1)
    return mMag_c

def matMag_h(magdata_h):
    # Magnetisation heating
    HintMag_h = magdata_h[0, 1:]
    TempMag_h = magdata_h[1:, 0]
    mMag_h = RectBivariateSpline(TempMag_h, HintMag_h, magdata_h[1:, 1:], ky=1, kx=1)
    return mMag_h

def matS_c(datstot_c):
    # Entropy Cooling
    HintStot_c = datstot_c[0, 1:]
    TempStot_c = datstot_c[1:, 0]
    mS_c = RectBivariateSpline(TempStot_c, HintStot_c, datstot_c[1:, 1:], kx=1, ky=1)
    return mS_c

def matS_h(datstot_h):
    # Entropy Heating
    HintStot_h = datstot_h[0, 1:]
    TempStot_h = datstot_h[1:, 0]
    mS_h = RectBivariateSpline(TempStot_h, HintStot_h, datstot_h[1:, 1:], kx=1, ky=1)
    return mS_h