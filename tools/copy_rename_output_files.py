import shutil
import os
import numpy as np

# %%%%%%%%%%% README %%%%%%%%%
# Write full folder names starting from the disk name
# old_folder and new_folder can be the same
# File named index.txt with old and new indices must be placed in the new folder

# %%%%%%%%%%%%%%%%%%% Inputs %%%%%%%%%%%%%%%%%%%%
old_directory = 'C:/Users/dfpinedaquijan/surfdrive/PhD Project/Numerical Model/MCHP_model_DP/output/FAME_MnFePSi/FAME_Dsp300um_B1400mT_ff_vfl4'
new_directory = 'C:/Users/dfpinedaquijan/surfdrive/PhD Project/Numerical Model/MCHP_model_DP/output/FAME_MnFePSi/FAME_Dsp300um_B1400mT_ff_vfl_intern_voids'
old_file_names = ".0FAME_Dsp300um_B1400mT_ff_vfl4_reused"
new_file_names = ".0FAME_Dsp300um_B1400mT_ff_vfl_intern_voids"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def cp_mv_output_files(old_folder, new_folder, old_name_of_files, new_name_of_files):
    print(new_folder)
    index = np.loadtxt(new_folder + '/' + 'index.txt')
    #index = np.hstack((np.reshape(np.linspace(162, 242, 81), [81, 1]), np.reshape(np.linspace(162, 242, 81), [81, 1])))
    # This is a comment
    for row in index:
        a = old_folder + "/" + str(int(row[0])) + old_name_of_files + ".txt"
        # a = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Numerical Model/MCHP_model_DP/output/" + old_folder + "/" + str(int(row[0])) + old_name_of_files + str(int(row[0])) + ".txt"
        b = new_folder + "/" + str(int(row[1])) + new_name_of_files + ".txt"
        file = str(int(row[0])) + old_name_of_files + ".txt"
        # file = str(int(row[0])) + old_name_of_files + str(int(row[0])) + ".txt"  # Use this for old style of naming of files
        if file in os.listdir(old_folder):  # "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Numerical Model/MCHP_model_DP/" +
            shutil.copyfile(a, b)  # Use this for copying a file and pasting it to a new folder
            # os.rename(a, b)  # Use this for just renaming the files when old and new folders are the same.

        # When reused is included in the name of the files

        # aa = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Numerical Model/MCHP_model_DP/output/" + old_folder + "/" + str(int(row[0])) + old_name_of_files + "_reused.txt"
        # file2 = str(int(row[0])) + old_name_of_files + "_reused.txt"
        # if file2 in os.listdir('output/' + old_folder):
        #     shutil.copyfile(aa, b)  # Use this for copying a file and pasting it to a new folder