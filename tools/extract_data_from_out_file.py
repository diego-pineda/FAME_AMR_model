import numpy as np
import os
from tools.write_data_to_file import FileSaveMatrix


directory = './output/FAME_MnFePSi/SCOP_article/outputs'
number_of_cases = 242


results = np.ones((number_of_cases, 23))
i = 0
for files in os.listdir(directory):
    myfile = open(directory + '/' + files, "rt")
    contents = myfile.read()
    myfile.close()
    # case = int(((contents.split('\n'))[7].split(':'))[2])   # TODO: Warning! when starting from pickle file placed in ./pickleddata the first index is 1 instead of 7
    # case = int(((contents.split('\n'))[4].split(' '))[3])
    for line in contents.split('\n'):
        if line.startswith('This is job'):
            case = int((line.split(' '))[3])
            results[i, 0] = case
        if line.startswith('Enthalpy_flow_cold_side_tF'):
            h_c1 = float((line.split(' '))[2])
            results[i, 1] = h_c1
        if line.startswith('Enthalpy_flow_hot_side_tF'):
            h_h1 = float((line.split(' '))[2])
            results[i, 2] = h_h1
        if line.startswith('Enthalpy_flow_cold_side_ave_cp'):
            h_c2 = float((line.split(' '))[2])
            results[i, 3] = h_c2
        if line.startswith('Enthalpy_flow_hot_side_ave_cp'):
            h_h2 = float((line.split(' '))[2])
            results[i, 4] = h_h2
        if line.startswith('Power in out cold side'):
            Qc1 = float((line.split(' '))[6])
            results[i, 5] = Qc1
        if line.startswith('Power in out hot side'):
            Qh1 = float((line.split(' '))[6])
            results[i, 6] = Qh1
        if line.startswith('Qc variable cp'):
            Qc2 = float((line.split(' '))[4])
            results[i, 7] = Qc2
        if line.startswith('Qh variable cp'):
            Qh2 = float((line.split(' '))[4])
            results[i, 8] = Qh2
        if line.startswith('Cycle average cooling capacity'):
            Qc3 = float((line.split(' '))[5])
            results[i, 9] = Qc3
        if line.startswith('Cycle average heating capacity'):
            Qh3 = float((line.split(' '))[5])
            results[i, 10] = Qh3
        if line.startswith('Cycle average heat leaks'):
            Qleak = float((line.split(' '))[5])
            results[i, 11] = Qleak
        if line.startswith('Cycle average pumping power'):
            Wpump = float((line.split(' '))[5])
            results[i, 12] = Wpump
        if line.startswith('Cycle average magnetic power'):
            if (line.split(' '))[5] != '=':
                if (line.split(' '))[3] != 'power2':
                    Wmag1 = float((line.split(' '))[5])
                    results[i, 13] = Wmag1
        if line.startswith('Cycle average magnetic power2'):
            if (line.split(' '))[5] != '=':
                Wmag2 = float((line.split(' '))[5])
                results[i, 14] = Wmag2
        if line.startswith('Cycle average magnetic power old'):
            Wmag1old = float((line.split(' '))[6])
            results[i, 15] = Wmag1old
        if line.startswith('Cycle average magnetic power2 old'):
            Wmag2old = float((line.split(' '))[6])
            results[i, 16] = Wmag2old
        if line.startswith('E_accum_liq'):
            E_accu_liq = float((line.split(' '))[2])
            results[i, 17] = E_accu_liq
        if line.startswith('Q_diff_cold'):
            Q_diff_cold = float((line.split(' '))[2])
            results[i, 18] = Q_diff_cold
        if line.startswith('Q_diff_hot'):
            Q_diff_hot = float((line.split(' '))[2])
            results[i, 19] = Q_diff_hot
        if line.startswith('Q_MCE2'):
            Q_MCE = float((line.split(' '))[2])
            results[i, 20] = Q_MCE
        if line.startswith('Error 8_2'):
            error8_2 = float((line.split(' '))[3])
            results[i, 21] = error8_2
        if line.startswith('Error 12'):
            error12 = float((line.split(' '))[3])
            results[i, 22] = error12
    # error8_2 = ((Qh2 + Qleak - Qc2 + Q_diff_cold - Q_diff_hot) - (Wpump - Wmag2old))*100/(Wpump - Wmag2old) # float((line.split(' '))[3])
    # results[i, 21] = error8_2
    # error12 = ((h_h2 + Qleak - h_c2 + Q_diff_cold - Q_diff_hot) - (Wpump - Wmag2old))*100/(Wpump - Wmag2old) # float((line.split(' '))[3])
    # results[i, 22] = error12

    i = i+1

results = np.delete(results, slice(i, number_of_cases, 1), 0)
print(results)

file_path = './' + directory + '/outputs_per_case.txt'
FileSaveMatrix(file_path, results)
