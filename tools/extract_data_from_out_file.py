import numpy as np
import os
from tools.write_data_to_file import FileSaveMatrix
import openpyxl

# README: select the whole script and run it from the Python console.
root_directory = './output/FAME_MnFePSi/'
batch_name = 'MAGNETO_40K_PSB_120mm_250um_26layers_850mT'
number_of_cases = 128


def write_to_excel_array(filename, sheet_name, start_row, start_col, array):
    """
    Writes a 2D array to a specified range in an Excel file.

    Args:
        filename: The name of the Excel file.
        sheet_name: The name of the sheet to write to.
        start_row: The starting row index of the range.
        start_col: The starting column index of the range.
        array: The 2D array to write.
    """

    try:
        # Load the workbook
        workbook = openpyxl.load_workbook(filename)

        # Get the specified sheet
        sheet = workbook[sheet_name]

        # Write the array to the specified range
        for row_index, row in enumerate(array, start=start_row):
            for col_index, value in enumerate(row, start=start_col):
                sheet.cell(row=row_index, column=col_index).value = value

        # Save the workbook
        workbook.save(filename)

        # Close the workbook
        workbook.close()

        print("Array written successfully to Excel file.")

    except Exception as e:
        print(f"Error occurred: {e}")


directory = root_directory + batch_name + '/out_files/'
text_file_name = root_directory + batch_name + '/' + batch_name + '.txt'
excel_file_name = root_directory + batch_name + '/' + batch_name + '.xlsx'

results = np.ones((number_of_cases, 23))  # 23 is the number of columns that are obtained from the .out files
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
            results[case, 0] = case
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 1, case)
        if line.startswith('Enthalpy_flow_cold_side_tF'):
            h_c1 = float((line.split(' '))[2])
            results[case, 1] = h_c1
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 2, h_c1)
        if line.startswith('Enthalpy_flow_hot_side_tF'):
            h_h1 = float((line.split(' '))[2])
            results[case, 2] = h_h1
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 3, h_h1)
        if line.startswith('Enthalpy_flow_cold_side_ave_cp'):
            h_c2 = float((line.split(' '))[2])
            results[case, 3] = h_c2
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 4, h_c2)
        if line.startswith('Enthalpy_flow_hot_side_ave_cp'):
            h_h2 = float((line.split(' '))[2])
            results[case, 4] = h_h2
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 5, h_h2)
        if line.startswith('Power in out cold side'):
            Qc1 = float((line.split(' '))[6])
            results[case, 5] = Qc1
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 6, Qc1)
        if line.startswith('Power in out hot side'):
            Qh1 = float((line.split(' '))[6])
            results[case, 6] = Qh1
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 7, Qh1)
        if line.startswith('Qc variable cp'):
            Qc2 = float((line.split(' '))[4])
            results[case, 7] = Qc2
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 8, Qc2)
        if line.startswith('Qh variable cp'):
            Qh2 = float((line.split(' '))[4])
            results[case, 8] = Qh2
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 9, Qh2)
        if line.startswith('Cycle average cooling capacity'):
            Qc3 = float((line.split(' '))[5])
            results[case, 9] = Qc3
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 10, Qc3)
        if line.startswith('Cycle average heating capacity'):
            Qh3 = float((line.split(' '))[5])
            results[case, 10] = Qh3
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 11, Qh3)
        if line.startswith('Cycle average heat leaks'):
            Qleak = float((line.split(' '))[5])
            results[case, 11] = Qleak
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 12, Qleak)
        if line.startswith('Cycle average pumping power'):
            Wpump = float((line.split(' '))[5])
            results[case, 12] = Wpump
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 13, Wpump)
        if line.startswith('Cycle average magnetic power'):
            if (line.split(' '))[5] != '=':
                if (line.split(' '))[3] != 'power2':
                    Wmag1 = float((line.split(' '))[5])
                    results[case, 13] = Wmag1
                    # write_to_excel(excel_file_name, 'Sheet1', case+2, 14, Wmag1)
        if line.startswith('Cycle average magnetic power2'):
            if (line.split(' '))[5] != '=':
                Wmag2 = float((line.split(' '))[5])
                results[case, 14] = Wmag2
                # write_to_excel(excel_file_name, 'Sheet1', case+2, 15, Wmag2)
        if line.startswith('Cycle average magnetic power old'):
            Wmag1old = float((line.split(' '))[6])
            results[case, 15] = Wmag1old
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 16, Wmag1old)
        if line.startswith('Cycle average magnetic power2 old'):
            Wmag2old = float((line.split(' '))[6])
            results[case, 16] = Wmag2old
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 17, Wmag2old)
        if line.startswith('E_accum_liq'):
            E_accu_liq = float((line.split(' '))[2])
            results[case, 17] = E_accu_liq
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 18, E_accu_liq)
        if line.startswith('Q_diff_cold'):
            Q_diff_cold = float((line.split(' '))[2])
            results[case, 18] = Q_diff_cold
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 19, Q_diff_cold)
        if line.startswith('Q_diff_hot'):
            Q_diff_hot = float((line.split(' '))[2])
            results[case, 19] = Q_diff_hot
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 20, Q_diff_hot)
        if line.startswith('Q_MCE2'):
            Q_MCE = float((line.split(' '))[2])
            results[case, 20] = Q_MCE
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 21, Q_MCE)
        if line.startswith('Error 8_2'):
            error8_2 = float((line.split(' '))[3])
            results[case, 21] = error8_2
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 22, error8_2)
        if line.startswith('Error 12'):
            error12 = float((line.split(' '))[3])
            results[case, 22] = error12
            # write_to_excel(excel_file_name, 'Sheet1', case+2, 23, error12)
    # error8_2 = ((Qh2 + Qleak - Qc2 + Q_diff_cold - Q_diff_hot) - (Wpump - Wmag2old))*100/(Wpump - Wmag2old) # float((line.split(' '))[3])
    # results[i, 21] = error8_2
    # error12 = ((h_h2 + Qleak - h_c2 + Q_diff_cold - Q_diff_hot) - (Wpump - Wmag2old))*100/(Wpump - Wmag2old) # float((line.split(' '))[3])
    # results[i, 22] = error12

    i = i+1

results = np.delete(results, slice(i, number_of_cases, 1), 0)

write_to_excel_array(excel_file_name, 'Output_data', 2, 1, results)
FileSaveMatrix(text_file_name, results)


# TODO:
#  1) Make this code either a function or executable from terminal so that it does not change every time it is run
#  2) Return a list of missing cases
#  3) Remove data that is not used
#  4) put write_to_excel_array() function maybe in tools folder as an independent file

