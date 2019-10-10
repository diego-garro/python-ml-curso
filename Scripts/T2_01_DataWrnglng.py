#-*- coding: utf-8 -*-

import pandas as pd

# Data Wrangling

main_path = '/home/diego/Documentos/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/customer-churn-model/Customer Churn Model.txt'
full_path = main_path + file_name

data = pd.read_csv(full_path)
print(data.head())

# Extraer una sola columna
#account_length = data['Account Length']
#print(account_length.head())

# Extraer varias columnas
#desired_cols = ['Account Length', 'Phone', 'Eve Charge', 'Day Calls']
#subset = data[desired_cols]
#print(subset.head())

# Truco para obtener las columnas deseadas sin excribirlas todas
not_desired_cols = ['Account Length', 'Phone', 'Eve Charge', 'Day Calls']
all_column_list = data.columns.values.tolist()
sublist = [x for x in all_column_list if x not in not_desired_cols]
subset = data[sublist]
print(subset.head())