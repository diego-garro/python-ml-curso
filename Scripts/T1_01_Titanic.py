#-*- coding: utf-8 -*-

# Carga de datos a través de la función read_csv
import pandas as pd

main_path = '/home/diego/Documentos/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/titanic/titanic3.csv'
full_path = main_path + file_name

data = pd.read_csv(full_path)

print(data.head())

data2 = pd.read_csv(main_path + '/customer-churn-model/Customer Churn Model.txt')
print(data2.head())
print(data2.columns.values)

data3 = open(main_path + '/customer-churn-model/Customer Churn Model.txt', 'r')
cols = data3.readline().strip().split(',')
n_cols = len(cols)

counter = 0
main_dict = {}
for col in cols:
    main_dict[col] = []

for line in data3:
    values = line.strip().split(',')
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1

print('El data set tiene %d filas y %d columnas.'%(counter, n_cols))

df3 = pd.DataFrame(main_dict)
print(df3.head())

# Carga de datos desde una url

medals_url = 'http://winterolympicsmedals.com/medals.csv'
medals_data = pd.read_csv(medals_url)
print(medals_data.head())

import urllib3
http = urllib3.PoolManager()
r = http.request('GET', medals_url)
print(r.status)
print(r.data.decode('utf-8'))

f = open('medals.csv', 'w')
f.write(r.data.decode('utf-8'))
f.close()

import csv
csv_r = csv.reader('medals.csv')
print(csv_r)

for row in csv_r:
    print(row)

file_name = '/titanic/titanic3.xls'
titanic2 = pd.read_excel(main_path + file_name, 'titanic3')

titanic2.to_csv(main_path + '/titanic/titanic_custom.csv')