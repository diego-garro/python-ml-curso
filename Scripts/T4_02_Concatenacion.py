#-*- coding: utf-8 -*-

import pandas as pd

main_path = '/home/diego/Documentos/Developer/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/wine/winequality-red.csv'
full_path = main_path + file_name

red_wine = pd.read_csv(full_path, sep=';')
print(red_wine.head())
print(red_wine.columns.values)

file_name = '/wine/winequality-white.csv'
full_path = main_path + file_name
white_wine = pd.read_csv(full_path, sep=';')
print(white_wine.head())
print(white_wine.columns.values)

wine_data = pd.concat([red_wine, white_wine], axis=0)
print(wine_data.shape)