
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

main_path = '/home/diego/Documentos/Developer/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/ads/Advertising.csv'
full_path = main_path + file_name

data = pd.read_csv(full_path)
print(data.head())

# Dividir el dataset en conjunto de entrenamiento y de test
