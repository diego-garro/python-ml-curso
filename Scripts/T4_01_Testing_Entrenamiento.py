#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

main_path = '/home/diego/Documentos/Developer/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/customer-churn-model/Customer Churn Model.txt'
full_path = main_path + file_name

data = pd.read_csv(full_path)
print(len(data))

# Método de particionamiento de los datos usando una distribución de probabilidades
# En este caso se usa 80% / 20%, si se quiere otro particionamiento solo se cambia
# el valor en 'check = (a<valor)'
a = np.random.randn(len(data))
check = (a<0.8)

data_train = data[check]
data_test = data[~check]
print(len(data_train), len(data_test))

#plt.hist(check)
#plt.show()

# Usando la librería sklearn con la función train_test_split()
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)
print(len(train), len(test))

# Usando una función shuffle
from sklearn.utils import shuffle

data = shuffle(data)
#print(data.head())

cut_id = int(0.8 * len(data))
train = data[:cut_id]
test = data[cut_id:]
print(len(train), len(test))
