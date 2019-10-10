#-*- coding: utf-8 -*-
# Resumen de los datos, dimensiones y estructuras

import pandas as pd

main_path = '/home/diego/Documentos/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/titanic/titanic3.csv'
full_path = main_path + file_name

data = pd.read_csv(full_path)

# Como ver el contenido de las primeras lineas del dataset
print(data.head(10))

# La forma del dataset
print('La forma del dataset es: {}'.format(data.shape))

# Ver los ultimos datos
print(data.tail(8))


# Los valores de las cabeceras del dataset
print(data.columns)

# Vamos a hacer un resumen de los estadísticos básicos de las variables numéricas

print('---La descripción de los datos---')
print(data.describe())

print('---El tipo de cada variable---')
print(data.dtypes)

# MISSING VALUES

print('Encuentra los valores nulos de una variable, de modo que si es nulo saldrá como True')
print(pd.isnull(data['body']).values)

print('Pregunta la suma de valores nulos en la variable')
print(pd.isnull(data['body']).values.ravel().sum())

print('Pregunta la suma de valores no nulos en la variable')
print(pd.notnull(data['body']).values.ravel().sum())

"""
Los valores que faltan en un dataset pueden venir por dos razones:
* Extracción de los datos
* Recolección de los datos
"""

#  Borrar los valores que faltan

# Borra las filas que tengan todas sus columnas como NaN
#data2 = data.dropna(axis=0, how='all')
#print(data2.head())

#  Cómputo de los valores faltantes
data3 = data
data3['body'] = data3['body'].fillna(0)
data3['home.dest'] = data3['home.dest'].fillna('Desconocido')
print(data3.head(20))

# Crear variables dummy
def createDummies(df, var_name):
    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = df.drop(var_name, axis=1)
    df = pd.concat([df, dummy], axis=1)
    return df

data4 = createDummies(data3, 'sex')
print(data4.head())