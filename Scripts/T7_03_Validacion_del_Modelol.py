
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

main_path = '/home/diego/Documentos/Developer/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/ads/Advertising.csv'
full_path = main_path + file_name

data = pd.read_csv(full_path)
print(data.head())

# Dividir el dataset en conjunto de entrenamiento y de test

a = np.random.randn(len(data))
#plt.hist(a)
#plt.show()

check = (a<0.8)
training = data[check]
testing = data[~check]

lm = smf.ols(formula="Sales~TV+Radio", data=training).fit()
print(lm.summary())

# Validación del modelo con el conjunto de testing

sales_pred = lm.predict(testing)
print(sales_pred)

SSD = np.sum((testing["Sales"] - sales_pred)**2)
RSE = np.sqrt(SSD / (len(testing) - 2 - 1))
sales_mean = np.mean(testing["Sales"])
error = RSE / sales_mean
print("SSD: {}\nRSE: {}\nerror: {}".format(SSD, RSE, error))


