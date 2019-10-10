#-*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


main_path = '/home/diego/Documentos/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/customer-churn-model/Customer Churn Model.txt'
full_path = main_path + file_name

data = pd.read_csv(full_path)
print(data.head())

# Visualizacion de datos ScatterPlots
#fig = data.plot(kind='scatter', x='Day Mins', y='Day Charge')
#plt.show()

# Usando la librería matplotlib

# Scatter plot
#fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
#data.plot(kind='scatter', x='Day Mins', y='Day Charge', ax=axs[0][0])
#data.plot(kind='scatter', x='Night Mins', y='Night Charge', ax=axs[0][1])
#data.plot(kind='scatter', x='Day Calls', y='Day Charge', ax=axs[1][0])
#data.plot(kind='scatter', x='Night Calls', y='Night Charge', ax=axs[1][1])
#plt.show()

# Histograma
#plt.hist(data['Day Calls'], bins=10)
#plt.show()

# Boxplot, diagrama de caja y bigotes
plt.boxplot(data['Day Calls'])
plt.ylabel('Número de llamadas diarias')
plt.title('Boxplot de las llamadas diarias')
plt.show()

