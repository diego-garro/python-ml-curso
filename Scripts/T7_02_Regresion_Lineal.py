
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

main_path = '/home/diego/Documentos/Developer/Python/Scripts/MachineLearning/Curso_JG_Gomila/datasets'
file_name = '/ads/Advertising.csv'
full_path = main_path + file_name

data = pd.read_csv(full_path)
print(data.head())

lm = smf.ols(formula="Sales~TV", data=data).fit()
print(lm.summary())

sales_pred = lm.predict(pd.DataFrame(data["TV"]))
print(sales_pred)

#data.plot(kind="scatter", x="TV", y="Sales")
#plt.plot(pd.DataFrame(data["TV"]), sales_pred, c="red", linewidth=2)
#plt.show()

data["Sales_Pred"] = 7.032594 + 0.047537 * data["TV"]
data["RSE"] = (data["Sales"] - data["Sales_Pred"])**2
SSD = sum(data["RSE"])
RSE = np.sqrt(SSD / (len(data)-2))
sales_m = np.mean(data["Sales"])
error = RSE / sales_m
print("""
RSE: {}
SSD: {}
sales_m: {}
error: {}""".format(RSE, SSD, sales_m, error))

#plt.hist((data["Sales"] - data["Sales_Pred"]))
#plt.show()

# Regresión lineal múltiple
# Se usa el paquete statsmodel para esta regresión

lm2 = smf.ols(formula="Sales~TV+Newspaper", data=data).fit()
print(lm2.params)
print(lm2.rsquared, lm2.rsquared_adj)

sales_pred = lm2.predict(data[["TV", "Newspaper"]])
print("Sales:\n{}".format(sales_pred))

SSD = sum((data["Sales"]-sales_pred)**2)
print("SSD: {}".format(SSD))

RSE = np.sqrt(SSD / (len(data)-2-1))
print("RSE: {}".format(RSE))

error = RSE / sales_m
print("Error: {}".format(error))

#Resumen
print(lm2.summary())