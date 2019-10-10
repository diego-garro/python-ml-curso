# Modelo de regresión lineal
# Modelo con datos simulados
# y = a + b * x
# X: 100 valores distribuidos según una N(1.5, 2.5)
# Ye = 10 + 0.8 * X
# e estará distribuido según una N(0, 0.8)

import pandas as pd
import numpy as np

x = 1.5 + 2.5 * np.random.randn(100)
res = 0 + 0.8 * np.random.randn(100)
y_pred = 10 + 0.8 * x
y_act = 10 + 0.8 * x + res

x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()

data = pd.DataFrame(
    {
        'x' : x_list,
        'y_actual' : y_act_list,
        'y_prediccion' : y_pred_list
    }
)

print(data.head())

import matplotlib.pyplot as plt

y_mean = [np.mean(y_act) for i in range(1, len(x_list) + 1)]

plt.plot(x, y_pred)
plt.plot(x, y_act, 'ro')
plt.plot(x, y_mean, 'g')
plt.title("Valor Actual vs Predicción")
#plt.show()

data["SSR"] = (data["y_prediccion"] - np.mean(y_act))**2
data["SSD"] = (data["y_prediccion"] - data["y_actual"])**2
data["SST"] = (data["y_actual"] - np.mean(y_act))**2

SSR = sum(data["SSR"])
SSD = sum(data["SSD"])
SST = sum(data["SST"])

print("SSR: {}".format(SSR))
print("SSD: {}".format(SSD))
print("SST: {}".format(SST))

print(data.head())

print("SSR + SSD: {}".format(SSR + SSD))
print("R2 = SSR / SST: {}".format(SSR / SST))