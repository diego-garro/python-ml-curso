
"""
 Regresión lineal en Python
 El paquete scikit-learn para la regresión lineal y la selección de rasgos
"""

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import pandas as pd
import numpy as np

data = pd.read_csv("../datasets/ads/Advertising.csv")
print(data.head())

feature_cols = ["TV", "Radio", "Newspaper"]

X = data[feature_cols]
Y = data["Sales"]

estimator = SVR(kernel="linear")
selector = RFE(estimator, 2, step=1)
selector = selector.fit(X, Y)

print("Variables seleccionadas: {}".format(selector.support_))
print("Ranking de las variables: {}".format(selector.ranking_))

from sklearn.linear_model import LinearRegression

X_pred = X[["TV", "Radio"]]
lm = LinearRegression()
lm.fit(X_pred, Y)

print("lm.intercept: {}\nlm.coef: {}\nlm.score: {}".format(lm.intercept_, lm.coef_, lm.score(X_pred, Y)))
