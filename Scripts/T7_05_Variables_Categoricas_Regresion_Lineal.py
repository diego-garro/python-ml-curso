
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../datasets/ecom-expense/Ecom Expense.csv")
print(df.head())

dummy_gender = pd.get_dummies(df["Gender"], prefix="Gender")
dummy_city_tier = pd.get_dummies(df["City Tier"], prefix="City")

print(dummy_gender.head())
print(dummy_city_tier.head())

column_names = df.columns.values.tolist()
df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
print(df_new.head())

df_new = df_new[column_names].join(dummy_city_tier)
print(df_new.head())

feature_cols = ["Monthly Income", "Transaction Time", "Gender_Female",
			    "Gender_Male", "City_Tier 1", "City_Tier 2", "City_Tier 3",
			    "Record"]

X = df_new[feature_cols]
Y = df_new["Total Spend"]

lm = LinearRegression()
lm.fit(X, Y)

print("lm.intercept_ : {}\nlm.coef_ : {}".format(lm.intercept_, lm.coef_))
print(list(zip(feature_cols, lm.coef_)))
print("lm.score: {}".format(lm.score(X, Y)))


