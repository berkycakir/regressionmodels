import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df=pd.read_csv("multiple-linear-regression-dataset.csv",sep=";")
x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)
print("b0: ",multiple_linear_regression.intercept_)
print(multiple_linear_regression.predict(np.array([[10,35],[5,35]])))