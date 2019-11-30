import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial-regression.csv",sep=";")
y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)
plt.scatter(x,y)
plt.ylabel("hiz")
plt.xlabel("fiyat")
plt.show()

lr=LinearRegression()
lr.fit(x,y)

y_head=lr.predict(x)
plt.plot(x,y_head,color="blue")

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression=PolynomialFeatures(degree=4)
x_polynomial=polynomial_regression.fit_transform(x)
linear_regression2=LinearRegression()
linear_regression2.fit(x_polynomial,y)

y_head2=linear_regression2.predict(x_polynomial)
plt.plot(x,y_head2,color="green",label="poly")
plt.show()