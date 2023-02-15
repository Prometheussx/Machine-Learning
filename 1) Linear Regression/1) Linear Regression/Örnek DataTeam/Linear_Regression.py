# -*- coding: utf-8 -*-
"""
Spyder Editor
 
Machine Learning 
owned:Erdem Taha Sokullu

This is a temporary script file.
"""
#import Library
import pandas as pd
import matplotlib.pyplot as plt

#import Data
df = pd.read_csv("linear_regression_dataset.csv", sep = ";")

#Plot Data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")


#Sklearn Library
from sklearn.linear_model import LinearRegression

#lineaer regression model
linear_reg = LinearRegression()
#NOT 1 : Bu yapılar pandas ama sklearnde numpy daha işlevli olduğu için numpy deönüştürücez
#bunun içinde datanın yanına .values ekliyoruz
#NOT 2 : Pandas shape (14,) ise bunu sklearn kabul etmiyor bunun yeerine reshape(-1,1) ekliyerek (14,1) haline getirmeliyiz
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

#liear reg fit
linear_reg.fit(x,y)

import numpy as np
#linear_reg predict
b0=linear_reg.predict([[0]]) #bu bize y eksenini kestiği noktayı verir (intercept) #1163

b0_ = linear_reg.intercept_ # bu komutta kesişim noktasını verir #1163

b1 = linear_reg.coef_ #b1 verir yani eğimi  verir (slope) #1138

#maas = 1663 +  1138*deneyim (y = b0 + b1*x)
deneme = 1663.89519747 + 1138.34819698*10
print(deneme)    

print(linear_reg.predict([[11]]))
#predict komutu istenilen x sayısının karşılık cevabını verir üsteki denklemi kullanarak

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) #istenilen deneyimlistesi

plt.scatter(x,y, color="green") # noktalı arayüz oluşturma komutu
plt.show()

y_head = linear_reg.predict(array) #istenilen deneyim listesine göre karşılık gelen maaş değerlerini belirler
plt.plot(array, y_head,color = "red") # bu ise elimizdeki maaş ve deneyim denk gelişlerine göre bir line çizer












