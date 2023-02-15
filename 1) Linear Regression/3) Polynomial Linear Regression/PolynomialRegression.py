# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:51:11 2023

@author: erdem
"""
#Polynomil regression sonlu sonu sabitleşen veriler için kullanılır üssel fonksiyon olarak tanımlanabilir 
import pandas as pd
import matplotlib.pyplot as plt 
df = pd.read_csv("polynomialregression.csv",sep=";")

x = df.araba_fiyat.values.reshape(-1,1) #bağımsız değişken 
#values series type'nı Array of int64 çevirir reshape ise (15,)size'ı (15,1)size yapar
y = df.araba_max_hiz.values.reshape(-1,1) #bağımlı değişken

plt.scatter(x,y)
plt.ylabel("Arabanın Max Hızı")
plt.xlabel("Arabanın Fiyatı")
plt.show()

#linear regression & Multiple linear regression
#Linear: y=b0+b1*x
#Multiple: y=b0+b1*x1+b2*x2...bn*xn

#%% Linear Regression
#from sklearn.linear_model import LinearRegression

#lr = LinearRegression()

#lr.fit(x,y)
#y_head
#y_head=lr.predict(x)

#plt.plot(x,y_head,color="red")
#plt.show()
        


#%%Polynomail Linear Regression (y=b0+b1*x1+b2*x2^2...bn*xn^n)
#degree olan derece artıkça polynom karmaşıklaşır ve hata oranı azalır
#b0 da gizli bir x^0 vardır bu sebeple ne olursa olsun 1 dir
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polynomial_regression = PolynomialFeatures(degree =2)  #degree polinom derecesini belirtir kaç adet x var ise okadar eklenir

x_polynomial = polynomial_regression.fit_transform(x) #fit_transform x^2 kullan ve yeni x'i x^2 yap
#elimizdeki gerçek y değerlerine göre ve x değerlerine göre bir eğitim fit ettik ve bu eğitilmiş yapı ile yeni değeler tahmin etmek için x değerleri veriyoruz
#çizgi oluşturacağız linear yapıyla aynı olduğu için sadece x üssel hale alarak tekrar linear regression uyguladık
linear_regression = LinearRegression()
linear_regression.fit(x_polynomial,y)

y_head=linear_regression.predict(x_polynomial)

plt.plot(x,y_head,color="green",label="poly")
plt.lagend()
plt.show()








