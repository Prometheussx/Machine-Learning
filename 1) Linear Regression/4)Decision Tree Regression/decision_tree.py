# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:36:55 2023

@author: erdem
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data.csv",sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%  decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()   # random sate = 0
tree_reg.fit(x,y)
tree_reg.predict([[5.5]])
x_=np.arange(min(x),max(x),0.01).reshape(-1,1) #deneme amaçlı yeni veri oluşturduk en azx den en fazla x e 0.01 boşlukla veri yaptık
y_head =tree_reg.predict(x_)
#%% visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Tribün Seviyesi")
plt.ylabel("Tribün Fiyatı")
plt.show()
