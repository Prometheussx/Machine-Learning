# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 02:34:15 2023

@author: erdem
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:05:02 2023

@author: erdem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv",sep=";",header= None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100,random_state=42)#n_estimator kaç adet ağaç kullanılacak demektir random_state ise n adet sample data seçilirken kaç adet seçecğini random yapıyor genelde 42 yapılır bu ıd atar yani aynı yapıyı birkaç kez fit ederken farklı sonuç gelmesin diye ilk yapılan sonuc 42.ıd ye atanır ve değişmedikçe o kullanılır
   
rf.fit(x,y)

y_head = rf.predict(x)

#%% metrics
#sklearn ile r2_score unu ölçtüğümüz kod kalıbıdır prediction doğruluğunu ölçer
from sklearn.metrics import r2_score

print("r_score: ",r2_score(y,y_head))

#%%Plt

plt.scatter(x,y,color="red")
plt.plot(x,y_head,color="green")
plt.xlabel("Tribün Level")
plt.ylabel("Fiyat")

plt.show()
    