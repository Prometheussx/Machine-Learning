# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 00:24:46 2023

@author: erdem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df =pd.read_csv("data.csv",sep=";",header=None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

plt.scatter(x,y)
plt.show()
