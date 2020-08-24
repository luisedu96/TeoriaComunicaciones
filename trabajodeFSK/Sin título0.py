# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:11:06 2020

@author: luise
"""

import matplotlib.pyplot as plt
import numpy as np

t=np.arange(0,3600.2,0.2)
tejem=np.arange(0,600,0.2)

act=np.random.normal(1.3,0.004,6)  
dor=np.random.normal(0.1,0.004,len(tejem)-6)  

actlinea=np.array([])
dorlinea=np.array([])
for i in range(18001):
    actlinea=np.append(actlinea,1.3)

for i in range(18001):
    dorlinea=np.append(dorlinea,0.1)



I=np.array([])
I=np.append(I,act)
I=np.append(I,dor)

energy=np.array([])

for i in range(int(6)):
    energy=np.append(energy,I)
 
energy=np.append(energy,0.1)


fig,ax = plt.subplots()
ax.grid()
ax.fill_between(tejem,I,color='orange')
ax.axis([0,15,0,1.5])
ax.plot(t,actlinea,color='red',label='$1.3μA$')
ax.plot(t,dorlinea,color='green', label='$0.1μA$')
plt.legend(loc="upper left") 
ax.set_xlabel('$Tiempo [segundos]$', fontsize=14)
ax.set_ylabel('$Consumo [μA]$', fontsize=14)
ax.set_title('Consumo del sensor HDC1080 en 15 segundos')
plt.show()