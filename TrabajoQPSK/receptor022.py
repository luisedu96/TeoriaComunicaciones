# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:46:50 2020

@author: luise
"""

import numpy as np
import matplotlib.pyplot as plt
from clases import senal, codificador
import soundfile as sf
import sounddevice as sd
from scipy.signal import correlate

plt.close('all')
y=senal.cargar_sonido(archivo='senal de evaluacion.wav', nombre='señal recibida')
y.dibujar()
tmin=1.9
Yport=y.recortar(tmin=tmin, tmax=2.85)
port=senal.PLL(y=Yport.x, fs=Yport.fs, fc=2000, dibujar=True)
ph=senal(x=port.phase, fs=port.fs, nombre='Fase')
ph2=ph.mediaMovil(L=100, dibujar=True)

ta=0.025
tb=0.915
pha=ph2.getX(ta)
phb=ph2.getX(tb)
m=(phb-pha)/(tb-ta)
b=pha-m*(ta+tmin)
t=y.getT()
portadora=np.cos(2*np.pi*2000*t+(m*t+b))
portadoraI=-np.sin(2*np.pi*2000*t+(m*t+b))

#Recuperación de las dos partes de la señal en caso de que haya cuadratura
yR = senal(x=y.x*portadora, fs=y.fs, nombre='Señal recibida, parte real')
yI = senal(x=y.x*portadoraI, fs=y.fs, nombre='Señal recibida, parte imaginaria')
yR = yR.filtrar_pasabajo(fc=1000)
yI = yI.filtrar_pasabajo(fc=1000)

#Busqueda del inicio de las muestras
cod=codificador()
fc=2000
fs=44100
x=cod.seq1
x=codificador.bits2Polar(x)
K=100
B=44100/K
xn=senal(x=x, fs=B, nombre='Secuencia a transmitir')
xn.modularPulsos(K=K, tipo='RC', graficar=True, alfa=0.3, num=10)#igual que el transmisor
xn.x=xn.x/np.max(np.abs(xn.x))
xn.dibujar()

co=correlate(yR.x, xn.x, mode='valid')
ind=np.argmax(co)
plt.figure()
plt.plot(co)
plt.plot(ind,co[ind],'or')

samples=np.arange(ind+xn.lo, len(y.x), K)

ynR=np.array([yR.x[int(i)] for i in samples])
ynI=np.array([yI.x[int(i)] for i in samples])

ynR[ynR>0]=1
ynR[ynR<0]=-1
ynR=(ynR+1)/2
ynR=ynR[280:]

ynI[ynI>0]=1
ynI[ynI<0]=-1
ynI=(ynI+1)/2
ynI=ynI[280:]

yn=cod.agregarRI(ynR,ynI)
cad=cod.bits2Cad(yn)
print(cad)
