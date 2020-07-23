# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:13:42 2020

@author: Roberto H
"""

import numpy as np
import matplotlib.pyplot as plt
from clases import senal, codificador
import soundfile as sf
import sounddevice as sd

cad='Buenos días apreciados estudiantes. Esta es una señal en cuadatura para su recepcion'
cod=codificador()
bits=cod.cad2Bin(cad)
bitsR, bitsI=cod.separarRI(bits)

fc=2000
fs=44100
Lport=int(500*fs/fc)
xR=np.concatenate((cod.seq1,bitsR))
xR=codificador.bits2Polar(xR)
xI=codificador.bits2Polar(bitsI)
xI=np.concatenate((np.zeros_like(cod.seq1),xI))
plt.close('all')
K=100
B=fs/K
plt.close('all')
xnR=senal(x=xR, fs=B, nombre='Secuencia a transmitir parte Real')
xnR.modularPulsos(K=K, tipo='RC', graficar=True, alfa=0.3, num=10)
xnR.x=xnR.x/np.max(np.abs(xnR.x))
xnR.x=np.concatenate((np.ones(Lport), xnR.x))
xnR.dibujar()
xnI=senal(x=xI, fs=B, nombre='Secuencia a transmitir parte Imaginaria')
xnI.modularPulsos(K=K, tipo='RC', graficar=True, alfa=0.3, num=10)
xnI.x=xnI.x/np.max(np.abs(xnI.x))
xnI.x=np.concatenate((np.zeros(Lport), xnI.x))
xnI.dibujar()
y=senal.modularAMCuadratura(fc=fc, xnR=xnR, xnI=xnI)
y.x=y.x/np.max(np.abs(y.x))
y.reproducir()
y.dibujar()
y.espectro()
sf.write('salida.wav', y.x, int(y.fs))

