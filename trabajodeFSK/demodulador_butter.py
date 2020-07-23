# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 04:42:08 2020

@author: luise
"""

import numpy as np
import matplotlib.pyplot as plt
from clases import senal
import soundfile as sf
import sounddevice as sd

from scipy import fft, ifft
from scipy.fftpack import fftshift
from scipy.signal import correlate, resample, butter, lfilter, convolve

"""PROCESO DE TRANSMISIÓN"""
f1=2400
f2=1200
tsim=0.1
tmues=1/44100

t=np.arange(0,tsim,tmues)
v1=np.cos(2*np.pi*f1*t)
v2=np.cos(2*np.pi*f2*t)
#plt.figure(1)
#plt.plot(t,v1)
#plt.figure(2)
#plt.plot(t,v2)

x=[1,0,0,1,1,0,1,0,1,1]
xt=np.array([])
for xi in x:
    if(xi==0):
        xt=np.concatenate((xt,v1))
    else:
        xt=np.concatenate((xt,v2))
        
ruido =np.random.randn(int(44100*len(x)*tsim))#duracion del ruido agregado
ruido=ruido/np.max(np.abs(ruido))#normalizar el ruido
xt=xt+ruido#se le agrega ruido a la señal

xseñal=senal(x=xt, fs=44100, nombre='Señal modelada en fsk')
xseñal.dibujar()
xseñal.espectro()
sf.write(data=xseñal.x, file='salida.wav', samplerate=44100)
xseñal.reproducir()

"""PROCESO DE RECEPCIÓN"""
xseñal_v2=xseñal.filtrar_pasabanda(1200)#proceso de flitrado Passband
xseñal_v2.dibujar()
xseñal_v2.espectro()

xseñal_v1=xseñal.filtrar_pasabanda(2400)#proceso de flitrado Passband
xseñal_v1.dibujar()
xseñal_v1.espectro()

len_ventana=294#longitud de la ventana
num_ventanas=int(len(xseñal.x)/len_ventana)#numero de ventanas

#este método me separa la señal filtrada en ventanas y a la vez halla la energia para cada ventana
def energia_ventanas(num,señal,long):
    energias=[]
    for i in range(1,num+1):
        ventana_x=[]
        energia=0
        if(i == num):
            ventana_x=señal.x[long*(i-1):]
        else:
            ventana_x=señal.x[long*(i-1):long*i]
        for m in range (0, len(ventana_x)):
            energia=ventana_x[m]**2 + energia#se calcula la energia en cada ventana
        energias.append(energia)
    return energias


"""Para la señal filtrada en 1200hz"""
#este método me separa la señal filtrada en ventanas y a la vez halla la energia para cada ventana
energiasv2=energia_ventanas(num_ventanas, xseñal_v2, len_ventana)

"""Para la señal filtrada en 2400hz"""
energiasv1=energia_ventanas(num_ventanas, xseñal_v1, len_ventana)

muestra_sim=(1/tmues)*tsim#cantidad de muestras por simbolo
num_ventanas_sim = int(muestra_sim/len_ventana)#cantidad de ventanas por simbolo

#se crea un detector unicamente con 1
detector=[]
for l in range(num_ventanas_sim):
    detector.append(1)

#correlacion para detectar los simbolos segun su energia teniendo en cuenta la cantidad de ventanas por simbolo
correlacionv2=correlate(energiasv2, detector, mode='valid',method='auto')
correlacionv1=correlate(energiasv1, detector, mode='valid',method='auto')

#se compara las correlaciones con un paso de 15 elementos, para este caso, dentro de cada array. y asi detectar 
#que simbolo es o 1 o un 0, segun sea los picos mas altos.
simbolos_recuperados=[]
for paso in range(0, len(correlacionv2), num_ventanas_sim):
    if (correlacionv1[paso] > correlacionv2[paso]):
        simbolos_recuperados.append(0)
    else:
        simbolos_recuperados.append(1)
        








