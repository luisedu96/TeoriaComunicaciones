# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:13:42 2020

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


"""PROCESO DE TRANSMISION"""
f1=2400
f2=1200
tsim=0.1#tiempo de simbolo
tmues=1/44100#tiempo de muestreo

t=np.arange(0,tsim,tmues)#t [muestras]
v1=np.cos(2*np.pi*f1*t)#muestreo de la señal coseno de 2400hz
v2=np.cos(2*np.pi*f2*t)#muestreo de la señal coseno de 1200hz

simbolos=[1,0,0,1,1,0,1,0,1,1]#simbolos a enviar
xt=np.array([])#almacena las muestras de cos@1200 y cos@2400 segun sea el caso de un 0 o 1
for xi in simbolos:
    if(xi==0):
        xt=np.concatenate((xt,v1))
    else:
        xt=np.concatenate((xt,v2))
        
ruido =np.random.randn(int(44100*len(simbolos)*tsim))#duracion del ruido agregado
ruido=ruido/np.max(np.abs(ruido))#normalizacion del ruido
xt=xt+ruido#se le adiciona ruido a la señal

xseñal=senal(x=xt, fs=44100, nombre='Señal modelada en fsk')#se crea la señal a enviar
xseñal.dibujar()
xseñal.espectro()
sf.write(data=xseñal.x, file='salida.wav', samplerate=44100)
xseñal.reproducir()


"""PROCESO DE RECEPCION"""
muestra_sim=(1/tmues)*tsim#cantidad de muestras por simbolo
f_0=100
len_vent=int(44100/f_0)#longitud de ventana(cantidad de datos)
cantidad_vent=int(len(xseñal.x)/len_vent)#numero de ventanas

ventanas=[]#se almacena todas las ventanas

#separacion de la señal por ventanas
for i in range(1,cantidad_vent+1):
    vent=np.array([])
    if(i == cantidad_vent):
        vent=xseñal.x[len_vent*(i-1):]
    else:
        vent=xseñal.x[len_vent*(i-1):len_vent*i]
    ventana=senal(x=vent, fs=44100, nombre='ventana '+str(i))
    #ventana.espectro()
    ventanas.append(ventana)
    

sim_ven=[]#se almacena los simbolos que correponde a cada ventana segun a su frecuencia
for s in range(0, len(ventanas)):
    senal_fourierX=np.abs(fft(ventanas[s].x))#transfromada de fourier
    w=np.argmax(senal_fourierX)*f_0#se extrae el argumento maximo y se multiplica por f0 para saber la frecuencia en cada ventana
    distancia1=(1200-w)**2
    distancia2=(2400-w)**2
    if (distancia1<distancia2):
        sim_ven.append(1)
    else:
        sim_ven.append(0)

cant_ven_sim=int(muestra_sim/len_vent)#cantidad de ventanas por simbolo

#detector de simbolos
detector=[]
for i in range(cant_ven_sim):
    detector.append(1)
    
correlacion=correlate(sim_ven, detector, mode='valid',method='auto')#correlacion para detectar los simbolos y minimizar los errores

simbolo_recuperados=[]
#se recorre el array de la correlacion cada 10 elementos(en este caso) para saber los picos mas altos y bajos, que en cada
#10 elementos el detector se encuentra sobre el simbolo completo en la correlacion.
for k in range(0,len(correlacion),cant_ven_sim):
    if (correlacion[k]>(len(detector)/2)):
        simbolo_recuperados.append(1)
    else:
        simbolo_recuperados.append(0)





















