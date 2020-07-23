# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:25:39 2020

@author: luise
"""

import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, ifft
from scipy.fftpack import fftshift
from scipy.signal import correlate, resample, butter, lfilter, convolve
from commpy.filters import rcosfilter, rrcosfilter

class senal:
    def __init__(self, x=[], fs=1, nombre=''):
        self.x=x
        #self.x=self.x/np.max(np.abs(self.x)) #Normalización del audio grabado
        self.nombre=nombre
        self.fs=fs

    def getT(self):
        L=len(self.x)
        Tmax=L/self.fs
        t=np.arange(0, Tmax*1.1, 1/self.fs)
        t=t[0:L]
        return t
    
    #Grabar un sonido
    def grabar_sonido(self, duracion=5, fs=44100):
        self.x = sd.rec(int(duracion * fs), samplerate=fs, channels=1)[:,0]
        sd.wait()
        self.x=self.x/np.max(np.abs(self.x)) #Normalización del audio grabado
        self.fs=fs
        
    def reproducir(self):
        sd.playrec(self.x, samplerate=self.fs, channels=1)

    def dibujar(self, crear_figura=True):
        t=self.getT()
        if(crear_figura):
            plt.figure()
        plt.plot(t,self.x,'-b')
        plt.xlabel('Tiempo Seg')
        plt.ylabel('Señal')
        plt.title(self.nombre)

    def espectro(self, crear_figura=True):
        self.X=fft(self.x)
        L=len(self.x)
        self.df=self.fs/L
        f=np.linspace(-self.fs/2, self.fs/2-self.df, L)
        if(crear_figura):
            plt.figure()
        plt.plot(f, np.abs(fftshift(self.X)))
        plt.xlabel('Frecuencia Hz')
        plt.ylabel('Transformada de Fourier de la señal')
        plt.title('Espectro de '+self.nombre)
        
    def espectro_autocorrelacion(self, crear_figura=True):
        self.autocorrelacion(crear_figura)
        self.X=fft(self.xcorr)
        self.df=self.fs/len(self.X)
        f=np.linspace(-self.fs/2, self.fs/2-self.df, len(self.X))
        if(crear_figura):
            plt.figure()
        plt.plot(f, np.abs(fftshift(self.X)))
        plt.xlabel('Frecuencia Hz')
        plt.ylabel('Transformada de Fourier de la señal')
        plt.title('Espectro de '+self.nombre)
        
    def autocorrelacion(self, crear_figura=True):
        self.xcorr=correlate(self.x, self.x,mode='full',method='auto')
        self.xcorr=self.xcorr/np.max(np.abs(self.xcorr))
        if(crear_figura):
            plt.figure()
        L=len(self.x)
        Tmax=L/self.fs
        tau=np.linspace(-Tmax, Tmax, len(self.xcorr))
        plt.plot(tau,self.xcorr)
        plt.xlabel('Tau')
        plt.ylabel('Autocorrelación de '+self.nombre)

    def filtrar_pasabanda(self, fc, orden=4):
        Fmax = 0.5 * self.fs
        low = (fc-150) / Fmax
        high = (fc+150) / Fmax
        b, a = butter(orden, [low, high], btype='bandpass', analog=False)
        y = lfilter(b, a, self.x)
        return senal(y, self.fs, self.nombre+' filtrada '+str(fc))
    
    def filtrar_pasabajo(self, fc, orden=4):
        Fmax = 0.5 * self.fs
        fc_normalizada = fc / Fmax
        b, a = butter(orden, fc_normalizada, btype='lowpass', analog=False)
        y = lfilter(b, a, self.x)
        return senal(y, self.fs, self.nombre+' filtrada')

    def sobre_muestrear(self, K):
        y=resample(self.x,K*len(self.x))
        return senal(y, K*self.fs, self.nombre+' sobremuestreada')
    
    def derivar(self):
        y=np.gradient(self.x)
        return senal(y, self.fs, self.nombre+' derrivada')

    def modularPulsos(self, K=10, tipo='square', graficar=False, alfa=0.3, num=4): 
        self.K=K
        self.xn=self.x
        self.zero_pad()
        v, lo=self.pulso(K=K, alfa=alfa, num=num, tipo=tipo)
        self.lo=lo
        self.tipo=tipo
        y=convolve(self.x, v, mode='full')
        self.x=y
        if(graficar):
            self.dibujar()
            self.dibujar_simbolos()
        
    def zero_pad(self):
        r = np.zeros((len(self.x), self.K))
        r[:,0]=self.x
        r=np.ravel(r.reshape(1,len(self.x)*self.K))
        self.x=r
        self.fs=self.fs*self.K
        
    def pulso(self, K=10, alfa=0.3, num=4, tipo='square'):
        if(tipo=='square'):
            lo=np.ceil(K/2)-1
            return np.ones(K), lo
        if(tipo=='sinc'):
            n=np.arange(-K*num, K*num+1, 1)
            n[n==0]=-1000
            tmp=np.sin(np.pi*n/K)/(np.pi*n/K)
            lo=np.ceil(len(tmp)/2)-1
            n[n==-1000]=0
            tmp[n==0]=1
            return tmp, lo
        if(tipo=='squareRZ'):
            lo=np.ceil(K/2)-1
            n=np.linspace(0,K-1,K)
            x=np.ones_like(n)
            x[n>=K/2]=0
            return x, lo
        if(tipo=='Manchester'):
            lo=np.ceil(K/2)-1
            n=np.linspace(0,K-1,K)
            x=np.ones_like(n)
            x[n>=K/2]=-1
            return x, lo
        if(tipo=='RC'):
            n,v = rcosfilter(num*K, alfa, 1, K)
            lo=np.ceil(len(v)/2)
            return v, lo
        if(tipo=='rootRC'):
            n,v = rrcosfilter(num*K, alfa, 1, K)
            lo=np.ceil(len(v)/2)
            return v, lo
    
    def dibujar_simbolos(self):
        n=np.array(range(len(self.xn)))
        n=n*self.K+self.lo
        plt.plot(n/self.fs, self.xn,'xr')

    def modularAM(self, fc):
        dc=np.max(np.abs(self.x))
        t=self.getT()
        self.x=(self.x+dc)*np.sin(2*np.pi*fc*t)
    def demodularAM(self, fc):
        t=self.getT()
        self.x=self.x*np.sin(2*np.pi*fc*t)
        #filtrar la señal
        y=self.filtrar_pasabajo(fc=fc, orden=4)
        self.x=y.x
        self.x=self.x-np.mean(self.x)
        
    def convolucion(self, y):
        self.x=convolve(self.x, y, mode='full')
        