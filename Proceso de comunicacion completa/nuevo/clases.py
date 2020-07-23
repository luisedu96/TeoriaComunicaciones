# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:25:39 2020

@author: Roberto H
"""

import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, ifft
from scipy.fftpack import fftshift
from scipy.signal import correlate, resample, butter, lfilter, convolve
from commpy.filters import rcosfilter, rrcosfilter
from commpy.channels import awgn

class senal:
    def __init__(self, x=[], fs=1, nombre=''):
        self.x=x
        #self.x=self.x/np.max(np.abs(self.x)) #Normalización del audio grabado
        self.nombre=nombre
        self.fs=fs
    
    #Método estático para grabar un sonido
    def grabar_sonido(duracion=5, fs=44100, nombre='Señal grabada'):
        x = sd.rec(int(duracion * fs), samplerate=fs, channels=1)[:,0]
        sd.wait()
        return senal(x=x/np.max(np.abs(x), fs=fs, nombre=nombre)) #Normalización del audio grabado
    
    #Método estático para cargar un sonido desde un archivo
    def cargar_sonido(archivo='', nombre='Señal leida de archivo'):
        x, fs = sf.read(archivo)
        x=x/np.max(np.abs(x))
        return senal(x=x, fs=fs, nombre=nombre)
    
    def escribir_sonido(self, archivo=''):
        sf.write(archivo, self.x, int(self.fs))

    def recortar(self, tmin, tmax):
        a=int(tmin*self.fs)
        a=max(a,0)
        b=int(tmax*self.fs)
        b=min(len(self.x)-1, b)
        x=self.x[a:b]
        return senal(x=x, fs=self.fs, nombre=self.nombre)
    
    def getT(self):
        L=len(self.x)
        Tmax=L/self.fs
        t=np.arange(0, Tmax*1.1, 1/self.fs)
        t=t[0:L]
        #t=np.array(range(L))
        return t
    
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

    def filtrar_pasabajo(self, fc, orden=4):
        Fmax = 0.5 * self.fs
        fc_normalizada = fc / Fmax
        b, a = butter(orden, fc_normalizada, btype='low', analog=False)
        y = lfilter(b, a, self.x)
        return senal(y, self.fs, self.nombre+' filtrada')

    def filtrar_pasabanda(self, fc, orden=4):
        Fmax = 0.5 * self.fs
        fc_normalizada = fc / Fmax
        b, a = butter(orden, fc_normalizada, btype='bandpass', analog=False)
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
        #plt.plot(n, self.xn,'xr')

    def modularAM(self, fc):
        dc=np.max(np.abs(self.x))*0
        t=self.getT()
        self.x=(self.x+dc)*np.cos(2*np.pi*fc*t)
    
    def modularAMsinPortadora(self, fc):
        t=self.getT()
        self.x=self.x*np.cos(2*np.pi*fc*t)
    
    def modularAMCuadratura(fc, xnR, xnI):
        t=xnR.getT()
        xR=xnR.x*np.cos(2*np.pi*fc*t)
        xI=xnI.x*-np.sin(2*np.pi*fc*t)
        y=senal(x=xR+xI, fs=xnR.fs, nombre='Señal en cuadratura')
        return y
    
    def demodularAM(self, fc):
        t=self.getT()
        self.x=self.x*np.cos(2*np.pi*fc*t)-1j*self.x*np.sin(2*np.pi*fc*t)
        #filtrar la señal
        #y=self.filtrar_pasabajo(fc=fc, orden=4)
        #self.x=y.x
        #self.x=self.x-np.mean(self.x)
        
    def convolucion(self, y):
        self.x=convolve(self.x, y, mode='full')
        
    def getBW(self, umbral=0.9):
        E=np.sum(self.x*self.x)
        XW=fft(self.x)
        Ewacum=np.cumsum(np.real(XW*np.conj(XW))/len(self.x)/(E/2))
        ind=np.min(np.where(Ewacum>umbral))
        return self.fs/len(self.x)*ind

    def agc(self):
        xo=self.x.copy()
        x=self.x.copy()
        bw=self.getBW()
        L=int(self.fs/bw+1)
        n=len(xo)
        nf=int(np.ceil(n/L)*L-n)
        xf=np.zeros(nf)
        x=np.concatenate((x ,xf))
        escala=[]
        y=0*xo
        for k in range(int(len(x)/L)):
            i=int(k)
            a=i*L
            b=(i+1)*L
            arr=x[a:b]
            K=np.max(np.abs(arr))
            escala.append(K)
        for i in range(len(xo)):
            if(i<=L/2): 
                y[i]=xo[i]/escala[0]
            else:
                a=int(max(0, np.floor((i-L/2)/L)))
                b=a+1
                pa=int(L/2+a*L)
                if(b<len(escala)):
                    m=(escala[b]-escala[a])/L
                    B=escala[a]-m*pa
                    e=m*i+B
                    y[i]=xo[i]/e
                else:
                    y[i]=xo[i]/escala[a]
        
        return senal(x=y, fs=self.fs, nombre=self.nombre+'_AGC')

    def PLL(y=[], fs=44100, fc=0, dibujar=False): #Señal de referencia a seguir. 
        ts=1/fs
        Kd=0.5
        eta=0.707
        Bn=0.01*fs
        Ko=1
        Kp=1/(Kd*Ko)*4*eta/(eta+1/(4*eta))*(Bn/fs)
        Ki=1/(Kd*Ko)*4/((eta+1/(4*eta))**2)*(Bn/fs)**2
        
        ys=senal(x=y, fs=fs, nombre='Señal de referencia')
        yref=ys.agc()
        integrator_out = 0
        phase_estimate = 0*yref.x
        e_D = [] #phase-error output
        e_F = [] #loop filter output
        sin_out = 0*yref.x
        cos_out = 0*yref.x
        theta=0
        dtheta=2*np.pi*fc*ts #Cambio de frecuencia de la portadora
        
        for n in range(len(yref.x)-1):
            # phase detector
            e_D.append(yref.x[n] * sin_out[n])
        
            #loop filter
            integrator_out += Ki * e_D[n]
            e_F.append(Kp * e_D[n] + integrator_out)
        
            #NCO
            phase_estimate[n+1] = phase_estimate[n] + Ko * e_F[n]
            theta+=dtheta 
            
            sin_out[n+1] = -np.sin(theta+phase_estimate[n])
            cos_out[n+1] = np.cos(theta+phase_estimate[n])
        
        if(dibujar):
            # Create a Figure
            fig = plt.figure()
            
            # Set up Axes
            ax1 = fig.add_subplot(111)
            t=np.arange(0,len(cos_out))*ts
            ax1.plot(t,cos_out, label='PLL Output')
            plt.grid()
            ax1.plot(t,y, label='Input Signal')
            plt.legend()
            ax1.set_title('Waveforms')
        out=senal(x=cos_out, fs=fs, nombre='Señal de reloj')
        out.fc=fc
        out.phase=phase_estimate
        return out
    def mediaMovil(self, L, dibujar=False):
        z=0.0*np.array(self.x)
        for i in range(len(z)):
            a=max(0,i-L+1)
            n=i-a
            if(n==0):
                z[i]=self.x[i]
            else:
                z[i]=np.mean(self.x[a:(i+1)])
            #print(i,a,n,self.x[a:(i+1)], z[i])
        if(dibujar):
            t=self.getT()
            plt.figure()
            plt.plot(t,self.x)
            plt.plot(t,z)
        return senal(x=z, fs=self.fs, nombre='Media Movil')
    
    def awgn(self, snr):
        self.x=awgn(self.x, snr)
        
    def getX(self, t=0):
        ind=int(t*self.fs)
        return self.x[ind]
    
class codificador:
    def __init__(self, x=[], n=8):
        self.x=x
        self.n=n
        self.seq1=self.cad2Bin('345yu7tjgfbdret45yr6hfgbtdrfte5y6rh')
        
    def char2Bin(self, x):
        cad=bin(x)[2:]
        while(len(cad)<self.n):
            cad='0'+cad
        arr=[ord(item)-48 for item in cad]
        return arr
    
    def cad2Bin(self, cad):
        bits=[]
        for c in cad:
            wrd=self.char2Bin(ord(c))
            bits=bits+wrd
        return bits
    
    def getInt(self, x):
        n=len(x)
        K=np.power(2,n-1)
        suma=0
        for xi in x:
            suma=suma+K*xi
            K=K/2
        return suma
    
    def bits2Cad(self, bits, n=8):
        cad=''
        a=0
        while(a+n-1<len(bits)):
            arr=np.take(bits, range(a, a+n))
            num=self.getInt(arr)
            cad=cad+chr(int(num))
            a+=8
        return cad
    def bits2Polar(bits):
        return np.array(bits)*2-1
    
    def separarRI(self, x): #Toma una secuencia de bits y la separa en dos, intercalando los bits
        if(len(x) % 2 ==1):
            x=np.concatenate((x,[0]))
        n=len(x)
        i=np.arange(0,n,2)
        j=np.arange(1,n,2)
        R=[x[int(xi)] for xi in i]
        I=[x[int(xi)] for xi in j]
        return R,I

    def agregarRI(self, xR, xI): #Toma una secuencia de bits y la separa en dos, intercalando los bits
        bits=[]
        for i in range(len(xR)):
            bits.append(xR[i])
            bits.append(xI[i])
        return bits
    def mod2Sum(self, x,y):
        return (x+y)%2
    def mod2prod(self, x,y):
        return np.matmul(x,y) % 2
    
class blockCode:
    def G6_3():
        return blockCode(G=np.array([[1,0,0,1,1,1],[0,1,0,1,0,1],[0,0,1,0,1,1]]))
    def G7_4():
        return blockCode(G=np.array([[1,0,0,0,1,1,1],[0,1,0,0,1,0,1],[0,0,1,0,0,1,1],[0,0,0,1,1,1,0]]))
    
    def __init__(self, G):
        self.G=G
        self.k, self.N=G.shape
        self.cod=codificador()
        self.getCodeWords()
        
    def codificar(self, w):
        return self.cod.mod2prod(w,self.G)
    
    def codificarMsg(self, bits):
        i=0
        salida=np.array([])
        while(i+self.k<len(bits)):
            x=self.codificar(bits[i:(i+self.k)])
            salida=np.concatenate((salida, x))
            i+=self.k
        if(i<len(bits)):
            u=bits[i:len(bits)]
            falta=self.k-len(u)
            u=np.concatenate((u,np.zeros(falta)))
            salida=np.concatenate((salida, self.codificar(u)))
        return salida

    def getCodeWords(self):
        Nmax=np.power(2, self.k)
        dmin=100000
        salida=[]
        for i in range(Nmax):
            b=bin(i)[2:].zfill(self.k)
            x=[ord(item)-48 for item in b]
            x=np.array(x)
            x=self.codificar(x)
            if(np.sum(x)>0):
                dmin=min(np.sum(x), dmin)
            salida.append(x)
        self.cw=salida
        self.dmin=dmin
        return salida, dmin
    
    def hammingDist(self, x1, x2):
        return np.sum(self.cod.mod2Sum(x1,x2))
    
    def decodificar(self, x):
        dmin=self.dmin*100 #Inicializando a un valor grande
        for i in range(len(self.cw)):
            d=self.hammingDist(x, self.cw[i])
            if(d<dmin):
                dmin=d
                io=i
        b=bin(io)[2:].zfill(self.k)
        x=[ord(item)-48 for item in b]
        x=np.array(x)
        return x
    def decodificarMsg(self, bits):
        i=0
        salida=np.array([])
        while(i+self.N<len(bits)):
            x=self.decodificar(bits[i:(i+self.N)])
            salida=np.concatenate((salida, x))
            i+=self.N
        if(i<len(bits)):
            u=bits[i:len(bits)]
            falta=self.N-len(u)
            u=np.concatenate((u,np.zeros(falta)))
            salida=np.concatenate((salida, self.decodificar(u)))
        return salida
    def crearCodigo(K, N,dHmin=3):
        bc=blockCode.G6_3()
        #Busca crear por medio de rutinas heurísticas, códigos que tengan la mayor distancia de hamming posible
        #Primero vamos a crear los k símbolos y para cada uno de ellos las palabras que comienzan por dicho valor
        palabras=[]
        
        Nmax=np.power(2, K)
        for i in range(Nmax):
            M=N-K #Número de símbolos de paridad
            b=bin(i)[2:].zfill(K)
            x=[ord(item)-48 for item in b]
            x0=np.array(x)
            
            tmp=[]
            if(i>0):
                for j in range(np.power(2,M)):
                    b=bin(j)[2:].zfill(M)
                    x=[ord(item)-48 for item in b]
                    x=np.array(x)
                    x=np.concatenate((x0,x))
                    if(np.sum(x)>=dHmin):
                        tmp.append(x)
                palabras.append(tmp)
            else:
                palabras.append(np.zeros(N))
        dH=0
        for i in range(500):
            cod=[palabras[0]]
            used=[0]
            for j in range(1,len(palabras)):
                ind=np.random.randint(len(palabras[j]))
                cod.append(palabras[j][ind])
                used.append(ind)
            do=100
            for j in range(len(cod)-1):
                for k in range(j+1,len(cod)):
                    do=min(do, bc.hammingDist(cod[j], cod[k]))
            if(do>dH):
                codOut=used
                dH=do
                #print(dH, codOut)
        G=[]
        for n in range(K-1,-1,-1):
            i=np.power(2,n)
            #print(i)
            G.append(palabras[i][codOut[i]])
        G=np.array(G)
        return blockCode(G)
    
class convCode:
    def __init__(self, G):
        self.k=1
        self.N=len(G) #Número de salidas del codificador
        self.L=len(G[0])
        self.G=G
        self.cod=codificador()
        
    def codificar(self, msg):
        #inicialización del registro de desplazamiento
        msg=np.concatenate((msg, np.zeros(self.L)))
        x=np.zeros((self.L))
        y=np.array([])
        for i in range(len(msg)):
            x=np.roll(x,1)
            x[0]=msg[i]
            for j in range(self.N):
                yi=self.cod.mod2prod(x,self.G[j])
                #print(y, yi)
                y=np.concatenate((y,[yi]))
        return y
    
    def fsm(self, x, msg):
        x=np.concatenate(([msg],x))
        y=[]
        for j in range(self.N):
            y.append(self.cod.mod2prod(x,self.G[j]))
        return x[0:-1], np.array(y)
    
    def decodificar(self, bits):
        rutas=[[([0,0],0,0)]] #Rutas actuales del árbol de decodificación
        #La estructura de cada ruta es ([estado1, estado2, estado3], error)
        i=0
        while(i+self.N<len(bits)):
            rutas=self.agregarRutas(rutas, bits[i:(i+self.N)])
            rutas=self.eliminarRutas(rutas)
            i+=self.N
        if(i<len(bits)):
            u=bits[i:len(bits)]
            falta=self.N-len(u)
            print(u, falta)
            u=np.concatenate((u,np.zeros(falta)))
            #print(u)
            rutas=self.agregarRutas(rutas, u)
            rutas=self.eliminarRutas(rutas)
        
        emin=10000
        for ruta in rutas:
            error=[x[2] for x in ruta][-1]
            if(error<emin):
                salida=[x[1] for x in ruta]
                emin=error
            
        return salida[1:-3]
    
    def agregarRutas(self, rutas, msg):
        salida=[]
        for r in rutas:
            #En caso de llegar un 0
            rn=r.copy()
            endState, bit, error=rn[-1] #Estado actual o final y error acumulado
            newState, msgEmitido=self.fsm(endState, 0)
            #print(msgEmitido, msg)
            errorn=error+np.sum(self.cod.mod2Sum(msg, msgEmitido))
            rn.append((newState, 0, errorn))
            salida.append(rn)
            
            #En caso de llegar un 1
            rn=r.copy()
            endState, bit, error=rn[-1] #Estado actual o final y error acumulado
            newState, msgEmitido=self.fsm(endState, 1)
            errorn=error+np.sum(self.cod.mod2Sum(msg, msgEmitido))
            rn.append((newState, 1, errorn))
            salida.append(rn)
        return salida
    def eliminarRutas(self, rutas):
        #Quita las rutas de mayor error al mismo estado
        salida={}
        for ruta in rutas:
            endState, bit, error=ruta[-1]
            IntEndState=int(self.cod.getInt(endState))
            if(IntEndState in salida.keys()):
                #Existe, verificar menor error
                endState2, bit2, error2=salida[IntEndState][-1]
                if(error<error2):
                    salida[IntEndState]=ruta
            else:
                salida[IntEndState]=ruta
                
        return list(salida.values())
    
class interliever:
    def __init__(self):
        self.lenPos = 100
        self.mapeador = [70, 16, 11,  2, 57, 22,  7, 87, 74, 32, 96, 51, 75, 49, 86, 89, 23,
                         14, 38,  9,  3, 95, 82, 54, 85, 45, 94, 36, 66,  1, 40,  4, 33, 27,
                         81, 68, 44, 64, 20, 79, 8, 67, 28, 55, 78, 18,  5, 83, 92,  0, 39,
                         63, 15, 97, 41, 12, 47, 59, 76, 60, 43, 80, 21, 99, 91, 10, 90, 50,
                         98, 62, 88, 77, 56, 46, 52, 31, 53, 34, 58, 25, 93, 73, 37, 29, 35,
                         48, 24, 17, 13, 71, 61, 69, 84, 65, 30,  6, 19, 42, 72, 26]
    
    def mezclar(self, mensaje = []):
        desorden = np.array([])
        
        if(len(mensaje)%100 != 0):                                          #Se valida que la longitud del mensaje sea multiplo de 100
            falta = np.zeros(100-(len(mensaje)%100))
            mensaje = np.concatenate((mensaje, falta))
        
        for pedazo in range(0,len(mensaje),self.lenPos):                    #Se mezclan las posiciones del msg
            seccion = np.array(mensaje[pedazo:pedazo+self.lenPos])
            concatenador = []
            for pos in range(len(self.mapeador)):
                concatenador.append(seccion[self.mapeador[pos]])   
            desorden = np.concatenate((desorden,concatenador))   
        return desorden
    
    def reordenar(self, desorden = []):
        recepcion = []
        seccion = []

        if(len(desorden)%100 != 0):                                          #Se valida que la longitud del mensaje sea multiplo de 100
            falta = np.zeros(100-(len(desorden)%100))
            desorden = np.concatenate((desorden, falta))
        
        for pedazo in range(0,len(desorden),self.lenPos):                   #Se recuperan las posiciones originales del msg
            seccion =np.array(desorden[pedazo:pedazo+self.lenPos])
            concatenador = np.zeros(self.lenPos)
            for pos in range(len(self.mapeador)):
                concatenador[self.mapeador[pos]] = (seccion[pos])   
            recepcion = np.concatenate((recepcion,concatenador))
        return recepcion
    
            