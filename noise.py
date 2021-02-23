import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
import collections
import json
import os
class bikappa():
    def __init__(self):
        print("bikappa noise generate")
        self.K=1
        self.H=3
        self.L=-3
        self.bins=100
        self.sigma=30
        param={
            "K":self.K,
            "H":self.H,
            "L":self.L,
            "bins":self.bins,
            "sigma":self.sigma,
        }
        print("channel noise config",param)
    def bikappapdf(self,x):
        y=1/2/np.sqrt(np.pi)/self.sigma*(1+x**2/self.K/self.sigma**2)**-self.K
        # y=(1+x**2/self.K/self.sigma**2)**-self.K
        return y
    def getbikappa(self,noise_shape):
        step=(self.H-self.L)/(self.bins-1)
        tmpx=np.arange(self.L,self.H,step)
        d=self.bikappapdf(tmpx)
        D=np.cumsum(d)
        M,N=noise_shape[0],noise_shape[1]
        numbers=np.random.rand(M,N)
        y=numbers
        Y=np.zeros((1,M*N))
        multi=M*N
        maxX=D.shape[0]
        minX=1
        m=step
        b=self.L-m
        for c in range(multi):
            tmp1=np.where(y[c]>D)
            tmp2=np.where(y[c]<D)
            if tmp1[0].size==0:
                x=minX
            elif tmp2[0].size==0:
                x=maxX
            else:
                x_lo=np.max(tmp1)
                x_hi=np.min(tmp2)
                y_lo=D[x_lo]
                y_hi=D[x_hi]
                x=((x_hi-x_lo)/(y_hi-y_lo))*(y[c]-y_lo)+x_lo
            Y[0,c]=m*x+b
        Y=Y.reshape((M,N),order="F")
        return Y
    def addnoise(self,x,snr):
        noise=self.getbikappa(x,snr)
        x=x+noise
        return x
bikappaobj=bikappa()
def generate_noise(noise_shape, args, this_sigma=0.0):
    if args.channel == 'awgn':
        fwd_noise  = this_sigma * np.random.randn(noise_shape[0],noise_shape[1])
        # 注意调整noise形状对于二维数据和一维数据
    elif args.channel == 'bikappa':
        noise_num=noise_shape[0]*noise_shape[1]
        tmp=bikappaobj.getbikappa((noise_num,1))
        fwd_noise=tmp.reshape((noise_shape[0],noise_shape[1]))
    else:
        # Unspecific channel, use AWGN channel.
        fwd_noise  = this_sigma * np.random.randn(noise_shape)
    return fwd_noise