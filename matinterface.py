import transplant
import numpy as np
# 元类用于创建类，在python中类也是一种对象，类这种对象的创建由元类实现的
# type是pyhton内建的元类，python中内建的类都是type创建的，type不仅用来判断对象类型，也可以用来动态创建类
# __metaclass__属性用来实现自定义的元类，类似于lua中的元表
class SingletonMeta(type):
    _instances={}
    # 重载()运算符，使得类实例可以通过:"对象名()"方法调用函数
    def __call__(cls,*args,**kwargs):
        if cls not in cls._instances:
            instance=super().__call__(*args,**kwargs)
            cls._instances[cls]=instance
        return cls._instances[cls]
class Cmatlab(metaclass=SingletonMeta):
    def __init__(self):
        self.__matlab = transplant.Matlab(executable="D:/matlab2018/bin/matlab.exe",jvm=False, desktop=True)
    def close(self):
        self.__matlab.exit
    def modulate(self,modtype,msgint):
        mod_data=self.__matlab.modulation(modtype,msgint)
        return mod_data
    def demodulate(self,modtype,data_mod):
        rx_bits=self.__matlab.demodulation(modtype,data_mod)
        return rx_bits
    def wattersonmodel(self,ts,channel_type,samplenum):
        hresponse=self.__matlab.wattersonchannel(ts,channel_type,samplenum)
        return hresponse
    def filter(self,h,tx):
        return self.__matlab.filter(h,1,tx)
    def my_filter(self,h,tx):
        tmp=self.__matlab.myfilter(h,tx)
        return tmp
    def stbc22(self,data,Nfft,frame_num):
        tmp=self.__matlab.sc_stbc(data,Nfft,frame_num)
        return (tmp[0],tmp[1])
    def stbc_decode22(self,H1,H2,H3,H4,data1,data2,snr,flag):
        return self.__matlab.sc_stbc_decode_2x2(H1,H2,H3,H4,data1,data2,snr,flag)
    def lsestimator(self,data_pilot,interval,pilotseq,method):
        tmp=self.__matlab.ls_channelestimator(data_pilot,interval,pilotseq,method)
        return (tmp[0],tmp[1])
    def rlsestimator(self,sigma,lamda,tx1,tx2,train_len,durance_rls,blocknum,modtype,Nfft,Rx1,Rx2):
        tmp=self.__matlab.rls(sigma,lamda,tx1,tx2,train_len,durance_rls,blocknum,modtype,Nfft,Rx1,Rx2)
        return tmp
    def allsys(self,bits,h1,h2,h3,h4,snr,Nfft,blocknum,Nps,Nt,Ng,method,mod_type):
        tmp=self.__matlab.lsmimosystem(bits,h1,h2,h3,h4,snr,Nfft,blocknum,Nps,Nt,Ng,method,mod_type)
        return tmp
    def circlechanest(self,cplen,pilotnum,pilotlen,pilot,rxpilot):
        tmp=self.__matlab.circle_chan_est(cplen,pilotnum,pilotlen,pilot[0],pilot[1],rxpilot)
        return (tmp[0],tmp[1])
    def turboencode(self,msgint,interleaverindex):
        tmp=self.__matlab.turboencode(msgint,interleaverindex)
        return tmp
    def turbodecode(self,demod_bit,interleaverindex):
        tmp=self.__matlab.turbodecode(demod_bit,interleaverindex)
        return tmp
    def classicalturboencode(self,msgint,g,interleaverindex,puncture):
        tmp=self.__matlab.encoderm(msgint,g,interleaverindex,puncture)
        return tmp
    def classicalturbodecode(self,r,interleaverindex,puncture,L_c,L_total,dec_alg,niter,g,m):
        # 暂时不考虑迭代过程中的ber，只取最后一次迭代的结果
        tmp=self.__matlab.decoderm(r,interleaverindex,puncture,L_c,L_total,dec_alg,niter,g,m)
        return tmp
    def awgn(self,y,snr):
        snr=float(snr)
        tmp=self.__matlab.awgn(y,snr,'measured')
        return tmp
if __name__ == "__main__":
    m1=Cmatlab()
    m2=Cmatlab()
    if id(m1)==id(m2):
        print("Singleton works, both variables contain the same instance.")
    else:
        print("Singleton failed, variables contain different instances.")
    # 调用filter函数返回到python中是个list，包括两个numpy数组，只有第一个是正确结果（matlab输出的结果）
    a=np.array([1,0,1])
    a=np.reshape(a,(-1,1),order="F")
    b=np.array([i for i in range(144)])
    b=np.reshape(b,(-1,1),order="F")
    c=m1.filter(a,b)
    d=-1