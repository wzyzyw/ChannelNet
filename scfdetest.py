
import numpy as np
from turbocode import turbo,matlabturbo,classicalturbo
import sys
from get_args import get_args
import time
from models import DNCNN_train,DNCNN_predict,errors
from utils2 import generateEncodeData_test,getnoisesigma
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
def channeldecode(args,turbo,y,mode):
    if mode=="train":
        batchnum=args.num_block
    elif mode=="test":
        batchnum=args.num_test_block
    snr=args.train_channel_low
    decodebits=np.zeros((batchnum,args.block_len,args.code_rate_k))
    for idx_batch in range(batchnum):
        decodebits[idx_batch,:,:]=turbo.decoder(y[idx_batch,:,:],snr)
    return decodebits

if __name__=='__main__':
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)

    # put all printed things to log file
    logfile = open('./logs/'+identity+'_log.txt', 'a')
    sys.stdout = Logger('./logs/'+identity+'_log.txt', sys.stdout)

    args = get_args()
    print(args)

    if args.dec_alg=="pythonturbo":
        myturbo=turbo(args)
    elif args.dec_alg=="matlabturbo2":
        myturbo=classicalturbo()
    else:
        raise Exception("error turbo!!!")
    identity=333827
    sigmalist,snrlist=getnoisesigma(args.snr_test_start,args.snr_test_end,interval=args.snr_interval,mode=args.add_mode)
    bers=[]
    for sigma,snr in zip(sigmalist,snrlist):
        print("current snr=",snr)
        X_test_data,test_input,test_noise=generateEncodeData_test(args,'test',myturbo,sigma)
        test_label=test_input[:,:,:3]-test_noise
        test_output=DNCNN_predict(args,test_input,args.channel,identity)
        denoisesig=test_input[:,:,:3]-test_output
        # denoisesig=test_input
        # # channel decode
        # test_label_decodebits=channeldecode(args,myturbo,test_label,'test')
        # test_output_decodebits=channeldecode(args,myturbo,test_output,'test')
        denoisesig=np.round(denoisesig)
        test_ber=np.sum(np.logical_xor(denoisesig,test_label),axis=(0,1,2))/(args.num_block*args.block_len*args.code_rate_n)
        bers.append(test_ber)
    print("===>Test set SNR ",snrlist)
    print("===>Test set BER ",bers)

    

        
    

