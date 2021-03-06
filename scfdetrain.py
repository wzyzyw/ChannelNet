
import sys
import time

import numpy as np

from get_args import get_args
from models import DNCNN_predict, DNCNN_train, errors
from turbocode import turbo,matlabturbo,classicalturbo
from utils2 import generateEncodeData


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
    elif args.dec_alg=="matlabturbo1":
        myturbo=matlabturbo(args)
    elif args.dec_alg=="matlabturbo2":
        myturbo=classicalturbo(args)
    else:
        raise Exception("error turbo!!!")
    X_train_data,train_input,train_noise=generateEncodeData(args,'train',myturbo)
    X_valid_data,valid_input,valid_noise=generateEncodeData(args,'test',myturbo)
    newn=args.code_rate_n+args.remainn
    train_label=train_input[:,:,:newn]-train_noise
    valid_label=valid_input[:,:,:newn]-valid_noise
    DNCNN_train(args,train_input,train_noise,valid_input,valid_noise,args.channel,identity)
    # # identity=542302
    X_test_data,test_input,test_noise=generateEncodeData(args,'test',myturbo)
    test_label=test_input[:,:,:newn]-test_noise
    test_output=DNCNN_predict(args,test_input,args.channel,identity)
    denoisesig=test_input[:,:,:newn]-test_output
    # # channel decode
    # test_label_decodebits=channeldecode(args,myturbo,test_label,'test')
    # test_output_decodebits=channeldecode(args,myturbo,test_output,'test')
    denoisesig=np.round(denoisesig)
    test_ber=np.sum(np.logical_xor(denoisesig,test_label),axis=(0,1,2))/(args.num_block*args.block_len*args.code_rate_n)
    print("===>Test set BER ",float(test_ber))

    

        
    

