
import numpy as np
from turbocode import turbo
import sys
from get_args import get_args
import time
from models import DNCNN_train,DNCNN_predict,errors
from utils2 import generateData
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__=='__main__':
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)

    # put all printed things to log file
    logfile = open('./logs/'+identity+'_log.txt', 'a')
    sys.stdout = Logger('./logs/'+identity+'_log.txt', sys.stdout)

    args = get_args()
    print(args)

    turbocode=turbo(args)
    channel=1
    train_input,train_noise=generateData(args,channel,'train')
    valid_input,valid_noise=generateData(args,channel,'test')
    train_label=train_input-train_noise
    valid_label=valid_input-valid_noise
    DNCNN_train(args,train_input,train_label,valid_input,valid_label,args.channel,identity)
    # identity=710169
    test_input,test_noise=generateData(args,channel,'test')
    test_label=test_input-test_noise
    test_output=DNCNN_predict(args,test_input,args.channel,identity)
    denoisesig=np.round(test_output)
    test_ber=np.sum(np.logical_xor(denoisesig,test_label),axis=(0,1,2,3))/(args.num_block*channel*args.block_len*args.code_rate_k)
    print("===>Test set BER ",float(test_ber))

    

        
    

