
import numpy as np
from turbocode import turbo
import sys
from get_args import get_args
import time
from models import DNCNN_train,DNCNN_predict,errors
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

    myturbo=turbo(args)
    X_train_data,train_input,train_noise=generateEncodeData(args,'train',myturbo)
    X_valid_data,valid_input,valid_noise=generateEncodeData(args,'test',myturbo)
    train_label=train_input-train_noise
    valid_label=valid_input-valid_noise
    DNCNN_train(args,train_input,train_label,valid_input,valid_label,args.channel,identity)
    # # identity=542302
    X_test_data,test_input,test_noise=generateEncodeData(args,'test',myturbo)
    test_label=test_input-test_noise
    test_output=DNCNN_predict(args,test_input,args.channel,identity)
    # # channel decode
    # test_label_decodebits=channeldecode(args,myturbo,test_label,'test')
    # test_output_decodebits=channeldecode(args,myturbo,test_output,'test')
    denoisesig=np.round(test_output)
    test_ber=np.sum(np.logical_xor(denoisesig,test_label),axis=(0,1,2))/(args.num_block*args.block_len*args.code_rate_n)
    print("===>Test set BER ",float(test_ber))
