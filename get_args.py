__author__ = 'yihanjiang'
import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-enc1',  type=int, default=7)
    parser.add_argument('-enc2',  type=int, default=5)
    parser.add_argument('-feedback',  type=int, default=7)
    parser.add_argument('-M',  type=int, default=2, help="Number of delay elements in the convolutional encoder")
    # Channel related parameters
    ################################################################
    parser.add_argument('-channel', choices = ['awgn',             # AWGN
                                               "its", # HF ITS noise ,,
                                               "bikappa",# HF bikappa noise
                                               ],
                        default = 'awgn')

    # continuous channels training algorithms
    parser.add_argument('-train_channel_low', type=float, default  = 15.0)
    parser.add_argument('-train_channel_high', type=float, default =15.0)
    parser.add_argument('-init_nw_weight', type=str, default='./models/torch_model_decoder_036718.pt')

    # code rate is k/n, so that enable multiple code rates. This has to match the encoder/decoder nw structure.
    parser.add_argument('-code_rate_k', type=int, default=1)
    parser.add_argument('-code_rate_n', type=int, default=3)

    ################################################################
    # TurboAE encoder/decoder parameters
    ################################################################
    
    parser.add_argument('-num_iteration', type=int, default=6)
    parser.add_argument('-is_parallel', type=int, default=0)
    # CNN related
    parser.add_argument('-kernel_size', type=int, default=3)

    # CNN/RNN related
    parser.add_argument('-num_layer', type=int, default=14)


    ################################################################
    # Training ALgorithm related parameters
    ################################################################

    parser.add_argument('-dropout',type=float, default=0.5)

    parser.add_argument('-snr_test_start', type=float, default=15.0)
    parser.add_argument('-snr_test_end', type=float, default=15.0)
    parser.add_argument('-snr_points', type=int, default=9)

    parser.add_argument('-batch_size', type=int, default=100  )
    parser.add_argument('-num_epoch', type=int, default=200)
    parser.add_argument('-test_ratio', type=int, default=1,help = 'only for high SNR testing')
    # block length related
    parser.add_argument('-block_len', type=int, default=100)



    parser.add_argument('-num_block', type=int, default=7000)
    parser.add_argument('-num_test_block', type=int, default=3000)
    parser.add_argument('-test_channel_mode',
                        choices=['block_norm','block_norm_ste'],
                        default='block_norm')
    parser.add_argument('-train_channel_mode',
                        choices=['block_norm','block_norm_ste'],
                        default='block_norm')
   
    ################################################################
    # STE related parameters
    ################################################################
    parser.add_argument('-enc_quantize_level', type=float, default=2, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_value_limit', type=float, default=1.0, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_grad_limit', type=float, default=0.01, help = 'only valid for block_norm_ste')
    parser.add_argument('-enc_clipping', choices=['inputs', 'gradient', 'both', 'none'], default='both',
                        help = 'only valid for ste')

    ################################################################
    # Optimizer related parameters
    ################################################################
    parser.add_argument('-optimizer', choices=['adam', 'lookahead', 'sgd'], default='adam', help = '....:)')
    parser.add_argument('-lr', type = float, default=0.0001, help='decoder leanring rate')
    parser.add_argument('-momentum', type = float, default=0.9)



    ################################################################
    # MISC
    ################################################################
    parser.add_argument('--is_train',default=True)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--rec_quantize', action='store_true', default=False,
                        help='binarize received signal, which will degrade performance a lot')
    parser.add_argument('-rec_quantize_level', type=int, default=2,
                        help='binarize received signal, which will degrade performance a lot')
    parser.add_argument('-rec_quantize_limit', type=float, default=1.0,
                        help='binarize received signal, which will degrade performance a lot')

    parser.add_argument('--print_pos_ber', action='store_true', default=False,
                        help='print positional ber when testing BER')
    parser.add_argument('--print_pos_power', action='store_true', default=False,
                        help='print positional power when testing BER')
    parser.add_argument('--print_test_traj', action='store_true', default=False,
                        help='print test trajectory when testing BER')
    parser.add_argument('--precompute_norm_stats', action='store_true', default=False,
                        help='Use pre-computed mean/std statistics')


    args = parser.parse_args()

    return args
