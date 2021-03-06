"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import math
import numpy as np

import time
from noise import generate_noise

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)
def getnoisesigma(snr_low=0.0,snr_high=0.0):
	this_sigma_low = snr_db2sigma(snr_low)
	this_sigma_high= snr_db2sigma(snr_high)
	# mixture of noise sigma.
	this_sigma = (this_sigma_low - this_sigma_high) * np.random.rand(1) + this_sigma_high  #每个batch添加相同的噪声
	this_snr=-20.0 * np.log10(this_sigma)
	return (this_sigma,this_snr)
def generateData2(args,channel,mode):
	if mode=="train":
		batchnum=args.num_block
	elif mode=="test":
		batchnum=args.num_test_block
	X_train=np.random.randint(0,2,(batchnum,channel,args.block_len, args.code_rate_k))
	noise_shape = (batchnum,channel,args.block_len, args.code_rate_k)
	this_sigma,this_snr=getnoisesigma(args.train_channel_low,args.train_channel_high)
	# print("batch_idx=",batch_idx,"noise sigma=",this_sigma,"this_snr=",this_snr)
	fwd_noise  = generate_noise(noise_shape, args, this_sigma)
	codes=X_train
	received_codes=codes + fwd_noise
	inputdata=received_codes
	return (inputdata,fwd_noise)
def generateData(args,mode):
	if mode=="train":
		batchnum=args.num_block
	elif mode=="test":
		batchnum=args.num_test_block
	X_train=np.random.randint(0,2,(batchnum,args.block_len, args.code_rate_k))
	noise_shape = (batchnum,args.block_len, args.code_rate_k)
	this_sigma,this_snr=getnoisesigma(args.train_channel_low,args.train_channel_high)
	# print("batch_idx=",batch_idx,"noise sigma=",this_sigma,"this_snr=",this_snr)
	fwd_noise  = generate_noise(noise_shape, args, this_sigma)
	codes=X_train
	received_codes=codes + fwd_noise
	inputdata=received_codes
	return (inputdata,fwd_noise)
def trainer(args,model,epoch,optimizer,criterion,turbo,use_cuda=False,verbose=True):
	device = torch.device("cuda" if use_cuda else "cpu")
	model.train()
	start_time = time.time()
	train_loss = 0.0
	for batch_idx in range(int(args.num_block/args.batch_size)):
		optimizer.zero_grad()
		model.zero_grad()
		block_len = args.block_len
		channel=1
		X_train    = torch.randint(0, 2, (args.batch_size, channel,block_len, args.code_rate_k), dtype=torch.float)

		noise_shape = (args.batch_size, channel,args.block_len, args.code_rate_k)
		this_sigma,this_snr=getnoisesigma(args.train_channel_low,args.train_channel_high)
		# print("batch_idx=",batch_idx,"noise sigma=",this_sigma,"this_snr=",this_snr)
		fwd_noise  = generate_noise(noise_shape, args, this_sigma)

		X_train, fwd_noise = X_train.to(device), fwd_noise.to(device)

		num_block=X_train.shape[0]
		# x_code=[]
		# for idx in range(num_block):
		# 	np_inputs=np.array(X_train[idx,:,0].type(torch.IntTensor).detach())
		# 	[sys,par1,par2]=turbo.encoder(np_inputs)
		# 	xx = np.array([sys, par1, par2]).T
		# 	x_code.append(xx)
		# codes=torch.from_numpy(np.array(x_code)).type(torch.FloatTensor)
		codes=X_train
		# codes=2.0*codes-1.0
		received_codes=codes + fwd_noise
		noisemap=this_sigma*torch.ones(args.batch_size, args.block_len, 1)
		# inputdata=torch.cat((noisemap,received_codes),2)
		inputdata=received_codes
		output=model(inputdata)
		loss = criterion(output, fwd_noise)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
		# for name,parms in model.named_parameters():
		# 	print("--->name:",name,"--->grad_requirs:",parms.requires_grad,"--->grad_value:",parms.grad)
	end_time = time.time()
	train_loss = train_loss /(args.num_block/args.batch_size)
	if verbose:
		print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss),' running time', str(end_time - start_time))
	return train_loss

def validate(args,model,epoch,optimizer,criterion,turbo,use_cuda=False,verbose=True):
	device = torch.device("cuda" if use_cuda else "cpu")
	model.eval()
	test_loss= 0.0
	test_ber=0.0
	with torch.no_grad():
		num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
		for batch_idx in range(num_test_batch):
			channel=1
			X_test     = torch.randint(0, 2, (args.batch_size, channel,args.block_len, args.code_rate_k), dtype=torch.float)

			noise_shape = (args.batch_size, channel,args.block_len, args.code_rate_k)
			this_sigma,this_snr=getnoisesigma(args.train_channel_low,args.train_channel_high)
			# print("batch_idx=",batch_idx,"noise sigma=",this_sigma,"this_snr=k",this_snr)
			fwd_noise  = generate_noise(noise_shape, args, this_sigma)

			X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)
			
			optimizer.zero_grad()
			# num_block=X_test.shape[0]
			# x_code=[]
			# for idx in range(num_block):
			# 	np_inputs=np.array(X_test[idx,:,0].type(torch.IntTensor).detach())
			# 	[sys,par1,par2]=turbo.encoder(np_inputs)
			# 	xx = np.array([sys, par1, par2]).T
			# 	x_code.append(xx)
			# codes=torch.from_numpy(np.array(x_code)).type(torch.FloatTensor)
			
			codes=X_test
			# codes=2.0*codes-1.0
			received_codes=codes + fwd_noise
			noisemap=this_sigma*torch.ones(args.batch_size, args.block_len, 1)
			# inputdata=torch.cat((noisemap,received_codes),2)
			inputdata=received_codes
			output=model(inputdata)
			test_loss += F.mse_loss(output, fwd_noise)
			denoisesig=received_codes-output
			denoisesig=torch.clamp(denoisesig,0,1)
			np_outputs=denoisesig.data.numpy()
			np_outputs=np.round(np_outputs)
			# x_code=np.array(x_code)
			x_code=np.array(codes.type(torch.IntTensor).detach())
			test_ber=test_ber+np.sum(np.logical_xor(np_outputs,x_code),axis=(0,1,2,3))


	test_loss /= num_test_batch
	test_ber/=(num_test_batch*args.batch_size*args.block_len* args.code_rate_k)
	if verbose:
		print('====> Test set MSE loss', float(test_loss),"===>Test set BER ",float(test_ber))
def calber(y_true,y_pred):
	ber=np.sum(np.bitwise_xor(y_true,y_pred))
	return ber
def test(args,model,turbo,use_cuda=False,verbose=True):
	device = torch.device("cuda" if use_cuda else "cpu")
	model.eval()
	ber_res=[]
	snr_interval=(args.snr_test_end-args.snr_test_start)*1.0/(args.snr_points-1)
	snrs=[snr_interval*item+args.snr_test_start for item in range(args.snr_points)]
	print("SNRS:",snrs)
	sigmas=snrs
	for sigma,this_snr in zip(sigmas,snrs):
		num_test_batch=int(args.num_block/(args.batch_size))
		test_ber=0.0
		train_ber=0.0
		for batch_idx in range(num_test_batch):
			X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)

			noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
			fwd_noise  = generate_noise(noise_shape, args, sigma)

			X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)
			num_block=X_test.shape[0]
			x_code=[]
			for idx in range(num_block):
				np_inputs=np.array(X_test[idx,:,0].type(torch.IntTensor).detach())
				[sys,par1,par2]=turbo.encoder(np_inputs)
				xx = np.array([sys, par1, par2]).T
				decodebits=turbo.decoder(xx,this_snr)
				x_code.append(xx)
				# train_ber+=calber(np_inputs,decodebits[:,0])
			codes=torch.from_numpy(np.array(x_code)).type(torch.FloatTensor)
			codes=2.0*codes-1.0
			received_codes=codes + fwd_noise
			noisemap=sigma*torch.ones(args.batch_size, args.block_len, 1)
			inputdata=torch.cat((noisemap,received_codes),2)
			output=model(inputdata)
			np_outputs=np.array(output.type(torch.IntTensor).detach())
			np_outputs=np_outputs>0
			x_code=np.array(x_code)
			test_ber=test_ber+np.sum(np.bitwise_xor(np_outputs,x_code),axis=(0,1,2))
			# output=received_codes
			# for idx in range(num_block):
			# 	np_inputs=np.array(output[idx,:,:].type(torch.IntTensor).detach())
			# 	decodebits=turbo.decoder(np_inputs,this_snr)
			# 	y_true=np.array(X_test[idx,:,0].type(torch.IntTensor).detach())
			# 	test_ber+=calber(y_true,decodebits[:,0])
		test_ber/=(num_test_batch*args.batch_size*args.block_len* args.code_rate_n)
		train_ber/=(num_test_batch*args.batch_size*args.block_len* args.code_rate_k)
		print("snr=",this_snr,"trainber=",train_ber,"testber=",test_ber)
		ber_res.append(test_ber)
	print("BERS:",ber_res)	
			
			

