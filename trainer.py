"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
from torch import einsum
import math
from quantization_utils.quant_modules import *
from optimizer import SGD_adalr
from einops import rearrange
import os
from collections import deque

__all__ = ["Trainer"]



class ClassifierListImageNet2(object):
	def __init__(self) -> None:
		self.block_1 = nn.Sequential(
			nn.Linear(56*56,2048),
			nn.ReLU(),
			nn.Linear(2048,1000,bias=False)
		)
		self.block_2 = nn.Sequential(
			nn.Linear(28**2,2048),
			nn.ReLU(),
			nn.Linear(2048,1000,bias=False)
		)
		self.block_3 = nn.Sequential(
			nn.Linear(14**2,2048),
			nn.ReLU(),
			nn.Linear(2048,1000,bias=False)
		)
		self.opt_1 = torch.optim.SGD(
			self.block_1.parameters(),
			lr=0.00001,
			momentum=0.9
		)
		self.opt_2 = torch.optim.SGD(
			self.block_2.parameters(),
			lr=0.00001,
			momentum=0.9
		)
		self.opt_3 = torch.optim.SGD(
			self.block_3.parameters(),
			lr=0.00001,
			momentum=0.9
		)
		self.block = nn.ModuleList([self.block_1,self.block_2,self.block_3]).cuda()
		self.opts = [self.opt_1,self.opt_2,self.opt_3]
		self.kdloss = nn.KLDivLoss(reduction='batchmean').cuda()
		self.CeLoss = nn.CrossEntropyLoss().cuda()
		self.initial()
	def train(self,xs,teacher_output,T,alpha,target=None):
		self.block.train()
		b = F.softmax(teacher_output / T, dim=1)
		for idx,x in enumerate(xs) :
			x = rearrange(torch.mean(x,dim=1),'b h w -> b (h w)')
			o = self.block[idx](x)
			a = F.log_softmax(o / T, dim=1)
			c = (alpha * T * T)
			KD_loss = self.kdloss(a,b) * c
			self.opts[idx].zero_grad()
			KD_loss.backward()
			self.opts[idx].step()
	def fix(self):
		self.block.eval()
	def initial(self):
		w = torch.randn_like(self.block[0][-1].weight.data).cuda()
		for block in self.block:
			block[-1].weight.data = w.clone()
			block[-1].weight.requires_grad = True
	def set_train(self):
		self.block.train()

	def extract_embedding(self):
		return [ b[-1].weight.clone().detach() for b in self.block]
		

class ClassifierListMobileNet(object):
	def __init__(self) -> None:
		self.block_1 = nn.Sequential(
			nn.Linear(112*112,1280),
			nn.ReLU(),
			nn.Linear(1280,1000,bias=False)
		)
		self.block_2 = nn.Sequential(
			nn.Linear(56*56,1280),
			nn.ReLU(),
			nn.Linear(1280,1000,bias=False)
		)
		self.block_3 = nn.Sequential(
			nn.Linear(28*28,1280),
			nn.ReLU(),
			nn.Linear(1280,1000,bias=False)
		)
		self.opt_1 = torch.optim.SGD(
			self.block_1.parameters(),
			lr=0.00001,
			momentum=0.9
		)
		self.opt_2 = torch.optim.SGD(
			self.block_2.parameters(),
			lr=0.00001,
			momentum=0.9
		)
		self.opt_3 = torch.optim.SGD(
			self.block_3.parameters(),
			lr=0.00001,
			momentum=0.9
		)
		self.block = nn.ModuleList([self.block_1,self.block_2,self.block_3]).cuda()
		self.opts = [self.opt_1,self.opt_2,self.opt_3]
		self.kdloss = nn.KLDivLoss(reduction='batchmean').cuda()
		self.CeLoss = nn.CrossEntropyLoss().cuda()
		self.initial()
	def train(self,xs,teacher_output,T,alpha,target=None):
		self.block.train()
		b = F.softmax(teacher_output / T, dim=1)
		for idx,x in enumerate(xs) :
			x = rearrange(torch.mean(x,dim=1),'b h w -> b (h w)')
			o = self.block[idx](x)
			a = F.log_softmax(o / T, dim=1)
			c = (alpha * T * T)
			KD_loss = self.kdloss(a,b) * c
			self.opts[idx].zero_grad()
			KD_loss.backward()
			self.opts[idx].step()
	def fix(self):
		self.block.eval()
	def initial(self):
		w = torch.randn_like(self.block[0][-1].weight.data).cuda()
		for block in self.block:
			block[-1].weight.data = w.clone()
			block[-1].weight.requires_grad = True
	def set_train(self):
		self.block.train()

	def extract_embedding(self):
		return [ b[-1].weight.clone().detach() for b in self.block]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G,
	             train_loader, test_loader, settings, logger, tensorboard_logger=None,
	             opt_type="SGD", optimizer_state=None, run_count=0):
		"""
		init trainer
		"""
		
		self.settings = settings
		
		self.model = utils.data_parallel(
			model, self.settings.nGPU, self.settings.GPU)
		self.model_teacher = utils.data_parallel(
			model_teacher, self.settings.nGPU, self.settings.GPU)

		self.all_idx = torch.IntTensor([x for x in range(self.settings.nClasses)]).cuda()
		self.no_noise = torch.zeros(self.settings.nClasses,self.settings.latent_dim).cuda()

		self.generator = utils.data_parallel(
			generator, self.settings.nGPU, self.settings.GPU)

		self.train_loader = train_loader
		self.test_loader = test_loader
		self.tensorboard_logger = tensorboard_logger
		self.log_soft = nn.LogSoftmax(dim=1)
		self.MSE_loss = nn.MSELoss().cuda()
		self.lr_master_S = lr_master_S
		self.lr_master_G = lr_master_G
		self.opt_type = opt_type

		self.images_collection = dict()
		


		if opt_type == "SGD" :
			print("using defuat SGD")
			self.optimizer_S = torch.optim.SGD(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				nesterov=True,
			)
		else:
			assert False, "invalid type: %d" % opt_type
		if optimizer_state is not None:
			self.optimizer_S.load_state_dict(optimizer_state)

		self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
											betas=(self.settings.b1, self.settings.b2))

		self.logger = logger
		self.run_count = run_count
		self.scalar_info = {}
		self.mean_list = []
		self.var_list = []
		self.teacher_running_mean = []
		self.teacher_running_var = []
		self.save_BN_mean = []
		self.save_BN_var = []

		self.act_teacher_feature = []
		self.act_student_feature = [] 

		self.block_feature = []
		if self.model_teacher.__class__.__name__ == 'MobileNetV2':
			self.Classfier = ClassifierListMobileNet()
		else:
			self.Classfier = ClassifierListImageNet2()
		self.flag = False

		self.fix_G = False
	
	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_S = self.lr_master_S.get_lr(epoch)
		lr_G = self.lr_master_G.get_lr(epoch)
		# update learning rate of model optimizer
		for param_group in self.optimizer_S.param_groups:
			param_group['lr'] = lr_S
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr_G
		for opt in self.Classfier.opts:
			for param_group in opt.param_groups:
				param_group['lr'] = lr_G

	def loss_fn_kd(self, output, labels, teacher_outputs, linear=None):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha

		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""

		criterion_d = nn.CrossEntropyLoss(reduction='none').cuda()
		kdloss = nn.KLDivLoss(reduction='batchmean').cuda()

		alpha = self.settings.alpha
		T = self.settings.temperature
		a = F.log_softmax(output / T, dim=1)
		b = F.softmax(teacher_outputs / T, dim=1)
		c = (alpha * T * T)

		d = (-(linear*self.log_soft(output)).sum(dim=1)).mean()

		KD_loss = kdloss(a,b)*c
		return KD_loss

	
	def forward(self, images, teacher_outputs, labels=None, linear=None):
		"""
		forward propagation
		"""
		# forward and backward and optimize

		output = self.model(images)
		if labels is not None:
			loss = self.loss_fn_kd(output, labels, teacher_outputs, linear)
			return output, loss
		else:
			return output, None
	
	def backward_G(self, loss_G):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		loss_G.backward()
		self.optimizer_G.step()

	def backward_S(self, loss_S):
		"""
		backward propagation
		"""
		self.optimizer_S.zero_grad()
		loss_S.backward()
		self.optimizer_S.step()

	def backward(self, loss):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		self.optimizer_S.zero_grad()
		loss.backward()
		self.optimizer_G.step()
		self.optimizer_S.step()

	def hook_fn_forward(self,module, input, output):
		input = input[0]
		mean = input.mean([0, 2, 3])
		# use biased var in train
		var = input.var([0, 2, 3], unbiased=False)

		self.mean_list.append(mean)
		self.var_list.append(var)
		self.teacher_running_mean.append(module.running_mean)
		self.teacher_running_var.append(module.running_var)
	
	def register_block(self,module,input,output):
		self.block_feature.append(output)
	
	def register_teacher_block(self,module,input,output):
		self.act_teacher_feature.append(output)
	def register_student_block(self,module,input,output):
		self.act_student_feature.append(output)

	def collect_image(self, label, image_feature):
		labels = label.tolist()
		for label, f in zip(labels, image_feature):
			if self.images_collection.get(label) is None:
				self.images_collection[label] = deque(maxlen=1000)
				self.images_collection[label].append(f)
			else:
				self.images_collection[label].append(f)
	def sim_loss(self, label, image_feature):
			labels = label.tolist()
			loss = torch.zeros(1).cuda()
			for label, f in zip(labels, image_feature):
				f_mean = torch.stack(list(self.images_collection[label])).mean(dim=0).reshape(1, -1)
				loss += torch.clamp_min(F.cosine_similarity(f.reshape(1, -1), f_mean), 0)
			return loss / image_feature.size(0)
	def train(self, epoch):
		"""
		training
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		fp_acc = utils.AverageMeter()

		iters = 200
		self.update_lr(epoch)



		self.model.eval()
		self.model_teacher.eval()
		self.generator.train()
		
		start_time = time.time()
		end_time = start_time
		
		if epoch==0:
			for m in self.model_teacher.modules():
				if isinstance(m, nn.BatchNorm2d):
					m.register_forward_hook(self.hook_fn_forward)
			for n,p in self.model_teacher.named_modules():
				if n in ["features.stage1","features.stage1","features.stage2","features.stage3"] :
					p.register_forward_hook(self.register_block)
			
			for n,p in self.model_teacher.named_modules():
				if isinstance(p,nn.ReLU):
					p.register_forward_hook(self.register_teacher_block)
			for n,p in self.model.named_modules():
				if isinstance(p,QuantAct2):
					p.register_forward_hook(self.register_student_block)


		for i in range(iters):
			start_time = time.time()
			data_time = start_time - end_time
			multi_class = torch.rand(1)
			self.MERGE_PARAM = self.settings.multi_label_num
			MERGE_PROB = self.settings.multi_label_prob # superpose probability
			if multi_class<MERGE_PROB: 
				z = Variable(torch.randn(self.settings.batchSize, self.MERGE_PARAM,self.settings.latent_dim)).cuda()
				labels = Variable(torch.randint(0, self.settings.nClasses, (self.settings.batchSize,self.MERGE_PARAM))).cuda()
				linear = F.softmax(torch.randn(self.settings.batchSize,self.MERGE_PARAM),dim=1).cuda()
				z = z.contiguous()
				labels = labels.contiguous()
				labels_loss = Variable(torch.zeros(self.settings.batchSize,self.settings.nClasses)).cuda()

				images,attn_linear = self.generator(z, labels, linear)
				labels_loss.scatter_add_(1,labels,attn_linear.clone().detach())

			else:
				z = Variable(torch.randn(self.settings.batchSize, self.settings.latent_dim)).cuda()
				labels = Variable(torch.randint(0, self.settings.nClasses, (self.settings.batchSize,))).cuda()
				z = z.contiguous()
				labels = labels.contiguous()
				images = self.generator(z, labels)
				labels_loss = Variable(torch.zeros(self.settings.batchSize,self.settings.nClasses)).cuda()

				labels_loss.scatter_(1,labels.unsqueeze(1),1.0)
		
			self.mean_list.clear()
			self.var_list.clear()

			output_teacher_batch,out_feature = self.model_teacher(images,out_feature=True)
			_,_ = self.model(images,out_feature=True)
			sim_loss = torch.zeros(1).cuda()
			if multi_class>=MERGE_PROB:
				self.collect_image(labels, out_feature.clone().detach())
				sim_loss += self.sim_loss(label=labels, image_feature=out_feature.clone().detach())

			act_loss = torch.zeros(1).cuda()
			for i in range(len(self.act_teacher_feature)):
				act_loss +=  F.mse_loss(self.act_teacher_feature[i],self.act_student_feature[i])
			act_loss = act_loss / (len(self.act_teacher_feature) + 1e-6)


			if not self.flag:
				self.Classfier.train([i.clone().detach() for i in self.block_feature],
						output_teacher_batch.clone().detach(),self.settings.temperature,self.settings.alpha)

			self.block_feature.clear()


			loss_one_hot = (-(labels_loss*self.log_soft(output_teacher_batch)).sum(dim=1)).mean() 				

			# BN statistic loss
			BNS_loss = torch.zeros(1).cuda()

			for num in range(len(self.mean_list)):
				BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
					self.var_list[num], self.teacher_running_var[num])

			BNS_loss = BNS_loss / len(self.mean_list)
			# loss of Generator
			loss_G = loss_one_hot + 0.1 * BNS_loss  + sim_loss + 0.5 * act_loss

			self.act_student_feature.clear()
			self.act_teacher_feature.clear()

			self.backward_G(loss_G)

			output, loss_S = self.forward(images.detach(), output_teacher_batch.detach(), labels,linear=labels_loss)
				
			
			if epoch>= self.settings.warmup_epochs:
				self.backward_S(loss_S)

			if multi_class<MERGE_PROB:
				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, labels=labels[:,0],
					loss=loss_S, top5_flag=True, mean_flag=True)
			else:
				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, labels=labels,
					loss=loss_S, top5_flag=True, mean_flag=True)
			
			top1_error.update(single_error, images.size(0))
			top1_loss.update(single_loss, images.size(0))
			top5_error.update(single5_error, images.size(0))
			
			end_time = time.time()
			
			gt = labels.data.cpu().numpy()
			d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1).reshape(self.settings.batchSize,-1)== gt.reshape(self.settings.batchSize,-1))

			fp_acc.update(d_acc)

		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f] [S loss: %f] [Sim_loss %f] [Act loss %f]"
			% (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(), BNS_loss.item(), 
			loss_S.item(),sim_loss.item(),act_loss.item())
		)
		# print(f"cos sim {np.mean(sims,axis=0)}")

		return top1_error.avg, top1_loss.avg, top5_error.avg
	
	def test_student(self):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.cuda()
				images = images.cuda()
				output = self.model(images)

				loss = torch.ones(1)
				self.mean_list.clear()
				self.var_list.clear()
				self.block_feature.clear()
				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
		print()
		print(
			"Student Model Accuracy : %.4f%%"
			% (100.00-top1_error.avg))
		

	def test(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.cuda()
				images = images.cuda()
				output = self.model(images)

				loss = torch.ones(1)
				self.mean_list.clear()
				self.var_list.clear()
				self.block_feature.clear()
				self.act_student_feature.clear()
				self.act_teacher_feature.clear()
				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
		
		print(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
		)
		
		self.scalar_info['testing_top1error'] = top1_error.avg
		self.scalar_info['testing_top5error'] = top5_error.avg
		self.scalar_info['testing_loss'] = top1_loss.avg
		if self.tensorboard_logger is not None:
			for tag, value in self.scalar_info.items():
				self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
			self.scalar_info = {}
		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg

	def test_teacher(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				data_time = start_time - end_time

				labels = labels.cuda()
				if self.settings.tenCrop:
					image_size = images.size()
					images = images.view(
						image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
					images_tuple = images.split(image_size[0])
					output = None
					for img in images_tuple:
						if self.settings.nGPU == 1:
							img = img.cuda()
						img_var = Variable(img, volatile=True)
						temp_output, _ = self.forward(img_var)
						if output is None:
							output = temp_output.data
						else:
							output = torch.cat((output, temp_output.data))
					single_error, single_loss, single5_error = utils.compute_tencrop(
						outputs=output, labels=labels)
				else:
					if self.settings.nGPU == 1:
						images = images.cuda()

					output = self.model_teacher(images)

					loss = torch.ones(1)
					self.mean_list.clear()
					self.var_list.clear()
					self.block_feature.clear()
					self.act_student_feature.clear()
					self.act_teacher_feature.clear()

					single_error, single_loss, single5_error = utils.compute_singlecrop(
						outputs=output, loss=loss,
						labels=labels, top5_flag=True, mean_flag=True)
				#
				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))

				end_time = time.time()
				iter_time = end_time - start_time

		print(
				"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
		)

		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg


	def test_middle(self, epoch):
		"""
		testing
		"""
		top1_error_list = [utils.AverageMeter()]*3
		top1_loss_list = [utils.AverageMeter()]*3 
		top5_error_list = [utils.AverageMeter()]*3 

		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				data_time = start_time - end_time

				labels = labels.cuda()
				if self.settings.tenCrop:
					image_size = images.size()
					images = images.view(
						image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
					images_tuple = images.split(image_size[0])
					output = None
					for img in images_tuple:
						if self.settings.nGPU == 1:
							img = img.cuda()
						img_var = Variable(img, volatile=True)
						temp_output, _ = self.forward(img_var)
						if output is None:
							output = temp_output.data
						else:
							output = torch.cat((output, temp_output.data))
					single_error, single_loss, single5_error = utils.compute_tencrop(
						outputs=output, labels=labels)
				else:
					if self.settings.nGPU == 1:
						images = images.cuda()

					output = self.model_teacher(images)

					loss = torch.ones(1)
					self.mean_list.clear()
					self.var_list.clear()
					self.block_feature.clear()
					self.act_student_feature.clear()
					self.act_teacher_feature.clear()
					for i in range(len(self.block_feature)):
						single_error, single_loss, single5_error = utils.compute_singlecrop(
							outputs=self.block_feature[i], loss=loss,
							labels=labels, top5_flag=True, mean_flag=True)
						top1_error_list[i].update(single_error, images.size(0))
						top1_loss_list[i].update(single_loss, images.size(0))
						top5_error_list[i].update(single5_error, images.size(0))

				end_time = time.time()
				iter_time = end_time - start_time
		for i in range(3):
			print(
					"feature_map %d : [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
					% (i,epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error_list[i].avg))
			)

		self.run_count += 1

		return top1_error_list, top1_loss_list, top5_error_list
