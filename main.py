import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import random
import timm
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import vit_b_16,ViT_B_16_Weights
from torch import pca_lowrank
import wandb 


# option file should be modified according to your expriment
from options import Option

from dataloader import DataLoader
from trainer import Trainer

import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d

from einops import rearrange,repeat
from torch import einsum


def get_formatted_date_time():
	now = datetime.datetime.now()
	formatted_date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
	return formatted_date_time

def exists(val):
	return val is not None

def default(val, d):
	if exists(val):
		return val
	return d() if callable(d) else d

class CrossAttention(nn.Module):
	def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
		super().__init__()
		inner_dim = dim_head * heads
		context_dim = default(context_dim, query_dim)

		self.scale = dim_head ** -0.5
		self.heads = heads

		self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
		self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
		self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, query_dim),
			nn.Dropout(dropout)
		)

	def forward(self, x, context=None, mask=None,out_attn=False):
		h = self.heads

		q = self.to_q(x)
		context = default(context, x)
		k = self.to_k(context)
		v = self.to_v(context)

		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

		sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

		if exists(mask):
			mask = rearrange(mask, 'b ... -> b (...)')
			max_neg_value = -torch.finfo(sim.dtype).max
			mask = repeat(mask, 'b j -> (b h) () j', h=h)
			sim.masked_fill_(~mask, max_neg_value)

		# attention, what we cannot get enough of
		attn = sim.softmax(dim=-1)

		out = einsum('b i j, b j d -> b i d', attn, v)
		out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
		if out_attn:
			return self.to_out(out),torch.mean(attn,dim=1)
		return self.to_out(out)
class GEGLU(nn.Module):
	def __init__(self, dim_in, dim_out):
		super().__init__()
		self.proj = nn.Linear(dim_in, dim_out * 2)

	def forward(self, x):
		x, gate = self.proj(x).chunk(2, dim=-1)
		return x * F.gelu(gate)
class FeedForward(nn.Module):
	def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
		super().__init__()
		inner_dim = int(dim * mult)
		dim_out = default(dim_out, dim)
		project_in = nn.Sequential(
			nn.Linear(dim, inner_dim),
			nn.GELU()
		) if not glu else GEGLU(dim, inner_dim)

		self.net = nn.Sequential(
			project_in,
			nn.Dropout(dropout),
			nn.Linear(inner_dim, dim_out)
		)

	def forward(self, x):
		return self.net(x)
class BasicTransformerBlock(nn.Module):
	def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True,num_class=250):
		super().__init__()
		self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
		self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
		self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
									heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
		self.norm1 = nn.BatchNorm1d(num_class)
		self.norm2 = nn.BatchNorm1d(num_class)
		self.norm3 = nn.BatchNorm1d(num_class)


	def forward(self, x, context=None,out_attn=False):
		# attn_map,attn_score = self.attn1(x, context=context,out_attn=True)
		x = self.attn1(self.norm1(x)) + x
		attn_map,attn_score = self.attn2(self.norm2(x), context=context,out_attn=True)
		x = attn_map + x
		x = self.ff(self.norm3(x)) + x
		if out_attn:
			return x,attn_score
		return x

class Attention(nn.Module):
	def __init__(self, dim=512, heads = 4, dim_head = 128):
		super().__init__()
		self.scale = dim_head ** -0.5
		self.heads = heads
		self.to_qkv = nn.Linear(dim, dim_head * 3 * heads)

	def forward(self, x):
		# x shape batch,length,hidden_dim

		b, n, embed_dim = x.shape
		qkv = self.to_qkv(x).chunk(3, dim = 2) # b,hidden_dim *3 
		q, k, v = map(lambda t: rearrange(t, 'b r (d h) -> b h r d', h = self.heads), qkv)

		# q shape batch,heads,length,hidden_dim
		# k shape batch,heads,length,hidden_dim
		# v shape batch,heads,length,hidden_dim


		q = q * self.scale

		sim = einsum('b h l d, b h k d -> b h l k', q, k)
		attn = sim.softmax(dim = -1) # b,heads,length,length V: b,heads,length,hidden_dim
	
		out = einsum('b h q k, b h k d -> b h q d', attn, v)

		out = rearrange(out, 'b h l  d -> b l (h d)')
		return out  # b,length,hidden_dim

class Residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x):
		return x + self.fn(x)

class Generator_imagenet(nn.Module):
	def __init__(self, options=None, conf_path=None, teacher_weight=None, freeze=True):
		self.settings = options or Option(conf_path)

		super(Generator_imagenet, self).__init__()

		self.settings = options or Option(conf_path)
		if teacher_weight==None:
			self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
		else:
			self.label_emb = nn.Embedding.from_pretrained(teacher_weight, freeze=freeze)

		self.embed_normalizer = nn.BatchNorm1d(self.label_emb.weight.T.shape,affine=False,track_running_stats=False)
		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)
		self.scale = nn.Parameter(torch.tensor(0.))
		self.size = [2,4,6,8]
		self.attn = BasicTransformerBlock(dim=self.settings.latent_dim,n_heads=4,d_head=self.settings.latent_dim,context_dim=self.settings.latent_dim,dropout=0.3,num_class=self.settings.multi_label_num)
		self.mapping = nn.ModuleList([nn.Linear(self.settings.latent_dim,self.settings.latent_dim,bias=False) for i in range(4)])
	def forward(self, z, labels, linear=None):
		for l in self.mapping:
			z = l(z)
		if linear == None:
			gen_input = self.embed_normalizer(torch.add(self.label_emb(labels),z).T).T
		else:
			embed_norm = self.embed_normalizer(torch.add(self.label_emb(labels),z).T).T # 16,250,512 
			gen_input = embed_norm
			x = (gen_input * linear.unsqueeze(2)).sum(dim=1)
			latent_img = gen_input * linear.unsqueeze(2)
			size = 2
			for i in range(size):
				latent_img,attn_score = self.attn(latent_img,embed_norm,out_attn=True)
				attn_linear = rearrange(attn_score,'(b h) q -> b h q',h=4).mean(dim=1)
			gen_input = self.scale * latent_img.sum(dim=1) 
		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels, linear=linear) # 16,128,56,56
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels, linear=linear)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels, linear=linear)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		if linear == None:
			return img
		else:
			if torch.rand(1)<0.5:
				return img,linear
			return img,attn_linear


class ExperimentDesign:
	def __init__(self, generator=None, options=None, conf_path=None):
		self.settings = options or Option(conf_path)
		self.generator = generator
		self.train_loader = None
		self.test_loader = None
		self.model = None
		self.model_teacher = None
		
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0
		self.test_input = None

		self.unfreeze_Flag = True
		
		os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" 
		os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices 
		
		self.settings.set_save_path()
		self.logger = self.set_logger()
		self.settings.paramscheck(self.logger)

		self.prepare()
	
	def set_logger(self):
		logger = logging.getLogger('baseline')
		file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
		console_formatter = logging.Formatter('%(message)s')
		# file log
		file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
		file_handler.setFormatter(file_formatter)
		
		# console log
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(console_formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		
		logger.setLevel(logging.INFO)
		return logger

	def prepare(self):
		self._set_gpu()
		self._set_dataloader()
		self._set_model()
		self._replace()
		self.logger.info(self.model)
		self._set_trainer()
	
	def _set_gpu(self):
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
		cudnn.benchmark = True

	def _set_dataloader(self):
		# create data loader
		data_loader = DataLoader(dataset=self.settings.dataset,
								batch_size=self.settings.batchSize,
								data_path=self.settings.dataPath,
								n_threads=self.settings.nThreads,
								ten_crop=self.settings.tenCrop,
								logger=self.logger)
		
		self.train_loader, self.test_loader = data_loader.getloader()

	def _set_model(self):
		if self.settings.dataset in ["cifar100","cifar10"]:
			if self.settings.network in ["resnet20_cifar100","resnet20_cifar10"]:
				self.test_input = Variable(torch.randn(1, 3, 32, 32).cuda())
				self.model = ptcv_get_model(self.settings.network, pretrained=True)
				self.model_teacher = ptcv_get_model(self.settings.network, pretrained=True)
				self.model_teacher.eval()
			else:
				assert False, "unsupport network: " + self.settings.network

		elif self.settings.dataset in ["imagenet"]:
			if self.settings.network in ["resnet18","resnet50","mobilenetv2_w1"]:
				self.test_input = Variable(torch.randn(1, 3, 224, 224).cuda())
				self.model = ptcv_get_model(self.settings.network, pretrained=True)
				self.model_teacher = ptcv_get_model(self.settings.network, pretrained=True)
				self.model_teacher.eval()
			elif self.settings.network in ["vit_b_16"]:
				self.model = timm.create_model('vit_base_patch16_224',checkpoint_path='/root/ckpt/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors')
				self.model_teacher = timm.create_model('vit_base_patch16_224',checkpoint_path='/root/ckpt/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors')
				# self.model = timm.create_model('deit_tiny_patch16_224',checkpoint_path='/root/ckpt/deit_tiny_patch16_224.fb_in1k/model.safetensors')
				# self.model_teacher = timm.create_model('deit_tiny_patch16_224',checkpoint_path='/root/ckpt/deit_tiny_patch16_224.fb_in1k/model.safetensors')
				# self.model = timm.create_model('deit_base_patch16_224',checkpoint_path='/root/ckpt/deit_base_patch16_224.fb_in1k/model.safetensors')
				# self.model_teacher = timm.create_model('deit_base_patch16_224',checkpoint_path='/root/ckpt/deit_base_patch16_224.fb_in1k/model.safetensors')
				self.model_teacher.eval()
			else:
				assert False, "unsupport network: " + self.settings.network

		else:
			assert False, "unsupport data set: " + self.settings.dataset

	def _set_trainer(self):
		# set lr master
		lr_master_S = utils.LRPolicy(self.settings.lr_S,
								self.settings.nEpochs,
								self.settings.lrPolicy_S)
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									self.settings.nEpochs,
									self.settings.lrPolicy_G)

		params_dict_S = {
			'step': self.settings.step_S,
			'decay_rate': self.settings.decayRate_S
		}

		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}
		
		lr_master_S.set_params(params_dict=params_dict_S)
		lr_master_G.set_params(params_dict=params_dict_G)

		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			generator = self.generator,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			lr_master_S=lr_master_S,
			lr_master_G=lr_master_G,
			settings=self.settings,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			run_count=self.start_epoch)

	def quantize_model(self,model):
		"""
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
		
		weight_bit = self.settings.qw
		act_bit = self.settings.qa
		
		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		
		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6 or type(model) == nn.GELU :
			return nn.Sequential(*[model, QuantAct2(activation_bit=act_bit)])
		elif type(model) == nn.GELU:
			return nn.Sequential(*[model,QuantAct2(activation_bit=act_bit)])
		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model(m))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model(mod))
			return q_model
	
	def _replace(self):
		self.model = self.quantize_model(self.model)
	
	def freeze_model(self,model):
		"""
		freeze the activation range
		"""
		if type(model) == QuantAct2:
			model.fix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.freeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.freeze_model(mod)
			return model
	
	def unfreeze_model(self,model):
		"""
		unfreeze the activation range
		"""
		if type(model) == QuantAct2:
			model.unfix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.unfreeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.unfreeze_model(mod)
			return model
	def initial_act_range(self,model):
		if type(model) == QuantAct2:
			model.initial_clip_val()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.initial_act_range(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.initial_act_range(mod)
			return model
	def extracted_embedding(self):
		print(f"=============== extract embeding and initalize   ")
		embeds = self.trainer.Classfier.extract_embedding()

		self.trainer.Classfier.fix()
		embeds.append(self.trainer.generator.label_emb.weight.data.clone().detach())
		
		avg_em = torch.mean(torch.stack(embeds),dim=0)
		# self.trainer.generator.label_emb = self.trainer.generator.label_emb.from_pretrained(avg_em,freeze=False)
		self.trainer.generator.label_emb.weight.data = avg_em
		self.trainer.generator.label_emb.weight.requires_grad = True
		self.trainer.flag = True

	def run(self):
		best_top1 = 100
		best_top5 = 100
		start_time = time.time()

		test_error, test_loss, test5_error = self.trainer.test_teacher(0)
		best_ep = 0

		try:
			for epoch in range(self.start_epoch, self.settings.nEpochs):
				self.epoch = epoch
				self.start_epoch = 0

				if epoch < 4:
					print ("\n self.unfreeze_model(self.model)\n")
					self.unfreeze_model(self.model)
				train_error, train_loss, train5_error = self.trainer.train(epoch=epoch)
				self.freeze_model(self.model)
				if epoch == 4:
					print("==========>innitial clip range  ")
					self.initial_act_range(self.model)
				if epoch == (self.settings.warmup_epochs // 5) and self.settings.dataset  in ["imagenet"]:
					self.extracted_embedding()
					
				if self.settings.dataset in ["cifar100","cifar10"]:
					test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
				elif self.settings.dataset in ["imagenet"]:
					if epoch > self.settings.warmup_epochs - 2:
						test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
					else:
						test_error = 100
						test5_error = 100
				else:
					assert False, "invalid data set"
				if best_top1 >= test_error:
					best_ep = epoch+1
					best_top1 = test_error
					best_top5 = test5_error
					print('Saving a best checkpoint ...')
					torch.save(self.trainer.model.state_dict(),f"{self.settings.ckpt_path}/student_model_{self.settings.dataset}-{self.settings.network}-w{self.settings.qw}_a{self.settings.qa}.pt")
					torch.save(self.trainer.generator.state_dict(),f"{self.settings.ckpt_path}/generator_{self.settings.dataset}-{self.settings.network}-w{self.settings.qw}_a{self.settings.qa}.pt")
				
				self.logger.info("#==>Best Result of ep {:d} is: Top1 Error: {:f}, Top5 Error: {:f}, at ep {:d}".format(epoch+1, best_top1, best_top5, best_ep))
				self.logger.info("#==>Best Result of ep {:d} is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f} at ep {:d}".format(epoch+1 , 100 - best_top1,
																									100 - best_top5, best_ep))
				# wandb.log({"top1":100-best_top1,"top5":100-best_top5,"epoch":epoch+1})
		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		return best_top1, best_top5


def main():
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
						help='input the path of config file')
	parser.add_argument('--id', type=int, metavar='experiment_id',
						help='Experiment ID')
	parser.add_argument('--freeze', action='store_true')
	parser.add_argument('--multi_label_prob', type=float, default=0.0)
	parser.add_argument('--multi_label_num', type=int, default=2)
	parser.add_argument('--gpu', type=str, default="0")

	parser.add_argument('--randemb', default=True,type=bool)
	parser.add_argument('--no_DM', type=bool,default=True)

	parser.add_argument('--qw', type=int, default=None)
	parser.add_argument('--qa', type=int, default=None)

	parser.add_argument('--ckpt_path', type=str, default='./ckpt')

	parser.add_argument('--eval',action='store_true')




	args = parser.parse_args()
	print(args)

	os.makedirs(args.ckpt_path, exist_ok=True)
	
	option = Option(args.conf_path, args)
	option.manualSeed = args.id + 1
	option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)

	if option.dataset in ["imagenet"]:
		print(option.network)
		if option.network in ["resnet18","resnet50","mobilenetv2_w1"]:
			if option.network in ["mobilenetv2_w1"]:
				weight_t = ptcv_get_model(option.network, pretrained=True).output.weight.detach().squeeze(-1).squeeze(-1)
			else:
				weight_t = ptcv_get_model(option.network, pretrained=True).output.weight.detach()
			if args.randemb:
				weight_t = None
			generator = Generator_imagenet(option, teacher_weight=weight_t, freeze=args.freeze)

		elif option.network in ["vit_b_16"]:
			weight_t = None
			generator = Generator_imagenet(option, teacher_weight=weight_t, freeze=args.freeze)
		else:
			assert False, "unsupport network: " + option.network
	else:
		assert False, "invalid data set"

	experiment = ExperimentDesign(generator, option)
	# wandb.init(
	# 		project='resnet50',
	# 		config={
	# 			"network":option.network,
	# 			"epoch":option.nEpochs,
	# 			"id":option.manualSeed,
	# 			"multi_num":option.multi_label_num,
	# 			"multi_prob":option.multi_label_prob,
	# 			"qw":option.qw,
	# 			"qa":option.qa,
	# 			"start":get_formatted_date_time(),
	# 			"head":4,
	# 			"randEmb":False
	# 		}
	# 	)
	if args.eval:
		weight_path = f"{args.ckpt_path}/student_model_{option.dataset}-{option.network}-w{option.qw}_a{option.qa}.pt"
		experiment.trainer.model.load_state_dict(torch.load(weight_path))
		experiment.trainer.test_student()
	else:
		experiment.run()


if __name__ == '__main__':
	main()
