import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
from util.image_pool import ImagePool
from torch.autograd import Variable
import torchvision.transforms as transforms

###############################################################################
# Functions
###############################################################################

class ContentLoss():
	def initialize(self, loss):
		self.criterion = loss
			
	def get_loss(self, fakeIm, realIm):
		return self.criterion(fakeIm, realIm)

	def __call__(self, fakeIm, realIm):
		return self.get_loss(fakeIm, realIm)


def cx_loss(x, y):
	h = 0.5
	assert x.size() == y.size()
	N, C, H, W = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

	y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

	x_centered = x - y_mu
	y_centered = y - y_mu
	x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
	y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

	# The equation at the bottom of page 6 in the paper
	# Vectorized computation of cosine similarity for each pair of x_i and y_j
	x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
	y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)
	cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

	d = 1 - cosine_sim  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data
	d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H*W, 1)

	# Eq (2)
	d_tilde = d / (d_min + 1e-5)

	# Eq(3)
	w = torch.exp((1 - d_tilde) / h)

	# Eq(4)
	cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)

	# Eq (1)
	cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
	loss = torch.mean(-torch.log(cx + 1e-5))

	return loss


class PerceptualLoss():
	
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		model = model.eval()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model

	def initialize(self, loss):
		with torch.no_grad():
			self.criterion = loss
			self.contentFunc = self.contentFunc()
			self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			
	def get_loss(self, fakeIm, realIm):
		# print()
		# print(fakeIm.size(), realIm.size())

		fakeIm = (fakeIm + 1) / 2.0
		realIm = (realIm + 1) / 2.0
		fakeIm[0,:,:,:] = self.transform(fakeIm[0,:,:,:])
		realIm[0,:,:,:] = self.transform(realIm[0,:,:,:])
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		#loss = self.criterion(f_fake, f_real_no_grad) + 0.5 * nn.L1Loss()(fakeIm, realIm)
		loss = self.criterion(f_fake, f_real_no_grad)
		return torch.mean(loss)

	def __call__(self, fakeIm, realIm):
		return self.get_loss(fakeIm, realIm)


class PerceptualLosses():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		model = model.eval()
		for i, layer in enumerate(list(cnn)):
			model.add_module(str(i), layer)
			if i == conv_3_3_layer:
				break
		return model

	def initialize(self, loss):
		with torch.no_grad():
			self.criterion = loss
			self.contentFunc = self.contentFunc()

	def get_loss(self, fakeIm, realIm):
		# print()
		# print(fakeIm.size(), realIm.size())
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = 0.5 * self.criterion(f_fake, f_real_no_grad) + 0.5 * nn.L1Loss()(fakeIm, realIm) + \
			   1 * nn.MSELoss()(f_fake, f_real_no_grad)
		# loss = self.criterion(f_fake, f_real_no_grad)
		return torch.mean(loss)

	def __call__(self, fakeIm, realIm):
		return self.get_loss(fakeIm, realIm)
		
class GANLoss(nn.Module):
	def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
				 tensor=torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_l1:
			self.loss = nn.L1Loss()
		else:
			self.loss = nn.BCEWithLogitsLoss()

	def get_target_tensor(self, input, target_is_real):
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor.cuda()

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)

class DiscLoss(nn.Module):
	def name(self):
		return 'DiscLoss'

	def __init__(self):
		super(DiscLoss, self).__init__()

		self.criterionGAN = GANLoss(use_l1=False)
		self.fake_AB_pool = ImagePool(50)
		
	def get_g_loss(self,net, fakeB, realB):
		# First, G(A) should fake the discriminator
		pred_fake = net.forward(fakeB)
		return self.criterionGAN(pred_fake, 1)
		
	def get_loss(self, net, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.pred_fake = net.forward(fakeB.detach())
		self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

		# Real
		self.pred_real = net.forward(realB)
		self.loss_D_real = self.criterionGAN(self.pred_real, 1)

		# Combined loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		return self.loss_D

	def __call__(self, net, fakeB, realB):
		return self.get_loss(net, fakeB, realB)

class RelativisticDiscLoss(nn.Module):
	def name(self):
		return 'RelativisticDiscLoss'

	def __init__(self):
		super(RelativisticDiscLoss, self).__init__()

		self.criterionGAN = GANLoss(use_l1=False)
		self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
		self.real_pool = ImagePool(50)

	def get_g_loss(self, net, fakeB, realB):
		# First, G(A) should fake the discriminator
		self.pred_fake = net.forward(fakeB)

		# Real
		self.pred_real = net.forward(realB)
		errG = (self.criterionGAN(self.pred_real - torch.mean(net.forward(self.fake_B)), 0) +
					   self.criterionGAN(self.pred_fake - torch.mean(net.forward(self.real_B)), 1)) / 2
		return errG

	def get_loss(self, net, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.fake_B = self.fake_pool.query(fakeB.detach())
		self.real_B = self.real_pool.query(realB)
		self.pred_fake = net.forward(fakeB.detach())

		# Real
		self.pred_real = net.forward(realB)

		# Combined loss
		self.loss_D = (self.criterionGAN(self.pred_real - torch.mean(net.forward(self.fake_B)), 1) +
					   self.criterionGAN(self.pred_fake - torch.mean(net.forward(self.real_B)), 0)) / 2
		return self.loss_D

	def __call__(self, net, fakeB, realB):
		return self.get_loss(net, fakeB, realB)

class RelativisticDiscLossLS(nn.Module):
	def name(self):
		return 'RelativisticDiscLossLS'

	def __init__(self):
		super(RelativisticDiscLossLS, self).__init__()

		self.criterionGAN = GANLoss(use_l1=True)
		self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
		self.real_pool = ImagePool(50)

	def get_g_loss(self, net, fakeB, realB):
		# First, G(A) should fake the discriminator
		self.pred_fake = net.forward(fakeB)

		# Real
		self.pred_real = net.forward(realB)
		errG = (torch.mean((self.pred_real - torch.mean(net.forward(self.fake_B)) + 1) ** 2) +
				torch.mean((self.pred_fake - torch.mean(net.forward(self.real_B)) - 1) ** 2)) / 2
		return errG

	def get_loss(self, net, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.fake_B = self.fake_pool.query(fakeB.detach())
		self.real_B = self.real_pool.query(realB)
		self.pred_fake = net.forward(fakeB.detach())

		# Real
		self.pred_real = net.forward(realB)

		# Combined loss
		self.loss_D = (torch.mean((self.pred_real - torch.mean(net.forward(self.fake_B)) - 1) ** 2) +
					   torch.mean((self.pred_fake - torch.mean(net.forward(self.real_B)) + 1) ** 2)) / 2
		return self.loss_D

	def __call__(self, net, fakeB, realB):
		return self.get_loss(net, fakeB, realB)

class DiscLossLS(DiscLoss):
	def name(self):
		return 'DiscLossLS'

	def __init__(self):
		super(DiscLossLS, self).__init__()
		self.criterionGAN = GANLoss(use_l1=True)
		
	def get_g_loss(self,net, fakeB, realB):
		return DiscLoss.get_g_loss(self,net, fakeB, realB)
		
	def get_loss(self, net, fakeB, realB):
		return DiscLoss.get_loss(self, net, fakeB, realB)
		
class DiscLossWGANGP(DiscLossLS):
	def name(self):
		return 'DiscLossWGAN-GP'

	def __init__(self):
		super(DiscLossWGANGP, self).__init__()
		self.LAMBDA = 10
		
	def get_g_loss(self, net, fakeB, realB):
		# First, G(A) should fake the discriminator
		self.D_fake = net.forward(fakeB)
		return -self.D_fake.mean()
		
	def calc_gradient_penalty(self, netD, real_data, fake_data):
		alpha = torch.rand(1, 1)
		alpha = alpha.expand(real_data.size())
		alpha = alpha.cuda()

		interpolates = alpha * real_data + ((1 - alpha) * fake_data)

		interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)
		
		disc_interpolates = netD.forward(interpolates)

		gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
								  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
								  create_graph=True, retain_graph=True, only_inputs=True)[0]

		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
		return gradient_penalty
		
	def get_loss(self, net, fakeB, realB):
		self.D_fake = net.forward(fakeB.detach())
		self.D_fake = self.D_fake.mean()
		
		# Real
		self.D_real = net.forward(realB)
		self.D_real = self.D_real.mean()
		# Combined loss
		self.loss_D = self.D_fake - self.D_real
		gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
		return self.loss_D + gradient_penalty


def get_loss(model):

	if model['fea_loss'] == 'perceptual':
		fea_loss = PerceptualLoss()
		fea_loss.initialize(nn.MSELoss())
	# elif model['fea_loss'] == 'l1':
	# 	fea_loss = ContentLoss()
	# 	fea_loss.initialize(nn.L1Loss())
	elif model['fea_loss'] == 'cx':
		fea_loss = PerceptualLoss()
		fea_loss.initialize(cx_loss)
	elif model['fea_loss'] == 'both':
		fea_loss = PerceptualLosses()
		fea_loss.initialize(cx_loss)

	else:
		raise ValueError("ContentLoss [%s] not recognized." % model['fea_loss'])

	if model['pixel_loss'] == 'l1':
		pixel_loss = ContentLoss()
		pixel_loss.initialize(nn.L1Loss())
	elif model['pixel_loss'] == 'l2':
		pixel_loss = ContentLoss()
		pixel_loss.initialize(nn.MSELoss())
	else:
		raise ValueError("PixelLoss [%s] not recognized." % model['pixel_loss'])

	if model['disc_loss'] == 'wgan-gp':
		disc_loss = DiscLossWGANGP()
	elif model['disc_loss'] == 'lsgan':
		disc_loss = DiscLossLS()
	elif model['disc_loss'] == 'gan':
		disc_loss = DiscLoss()
	elif model['disc_loss'] == 'ragan':
		disc_loss = RelativisticDiscLoss()
	elif model['disc_loss'] == 'ragan-ls':
		disc_loss = RelativisticDiscLossLS()
	else:
		raise ValueError("GAN Loss [%s] not recognized." % model['disc_loss'])
	return fea_loss, pixel_loss, disc_loss