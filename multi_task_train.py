from __future__ import print_function

import torch
import torch.optim as optim
from data.data_loader import CreateDataLoader
import tqdm
import cv2
import yaml
from schedulers import WarmRestart, LinearDecay
import numpy as np
from models.networks import get_nets_multitask
from models.losses import get_loss
from models.models import get_model
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(filename='mtl1.log',level=logging.DEBUG)
writer = SummaryWriter('mtl1_runs')
REPORT_EACH = 10
torch.backends.cudnn.bencmark = True
cv2.setNumThreads(0)

class Trainer:
	def __init__(self, config):
		self.config = config
		self.train_dataset = self._get_datasets(config, 'train')
		self.val_dataset = self._get_datasets(config, 'test')
		self.best_metric = 0
		self.warmup_epochs = config['warmup_num']


	def train(self):
		self._init_params()
		for epoch in range(0, config['num_epochs']):
			if (epoch == self.warmup_epochs) and not(self.warmup_epochs == 0):
				self.netG.module.unfreeze()
				self.optimizer_G = self._get_optim(self.netG, self.config['optimizer']['lr_G'])
				self.scheduler_G = self._get_scheduler(self.optimizer_G)

			train_loss = self._run_epoch(epoch)
			val_loss, val_psnr = self._validate(epoch)
			self.scheduler_G.step()

			val_metric = val_psnr

			if val_metric > self.best_metric:
				self.best_metric = val_metric
				torch.save({
					'model': self.netG.state_dict()
				}, 'best_{}.h5'.format(self.config['experiment_desc']))
			torch.save({
				'model': self.netG.state_dict()
			}, 'last_{}.h5'.format(self.config['experiment_desc']))
			print(('val_loss={}, val_metric={}, best_metric={}\n'.format(val_loss, val_metric, self.best_metric)))
			logging.debug("Experiment Name: %s, Epoch: %d, Train Loss: %.3f, Val Accuracy: %.3f, Val Loss: %.3f, Best Loss: %.3f" % (
				self.config['experiment_desc'], epoch, train_loss, val_loss, val_metric, self.best_metric))

	def _run_epoch(self, epoch):
		self.netG = self.netG.train()
		losses_G = []
		losses_G_1 = []
		losses_G_2 = []
		losses_vgg_1 = []
		losses_vgg_2 = []
		losses_adv_1 = []
		losses_adv_2 = []
		psnrs = []
		ssim = []

		datasets = {"batches_per_epoch":[], "dataiterators":[]}
		for type, dataset in self.train_dataset.items():
			batches_per_epoch = len(dataset) // dataset.dataloader.batch_size
			datasets["batches_per_epoch"].append(batches_per_epoch)
			datasets["dataiterators"].append(iter(dataset))

		for param_group in self.optimizer_G.param_groups:
			lr = param_group['lr']
		#TODO: max/min iter tqdm
		tq = tqdm.tqdm(range(max(datasets["batches_per_epoch"])))
		tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
		#i = 0
		for i in tq:
			loss_G = 0
			for idx, dataset in enumerate(datasets["dataiterators"]):
				data = next(dataset)
				inputs, targets = self.model.get_input(data)
				if idx == 0:
					outputs, _ = self.netG(inputs)
					for _ in range(config['D_update_ratio']):
						self.optimizer_D1.zero_grad()
						loss_D1 = config['loss']['adv'] * self.criterionD(self.netD1, outputs, targets)
						loss_D1.backward(retain_graph=True)
						self.optimizer_D1.step()
					loss_adv = self.criterionD.get_g_loss(self.netD1, outputs)
					loss_content = self.criterionG(outputs, targets)
					losses_adv_1.append(loss_adv.item())
					losses_vgg_1.append(loss_content.item())
					lg1 = loss_content + config['loss']['adv'] * loss_adv
					losses_G_1.append(lg1.item())

				else:
					_, outputs = self.netG(inputs)
					for _ in range(config['D_update_ratio']):
						self.optimizer_D2.zero_grad()
						loss_D2 = config['loss']['adv'] * self.criterionD(self.netD2, outputs, targets)
						loss_D2.backward(retain_graph=True)
						self.optimizer_D2.step()
					loss_adv = self.criterionD.get_g_loss(self.netD2, outputs)
					loss_content = self.criterionG(outputs, targets)
					losses_adv_2.append(loss_adv.item())
					losses_vgg_2.append(loss_content.item())
					lg2 = loss_content + config['loss']['adv'] * loss_adv
					losses_G_2.append(lg2.item())

				self.optimizer_G.zero_grad()
				loss_G += loss_content + config['loss']['adv'] * loss_adv

			loss_G.backward()
			self.optimizer_G.step()
			#losses_G.append(loss_G.item())
				curr_psnr, curr_ssim = self.model.get_acc(outputs, targets)
				psnrs.append(curr_psnr)
				ssim.append(curr_ssim)
				mean_loss_G = np.mean(losses_G[-REPORT_EACH:])
				mean_loss_vgg = np.mean(losses_vgg[-REPORT_EACH:])
				mean_loss_adv = np.mean(losses_adv[-REPORT_EACH:])
				mean_psnr = np.mean(psnrs[-REPORT_EACH:])
				mean_ssim = np.mean(ssim[-REPORT_EACH:])
				if i % 100 == 0:
					writer.add_scalar('Train_G_Loss', mean_loss_G, i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_G_Loss_vgg', mean_loss_vgg, i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_G_Loss_adv', mean_loss_adv, i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_PSNR', mean_psnr, i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_SSIM', mean_ssim, i + (batches_per_epoch * epoch))
					writer.add_image('output', outputs)
					writer.add_image('target', targets)
					self.model.visualize_data(writer, data, outputs, i + (batches_per_epoch * epoch))
				tq.set_postfix(loss=self.model.get_loss(mean_loss_G, mean_psnr, mean_ssim, outputs, targets))
				#i += 1
			tq.close()
			return np.mean(losses_G)




	def _validate(self, epoch):
		self.netG = self.netG.eval()
		losses = []
		psnrs = []

		datasets = {"batches_per_epoch":[], "dataiterators":[]}
		for type, dataset in self.val_dataset.items():
			batches_per_epoch = len(dataset) // dataset.dataloader.batch_size
			datasets["batches_per_epoch"].append(batches_per_epoch)
			datasets["dataiterators"].append(iter(dataset))

		tq = tqdm.tqdm(range(max(datasets["batches_per_epoch"])))
		tq.set_description('Validation')
		for j in tq:
			for dataset in datasets["dataiterators"]:

				data = next(dataset)
				inputs, targets = self.model.get_input(data)
				outputs = self.netG(inputs)
				loss_content = self.criterionG(outputs, targets)
				loss_G = config['loss']['cont'] * loss_content + self.criterionD.get_g_loss(self.netD, outputs)
				losses.append(loss_G.item())
				curr_psnr = self.model.get_acc(outputs, targets)
				psnrs.append(curr_psnr)
		val_loss = np.mean(losses)
		val_psnr = np.mean(psnrs)
		tq.close()
		writer.add_scalar('Validation_Loss', val_loss, epoch)
		writer.add_scalar('Validation_PSNR', val_psnr, epoch)
		writer.add_image('output', outputs)
		writer.add_image('target', targets)
		return val_loss, val_psnr

	def _get_dataset(self, config, filename):
		data_loader = CreateDataLoader(config, filename)
		return data_loader.load_data()

	def _get_datasets(self, config, filename):
		if "datasets" not in config:
			return self._get_dataset(config, filename)
		else:
			train_dataset = {}
			for dataset in config["datasets"]:
				my_config = {"dataset": {"mode": dataset["type"]}, "batch_size": dataset["batch_size"],
							 "dataroot_train": dataset["dataroot_train"], "dataroot_val": dataset["dataroot_val"],
							 "fineSize": dataset["fineSize"], "num_workers": config["num_workers"]}
				train_dataset[dataset["type"]] = self._get_dataset(my_config, filename)
			return train_dataset



	def _get_optim(self, list_of_params, lr):
		if self.config['optimizer']['name'] == 'adam':
			optimizer = optim.Adam(filter(lambda p: p.requires_grad, list_of_params), lr=lr)
		elif self.config['optimizer']['name'] == 'sgd':
			optimizer = optim.SGD(filter(lambda p: p.requires_grad, list_of_params), lr=lr)
		elif self.config['optimizer']['name'] == 'adadelta':
			optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, list_of_params), lr=lr)
		else:
			raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
		return optimizer

	def _get_scheduler(self, optimizer):
		if self.config['scheduler']['name'] == 'plateau':
			scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
															 mode='min',
															 patience=self.config['scheduler']['patience'],
															 factor=self.config['scheduler']['factor'],
															 min_lr=self.config['scheduler']['min_lr'])
		elif self.config['optimizer']['name'] == 'sgdr':
			scheduler = WarmRestart(optimizer)
		elif self.config['scheduler']['name'] == 'linear':
			scheduler = LinearDecay(optimizer,
									min_lr=self.config['scheduler']['min_lr'],
									num_epochs=self.config['num_epochs'],
									start_epoch=self.config['scheduler']['start_epoch'])
		else:
			raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
		return scheduler

	def _init_params(self):
		dict_for_G, dict_for_D = get_nets_multitask(self.config['model'])
		self.decoder1 = dict_for_G['decoder1']
		self.decoder2 = dict_for_G['decoder2']
		self.encoder = dict_for_G['encoder']
		self.netD1 = dict_for_D['discr1']
		self.nedD2 = dict_for_D['discr2']
		self.decoder1.cuda()
		self.decoder2.cuda()
		self.encoder.cuda()
		#self.netG.cuda()
		self.netD1.cuda()
		self.netD2.cuda()
		self.model = get_model(self.config['model'])
		self.criterionG, self.criterionD = get_loss(self.config['model'])
		list_of_params = list(self.decoder1.parameters()) + list(self.decoder2.parameters()) + list(self.encoder.parameters())
		self.optimizer_G = self._get_optim(list_of_params, self.config['optimizer']['lr_G'])
		self.optimizer_D1 = self._get_optim(self.netD1.parameters(), self.config['optimizer']['lr_D'])
		self.optimizer_D2 = self._get_optim(self.netD2.parameters(), self.config['optimizer']['lr_D'])
		self.scheduler_G = self._get_scheduler(self.optimizer_G)
		self.scheduler_D1 = self._get_scheduler(self.optimizer_D1)
		self.scheduler_D2 = self._get_scheduler(self.optimizer_D2)



if __name__ == '__main__':
	with open('config/mtl_solver.yaml', 'r') as f:
		config = yaml.load(f)
	trainer = Trainer(config)
	trainer.train()


