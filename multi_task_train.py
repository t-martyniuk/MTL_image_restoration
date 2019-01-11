from __future__ import print_function

import torch
import torch.optim as optim
from data.data_loader import CreateDataLoader
import tqdm
import cv2
import yaml
from schedulers import WarmRestart, LinearDecay
import numpy as np
from models.networks import get_nets_multitask, EncoderDecoder
from models.losses import get_loss
from models.models import get_model
from tensorboardX import SummaryWriter
import logging


REPORT_EACH = 10
torch.backends.cudnn.bencmark = True
cv2.setNumThreads(0)

class Trainer:
	def __init__(self, config):
		self.config = config
		self.train_dataset = self._get_datasets(config, 'train')
		self.val_dataset = self._get_datasets(config, 'test')
		self.best_psnr = 0
		self.best_ssim = 0
		self.warmup_epochs = config['warmup_num']


	def train(self):
		self._init_params()
		for epoch in range(0, config['num_epochs']):
				# if (epoch == self.warmup_epochs) and not(self.warmup_epochs == 0):
				# 	self.netG.module.unfreeze()
				# 	self.optimizer_G = self._get_optim(self.netG, self.config['optimizer']['lr_G'])
				# 	self.scheduler_G = self._get_scheduler(self.optimizer_G)
			train_loss = self._run_epoch(epoch)

			val_loss, val_psnr, val_ssim = self._validate(epoch)
			self.scheduler_G.step()

			val_metric = val_psnr
			if val_ssim > self.best_ssim:
				self.best_ssim = val_ssim

			dict_models = {'encoder': self.encoder.state_dict()}
			for i in range(num_of_tasks):
				dict_models['decoder' + str(i)] = self.decoders[i].state_dict()

			if val_metric > self.best_psnr:
				self.best_psnr = val_metric

				torch.save(dict_models, 'best_{}.h5'.format(self.config['experiment_desc']))
			torch.save(dict_models, 'last_{}.h5'.format(self.config['experiment_desc']))
			print(('val_loss={}, val_psnr={}, val_ssim={},best_psnr={}, best_ssim={}\n'.
				   format(val_loss, val_psnr, val_ssim, self.best_psnr, self.best_ssim)))
			logging.debug("Experiment Name: %s, Epoch: %d, Train Loss: %.3f, Val Loss: %.3f, Val PSNR: %.3f, "
						  "Best PSNR: %.3f, Best SSIM: %.3f" % (self.config['experiment_desc'], epoch, train_loss,
																val_loss, val_metric, self.best_psnr, self.best_ssim))



	def _run_epoch(self, epoch):
		losses_G = []
		losses_G_i = {}
		losses_vgg_i = {}
		losses_adv_i = {}
		losses_l1_i = {}
		psnrs_i = {}
		ssims_i = {}
		mean_loss_vgg_i = {}
		mean_loss_adv_i = {}
		mean_loss_l1_i = {}
		mean_psnr_i = {}
		mean_ssim_i = {}
		mean_loss_G_i = {}

		max_len = 0
		for type, dataset in self.train_dataset.items():
			if len(dataset) > max_len:
				max_len = len(dataset)
				batches_per_epoch = len(dataset) // dataset.dataloader.batch_size

		datasets = {"dataiterators":[]}

		for param_group in self.optimizer_G.param_groups:
			lr = param_group['lr']

		mapping = {}

		for type, dataset in self.train_dataset.items():
			mapping[str(len(datasets['dataiterators']))] = dataset.dataset.name()
			datasets["dataiterators"].append(iter(dataset))
			losses_G_i[dataset.dataset.name()] = []
			losses_vgg_i[dataset.dataset.name()] = []
			losses_adv_i[dataset.dataset.name()] = []
			losses_l1_i[dataset.dataset.name()] = []
			psnrs_i[dataset.dataset.name()] = []
			ssims_i[dataset.dataset.name()] = []
		loss_di = {}

		print(mapping)

		tq = tqdm.tqdm(range(batches_per_epoch))
		tq.set_description('Epoch {}, lr {}'.format(epoch, lr))

		flag_train = False

		for i in tq:
			if flag_train:
				break
			loss_G = 0
			for idx, dataset in enumerate(datasets["dataiterators"]):
				name = mapping[str(idx)]
				try:
					data = next(dataset)
				except StopIteration:
					flag_train = True
					print('stopIter')
					break
				inputs, targets = self.model.get_input(data)
				# outputs = self.decoders[idx](self.encoder(inputs))
				outputs = inputs + self.decoders[idx](self.encoder(inputs))
				outputs = torch.clamp(outputs, min=-1, max=1)

				for _ in range(config['D_update_ratio']):
					self.optimizers_Di[idx].zero_grad()
					loss_di[name] = config['loss']['adv'] * self.criterionD(self.netsD[idx], outputs, targets)
					loss_di[name].backward(retain_graph=True)
					self.optimizers_Di[idx].step()
				loss_adv = self.criterionD.get_g_loss(self.netsD[idx], outputs)
				loss_content = self.criterionG(outputs, targets)
				loss_pix = self.criterionG_pix(outputs, targets)
				losses_adv_i[name].append(loss_adv.item())
				losses_vgg_i[name].append(loss_content.item())
				losses_l1_i[name].append(loss_pix.item())
				lg1 = loss_content + config['loss']['l1'] * loss_pix + config['loss']['adv'] * loss_adv
				losses_G_i[name].append(lg1.item())
				curr_psnr, curr_ssim = self.model.get_acc(outputs, targets)
				psnrs_i[name].append(curr_psnr)
				ssims_i[name].append(curr_ssim)
				mean_loss_vgg_i[name] = np.mean(losses_vgg_i[name][-REPORT_EACH:])
				mean_loss_adv_i[name] = np.mean(losses_adv_i[name][-REPORT_EACH:])
				mean_loss_l1_i[name] = np.mean(losses_l1_i[name][-REPORT_EACH:])
				mean_psnr_i[name] = np.mean(psnrs_i[name][-REPORT_EACH:])
				mean_ssim_i[name] = np.mean(ssims_i[name][-REPORT_EACH:])
				mean_loss_G_i[name] = np.mean(losses_G_i[name][-REPORT_EACH:])
				if i % 200 == 0:
					self.model.visualize_data('train', self.config, data, outputs, i + (batches_per_epoch * epoch),
											  name)

				self.optimizer_G.zero_grad()
				loss_G += lg1


			loss_G.backward()
			self.optimizer_G.step()
			losses_G.append(loss_G.item())

			mean_loss_G = np.mean(losses_G[-REPORT_EACH:])

			if i % 100 == 0:
				writer.add_scalar('Train_G_Loss', mean_loss_G, i + (batches_per_epoch * epoch))
				for name in mean_loss_G_i.keys():
					writer.add_scalar('Train_G_Loss_' + name, mean_loss_G_i[name],
									  i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_G_Loss_vgg_' + name, mean_loss_vgg_i[name],
									  i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_G_Loss_adv_' + name, mean_loss_adv_i[name],
									  i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_G_Loss_L1_' + name, mean_loss_l1_i[name],
									  i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_PSNR_' + name, mean_psnr_i[name],
									  i + (batches_per_epoch * epoch))
					writer.add_scalar('Train_SSIM_' + name, mean_ssim_i[name],
									  i + (batches_per_epoch * epoch))

			tq.set_postfix(loss=self.model.get_loss(mean_loss_G,
													np.mean(list(mean_psnr_i.values())),
													np.mean(list(mean_ssim_i.values()))))
		tq.close()
		return np.mean(losses_G)


	def _validate(self, epoch):

		losses_G = []
		losses_G_i = {}
		psnrs_i = {}
		ssims_i = {}
		val_psnr = {}
		val_ssim = {}
		val_loss_G = {}

		max_len = 0
		for type, dataset in self.val_dataset.items():
			if len(dataset) > max_len:
				max_len = len(dataset)
				batches_per_epoch = len(dataset) // dataset.dataloader.batch_size

		datasets = {"dataiterators":[]}
		mapping = {}

		for type, dataset in self.val_dataset.items():
			mapping[str(len(datasets['dataiterators']))] = dataset.dataset.name()
			datasets["dataiterators"].append(iter(dataset))
			losses_G_i[dataset.dataset.name()] = []
			psnrs_i[dataset.dataset.name()] = []
			ssims_i[dataset.dataset.name()] = []

		tq = tqdm.tqdm(range(batches_per_epoch))

		print('Validation')
		tq.set_description('Validation')

		# flag_val = False

		with torch.no_grad():
			for i in tq:
				# if flag_val:
				# 	break
				loss_G = 0
				for idx, dataset in enumerate(datasets["dataiterators"]):
					name = mapping[str(idx)]
					try:
						data = next(dataset)
					except StopIteration:
						# flag_val = True
						# print('stopIterVal')
						# break
						continue
					#print("\n=========", data['A'].size())
					inputs, targets = self.model.get_input(data)
					# outputs = self.decoders[idx](self.encoder(inputs))
					outputs = inputs + self.decoders[idx](self.encoder(inputs))
					outputs = torch.clamp(outputs, min=-1, max=1)

					loss_adv = self.criterionD.get_g_loss(self.netsD[idx], outputs)
					loss_content = self.criterionG(outputs, targets)
					loss_pix = self.criterionG_pix(outputs, targets)
					lg1 = loss_content + config['loss']['l1'] * loss_pix + config['loss']['adv'] * loss_adv
					losses_G_i[name].append(lg1.item())
					curr_psnr, curr_ssim = self.model.get_acc(outputs, targets)
					psnrs_i[name].append(curr_psnr)
					ssims_i[name].append(curr_ssim)
					if i%10 == 0:
						self.model.visualize_data('val', self.config, data, outputs, i + (batches_per_epoch * epoch),
												  name)

					loss_G += lg1
					losses_G.append(loss_G.item())


		for name in losses_G_i.keys():
			print('entered losses_G_i keys()')
			val_psnr[name] = np.mean(psnrs_i[name])
			val_ssim[name] = np.mean(ssims_i[name])
			val_loss_G[name] = np.mean(losses_G_i[name])
			writer.add_scalar('Val_PSNR_' + name, val_psnr[name], epoch)
			writer.add_scalar('Val_SSIM_' + name, val_ssim[name], epoch)
			writer.add_scalar('Val_Loss_G_' + name, val_loss_G[name], epoch)

		val_loss = np.mean(losses_G)
		writer.add_scalar('Validation_Loss', val_loss, epoch)

		tq.close()

		return val_loss, np.mean(list(val_psnr.values())), np.mean(list(val_ssim.values()))

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

		dict_for_G, self.netsD = get_nets_multitask(self.config['model'], self.config)
		self.encoder = dict_for_G['encoder']
		self.decoders = dict_for_G['decoders']
		self.encoder.cuda()
		for decoder in self.decoders:
			decoder.cuda()


		for netD in self.netsD:
			netD.cuda()

		self.model = get_model(self.config['model'])
		self.criterionG, self.criterionG_pix, self.criterionD = get_loss(self.config['model'])
		list_of_params = [x for y in self.decoders for x in y.parameters()]
		list_of_params = list_of_params + list(self.encoder.parameters())
		self.optimizer_G = self._get_optim(list_of_params, self.config['optimizer']['lr_G'])
		self.optimizers_Di = [self._get_optim(x.parameters(), self.config['optimizer']['lr_D']) for x in self.netsD]
		self.scheduler_G = self._get_scheduler(self.optimizer_G)
		self.schedulers_Di = [self._get_scheduler(x) for x in self.optimizers_Di]


if __name__ == '__main__':
	with open('config/mtl_solver.yaml', 'r') as f:
		config = yaml.load(f)
	exp_desc = config['experiment_desc']
	num_of_tasks = len(config['datasets'])
	logging.basicConfig(filename=exp_desc + '.log', level=logging.DEBUG)
	writer = SummaryWriter(exp_desc + '_runs')
	trainer = Trainer(config)
	trainer.train()


