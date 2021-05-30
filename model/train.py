import os
import glob
import time
import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from model.models import Generator, Discriminator
from data.word_dataset import WordImageDataset

class Trainer:
	def __init__(self, dataset, embedding_dir=None, model_dir=None):
		self.dataset = dataset
		self.model_dir = model_dir
		self.embedding_dir = embedding_dir
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.embedding_list = glob.glob(os.path.join(embedding_dir, '*.style.pt')) if embedding_dir is not None else None
	
	def train(self, freeze_encoder=False, restore_files=None, save_model_dir=None, save_img_path=None,
			  epochs=10, batch_size=4, log_step=100, sample_step=1000, model_save_epoch=5, **kwargs):
		# Data Loader
		data_loader = DataLoader(self.dataset, batch_size, **kwargs)
		
		# Create Models
		generator = Generator().to(self.device)
		discriminator = Discriminator(category_num=self.dataset.get_categories()).to(self.device)
		
		if self.model_dir:
			# [generator_filename, discriminator_filename] in restore_files parameter
			generator_path, discriminator_path = restore_files
			generator.load_state_dict(torch.load(os.path.join(self.model_dir, generator_path)))
			discriminator.load_state_dict(torch.load(os.path.join(self.model_dir, discriminator_path)))
		
		# Losses
		l1_criterion = nn.L1Loss(size_average=True).to(self.device)
		bce_criterion = nn.BCEWithLogitsLoss(size_average=True).to(self.device)
		mse_criterion = nn.MSELoss(size_average=True).to(self.device)
		
		# Optimizer
		if freeze_encoder:
			g_parameter = list(generator.decoder.parameters())
		else:
			g_parameter = list(generator.encoder.parameters()) + list(generator.decoder.parameters())
		g_optimizer = Adam(g_parameter, betas=(0.5, 0.999))
		d_optimizer = Adam(discriminator.parameters(), betas=(0.5, 0.999))
		
		# Training
		l1_losses, const_losses, category_losses, d_losses, g_losses = list(), list(), list(), list(), list()
		step = 0
		for epoch in range(epochs):
			for i, data in enumerate(data_loader):
				source, real_target, style_idx = data[0].to(self.device), data[1].to(self.device), data[2]
				style = torch.load(self.embedding_list[style_idx]).to(self.device) if self.embedding_dir else None
				
				# Forward Propagation
				# Generate fake image
				fake_target, encoded_source, _, style_vec = generator(source, style)

				# Scoring with Discriminator
				real_TS = torch.cat([source, real_target], dim=1)
				fake_TS = torch.cat([source, fake_target], dim=1)
				
				real_score, real_score_logit, real_cat_logit = discriminator(real_TS)
				fake_score, fake_score_logit, fake_cat_logit = discriminator(fake_TS)				
				
				
				# Get Losses
				# Calculate constant loss
				encoded_fake = generator.encoder(fake_target)[0]
				const_loss = mse_criterion(encoded_source, encoded_fake)

				# Calculate category loss
				real_category = torch.from_numpy(np.eye(self.dataset.get_categories())[style_idx]).float().to(self.device)
				
				real_category_loss = bce_criterion(real_cat_logit, real_category)
				fake_category_loss = bce_criterion(fake_cat_logit, real_category)
				category_loss = 0.5 * (real_category_loss + fake_category_loss)

				# Calculate binary loss
				one_labels = torch.ones([batch_size, 1]).to(self.device)
				zero_labels = torch.zeros([batch_size, 1]).to(self.device)
				
				real_binary_loss = bce_criterion(real_score_logit, one_labels)
				fake_binary_loss = bce_criterion(fake_score_logit, zero_labels)
				binary_loss = real_binary_loss + fake_binary_loss

				# L1 loss (real and fake image)
				l1_loss = l1_criterion(real_target, fake_target)
				
				# Cheat loss (for generator to fool discriminator)
				cheat_loss = bce_criterion(fake_score_logit, one_labels)

				# Summary generator loss and Discriminator loss
				g_loss = cheat_loss + l1_loss + fake_category_loss + const_loss
				d_loss = binary_loss + category_loss
				
				
				# Back Propagation
				discriminator.zero_grad()
				d_loss.backward(retain_graph=True)
				d_optimizer.step()

				generator.zero_grad()
				g_loss.backward(retain_graph=True)
				g_optimizer.step()

				# History loss
				l1_losses.append(int(l1_loss.data))
				const_losses.append(int(const_loss.data))
				category_losses.append(int(category_loss.data))
				d_losses.append(int(d_loss.data))
				g_losses.append(int(g_loss.data))

				# Logging
				if (i + 1) % log_step == 0:
					time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
					log_format = 'Epoch [%d/%d], step [%d/%d], l1_loss: %.4f, d_loss: %.4f, g_loss: %.4f' % \
								 (epoch + 1, epochs, i + 1, len(data_loader), l1_loss.item(), d_loss.item(), g_loss.item())
					print(time_stamp, log_format)
				
				# Save image
				if (i + 1) % sample_step == 0 and save_img_path:
					with torch.no_grad():
						fixed_source = self.dataset[0][0].to(self.device)
						fixed_fake_image = generator(fixed_source)[0]
						save_image(((fixed_fake_image + 1) / 2).clamp(0, 1), \
								   os.path.join(save_img_path, 'fake_samples-%d-%d.png' % \
															   (epoch + 1, i + 1)))

			# Save model
			if (epoch + 1) % model_save_epoch == 0 and save_model_dir:
				now = datetime.datetime.now()
				now_date = now.strftime("%m%d")
				now_time = now.strftime('%H:%M')
				
				model_dir = os.path.join(save_model_dir, '%d-%s-%s' % (epoch + 1, now_date, now_time))
				if not os.path.exists(model_dir):
					os.makedirs(model_dir, exist_ok=True)

				torch.save(generator.encoder.state_dict(), os.path.join(model_dir, '1_Encoder.pkl'))
				torch.save(generator.decoder.state_dict(), os.path.join(model_dir, '2_Decoder.pkl'))
				torch.save(discriminator.state_dict(), os.path.join(model_dir, '3_Discriminator.pkl'))

		# Save model
		end = datetime.datetime.now()
		end_date = end.strftime("%m%d")
		end_time = end.strftime('%H:%M')

		model_dir = os.path.join(save_model_dir, '%d-%s-%s' % (epochs, end_date, end_time))
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		
		torch.save(generator.encoder.state_dict(), os.path.join(model_dir, '1_Encoder.pkl'))
		torch.save(generator.decoder.state_dict(), os.path.join(model_dir, '2_Decoder.pkl'))
		torch.save(discriminator.state_dict(), os.path.join(model_dir, '3_Discriminator.pkl'))

		# Save losses
		losses = [l1_losses, const_losses, category_losses, d_losses, g_losses]
		torch.save(losses, os.path.join(save_model_dir, '%d-losses.pkl' % epochs))

		return losses
