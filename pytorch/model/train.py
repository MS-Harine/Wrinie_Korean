import os
import sys
import glob
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.models import Generator, Discriminator, Encoder, Decoder, StyleVector
from data.word_dataset import WordImageDataset

class Trainer:
	def __init__(self, dataset, embedding_dir=None, model_dir=None):
		self.dataset = dataset
		self.model_dir = model_dir
		self.embedding_dir = embedding_dir
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		if embedding_dir is None:
			self.style_vec = StyleVector(dataset.get_categories())
		else:
			self.style_vec = torch.load(os.path.join(embedding_dir, '4_StyleVec.pt'))
		self.style_vec = self.style_vec.to(self.device)
	
	def train(self, freeze_encoder=False, restore_files=None, save_model_dir=None, save_img_path=None,
			  epochs=10, batch_size=4, log_step=100, sample_step=1000, model_save_epoch=5, **kwargs):
		# Data Loader
		data_loader = DataLoader(self.dataset, batch_size, **kwargs)
		
		# Create Models
		encoder = Encoder().to(self.device)
		decoder = Decoder().to(self.device)
		discriminator = Discriminator(category_num=self.dataset.get_categories()).to(self.device)
		
		if self.model_dir:
			# [encoder_filename, decoder_filename, discriminator_filename] in restore_files parameter
			encoder_path, decoder_path, discriminator_path = restore_files
			encoder.load_state_dict(torch.load(os.path.join(self.model_dir, encoder_path)))
			decoder.load_state_dict(torch.load(os.path.join(self.model_dir, decoder_path)))
			discriminator.load_state_dict(torch.load(os.path.join(self.model_dir, discriminator_path)))
		
		# Losses
		l1_criterion = nn.L1Loss(reduction='mean').to(self.device)
		bce_criterion = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
		mse_criterion = nn.MSELoss(reduction='mean').to(self.device)
		
		# Optimizer
		if freeze_encoder:
			g_parameter = list(decoder.parameters())
		else:
			g_parameter = list(encoder.parameters()) + list(decoder.parameters())
		g_optimizer = Adam(g_parameter, betas=(0.5, 0.999))
		d_optimizer = Adam(discriminator.parameters(), betas=(0.5, 0.999))
		
		# Training
		l1_losses, const_losses, category_losses, d_losses, g_losses = list(), list(), list(), list(), list()
		step = 0
		torch.autograd.set_detect_anomaly(True)
		for epoch in range(epochs):
			for i, data in enumerate(data_loader):
				source, real_target, style_idx = data[0].to(self.device), data[1].to(self.device), data[2]
				style_idx = style_idx.cpu().detach().numpy()
				
				# Forward Propagation
				# Generate fake image
				fake_target, encoded_source, _ = Generator(encoder, decoder, source, self.style_vec, style_idx)
				fake_target = fake_target.detach()

				# Scoring with Discriminator
				real_TS = torch.cat([source, real_target], dim=1)
				fake_TS = torch.cat([source, fake_target], dim=1)
				
				real_score, real_score_logit, real_cat_logit = discriminator(real_TS)
				fake_score, fake_score_logit, fake_cat_logit = discriminator(fake_TS)				
				
				
				# Get Losses
				# Calculate constant loss
				encoded_fake = encoder(fake_target)[0]
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
				encoder.zero_grad()
				decoder.zero_grad()
				g_loss.backward(retain_graph=True)
				g_optimizer.step()

				if step % 5 == 0:
					discriminator.zero_grad()
					d_loss.backward()
					d_optimizer.step()

				# History loss
				l1_losses.append(int(l1_loss.data))
				const_losses.append(int(const_loss.data))
				category_losses.append(int(category_loss.data))
				d_losses.append(int(d_loss.data))
				g_losses.append(int(g_loss.data))

				# Logging
				if (step + 1) % log_step == 0:
					time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
					log_format = 'Epoch [%d/%d], step [%d/%d], l1_loss: %.4f, d_loss: %.4f, g_loss: %.4f' % \
								 (epoch + 1, epochs, i + 1, len(data_loader), l1_loss.item(), d_loss.item(), g_loss.item())
					print(time_stamp, log_format)
				
				# Save image
				if (step + 1) % sample_step == 0 and save_img_path:
					with torch.no_grad():
						fixed_source = np.tile(self.dataset[0][0], (1, self.dataset.get_categories()))
						print(fixed_source.shape)
						fixed_source = torch.from_numpy(fixed_source).to(self.device)
						fixed_fake_image = Generator(encoder, decoder, fixed_source, self.style_vec, np.arange(self.dataset.get_categories()))[0]
						for f in range(len(fixed_fake_image)):
							save_image(((fixed_fake_image[f] + 1) / 2).clamp(0, 1), os.path.join(save_img_path, 'fake_samples_%02d-%d-%d.png' % (f, epoch + 1, i + 1)))

				step += 1

			# Save model
			if (epoch + 1) % model_save_epoch == 0 and save_model_dir:
				now = datetime.datetime.now()
				now_date = now.strftime("%m%d")
				now_time = now.strftime('%H:%M')
				
				model_dir = os.path.join(save_model_dir, '%d-%s-%s' % (epoch + 1, now_date, now_time))
				if not os.path.exists(model_dir):
					os.makedirs(model_dir, exist_ok=True)

				torch.save(encoder.state_dict(), os.path.join(model_dir, '1_Encoder.pkl'))
				torch.save(decoder.state_dict(), os.path.join(model_dir, '2_Decoder.pkl'))
				torch.save(discriminator.state_dict(), os.path.join(model_dir, '3_Discriminator.pkl'))
				torch.save(self.style_vec, os.path.join(model_dir, '4_StyleVec.pt'))

		# Save model
		end = datetime.datetime.now()
		end_date = end.strftime("%m%d")
		end_time = end.strftime('%H:%M')

		model_dir = os.path.join(save_model_dir, '%d-%s-%s' % (epochs, end_date, end_time))
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		
		torch.save(encoder.state_dict(), os.path.join(model_dir, '1_Encoder.pkl'))
		torch.save(decoder.state_dict(), os.path.join(model_dir, '2_Decoder.pkl'))
		torch.save(discriminator.state_dict(), os.path.join(model_dir, '3_Discriminator.pkl'))
		torch.save(self.style_vec, os,path.join(model_dir, '4_StyleVec.pt'))

		# Save losses
		losses = [l1_losses, const_losses, category_losses, d_losses, g_losses]
		torch.save(losses, os.path.join(save_model_dir, '%d-losses.pt' % epochs))

		return losses
