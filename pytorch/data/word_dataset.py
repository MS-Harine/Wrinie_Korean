from torch.utils.data import Dataset
from .font2img import draw_single_char
import torch
import re

class WordImageDataset(Dataset):
	def __init__(self, string, source_font_path, target_fonts_path,
				 size=128, transform=None):
		self.string = re.sub('\s+', '', string)
		self.source_font = source_font_path
		self.target_fonts = target_fonts_path
		self.size = size
		self.transform = transform

	def __len__(self):
		return len(self.string) * len(self.target_fonts)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		font_idx = idx // len(self.string)
		ch_idx = idx % len(self.string)

		ch = self.string[ch_idx]
		source_img = draw_single_char(ch, self.source_font, self.size)
		target_img = draw_single_char(ch, self.target_fonts[font_idx], self.size)

		if self.transform:
			target_img = self.transform(target_img)

		return source_img, target_img, font_idx
	
	def get_categories(self):
		return len(self.target_fonts)

