from torch.utils.data import Dataset
from .font2img import draw_single_char
import torch

class WordImageDataset(Dataset):
	def __init__(self, string, font_dir, size=128, transform=None):
		self.string = string
		self.font_dir = font_dir
		self.size = size
		self.transform = transform

	def __len__(self):
		return len(self.words)

	def __getitem(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		ch = self.string[idx]
		ch_img = draw_single_char(ch, self.font_dir, self.size)
		return ch_img

