import argparse
from PIL import Image
import numpy as np
from pptx import Presentation
from pptx.util import Inches
from io import BytesIO
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import DenseNet
from albumentations import *
from albumentations.pytorch import ToTensor
import torch
import tables

class Dataset(object):
	def __init__(self, fname ,img_transform=None):
		#nothing special here, just internalizing the constructor parameters
		self.fname=fname

		self.img_transform=img_transform
		
		with tables.open_file(self.fname,'r') as db:
			self.classsizes=db.root.classsizes[:]
			self.nitems=db.root.imgs.shape[0]
		
		self.imgs = None
		self.labels = None
		
	def __getitem__(self, index):
		#opening should be done in __init__ but seems to be
		#an issue with multithreading so doing here. need to do it everytime, otherwise hdf5 crashes

		with tables.open_file(self.fname,'r') as db:
			self.imgs=db.root.imgs
			self.labels=db.root.labels
			self.fnames=db.root.filenames

			#get the requested image and mask from the pytable
			img = self.imgs[index,:,:,:]
			label = self.labels[index]
		
		
		img_new = img
		
		if self.img_transform:
			img_new = self.img_transform(image=img)['image']

		return img_new, label, img
	def __len__(self):
		return self.nitems