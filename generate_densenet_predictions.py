import argparse
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('pytable_path', type=str, help='The path to an h5 file that contains "patch", "ground_truth_label", and "fname" datasets.')	
	parser.add_argument('model_checkpoint', type=str, default=None, help='The path to a model checkpoint for the torch.load() method.')
	parser.add_argument('--patch_size', type=int, default=224, help='The width of a square patch.')
	parser.add_argument('--gpuid', type=int, default=0, help='The device id.')
	args = parser.parse_args()

	# set device 
	print(torch.cuda.get_device_properties(args.gpuid))
	torch.cuda.set_device(args.gpuid)
	device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')

	# compose image transforms
	img_transform = Compose([
		# VerticalFlip(p=.5),
		# HorizontalFlip(p=.5),
		# HueSaturationValue(hue_shift_limit=(-25,0),sat_shift_limit=0,val_shift_limit=0,p=1),
		# Rotate(p=1, border_mode=cv2.BORDER_CONSTANT,value=0),
		# RandomSizedCrop((args.patch_size,args.patch_size), args.patch_size,args.patch_size),
		CenterCrop(args.patch_size,args.patch_size),
		ToTensor()
	])

	# initialize dataset and dataloader
	dset = Dataset(args.pytable_path, img_transform)
	dloader = DataLoader(dset, batch_size=1, 
								shuffle=True, num_workers=8,pin_memory=True)


	# densenet structure params copied from:
	# https://github.com/choosehappy/PytorchDigitalPathology/blob/master/classification_lymphoma_densenet/train_densenet_albumentations.py
	num_classes=3    #number of classes in the data mask that we'll aim to predict
	in_channels= 3  #input channel of the data, RGB = 3

	growth_rate=32 
	block_config=(2, 2, 2, 2)
	num_init_features=64
	bn_size=4
	drop_rate=0

	# initialize DenseNet model
	model = DenseNet(growth_rate=growth_rate, block_config=block_config,
					num_init_features=num_init_features, 
					bn_size=bn_size, 
					drop_rate=drop_rate, 
					num_classes=num_classes).to(device)

	checkpoint = torch.load(args.model_checkpoint)
	model.load_state_dict(checkpoint["model_dict"])
	model.eval()
	model.to(device)
	print(model)
	print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
	
	# generate predictions
	dtype = tables.UInt8Atom()
	predictions = []
	for img, _, _ in tqdm(dloader):
		img = img.to(device)
		pred = model(img)
		p=pred.detach().cpu().numpy()
		predflat=np.argmax(p,axis=1).flatten()
		predictions.append(predflat)
		
	# save predictions to h5
	with tables.open_file(args.pytable_path, 'a') as f:
		if 'predictions' in f.root:
			predictions_dataset = f.root.predictions
		else:
			predictions_dataset = f.create_carray(f.root, "predictions", dtype, np.array(predictions).shape)

		predictions_dataset[:] = predictions
	
	print(f'Predictions have been saved to {args.pytable_path}')

	