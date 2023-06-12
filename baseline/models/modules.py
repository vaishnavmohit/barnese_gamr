import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

# from util import log

import logging
log = logging.getLogger(__name__)

class Encoder_conv(nn.Module):
	def __init__(self):
		super(Encoder_conv, self).__init__()
		log.info('Building convolutional encoder...')
		# Convolutional layers
		# input is of shape: B, C, H, W --> [B, 1, 128, 128]
		log.info('Conv layers...')
		self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		# Fully-connected layers
		log.info('FC layers...')
		self.fc1 = nn.Linear(4*4*32, 256)
		self.fc2 = nn.Linear(256, 128)
		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Convolutional layers
		conv1_out = self.relu(self.conv1(x))
		conv2_out = self.relu(self.conv2(conv1_out))
		conv3_out = self.relu(self.conv3(conv2_out))
		conv4_out = self.relu(self.conv4(conv3_out))
		conv5_out = self.relu(self.conv5(conv4_out))
		# Flatten output of conv. net
		conv5_out_flat = torch.flatten(conv5_out, 1)
		# Fully-connected layers
		fc1_out = self.relu(self.fc1(conv5_out_flat))
		fc2_out = self.relu(self.fc2(fc1_out))
		# Output
		z = fc2_out
		return z

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels, stride=1)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class Down_s(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.stride_conv = nn.Sequential(
			# nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels, stride=2)
		)

	def forward(self, x):
		return self.stride_conv(x)


class Enc_Conv_2(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(1, 64) # 128x128x64
		self.down1 = Down(64, 64) # 64x64x64
		self.down2 = Down(64, 128) # 32x32x128
		self.down3 = Down(128, 128) # 16x16x128
		self.down4 = Down(128, 256) # 8x8x128
		self.dropout1 = nn.Dropout2d(p=.2)
		self.down5 = Down(256, 256) # 4x4x256
		self.dropout2 = nn.Dropout2d(p=.2)
		self.down6 = Down(256, 512) # 2x2x512
		self.dropout3 = nn.Dropout2d()
		self.down7 = Down(512, 512) # 1x1x512
		self.dropout4 = nn.Dropout2d()
		self.fc = nn.Linear(512, 128) # 128

	def forward(self, x):
		x = self.inc(x)
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.dropout1(self.down4(x))
		x = self.dropout2(self.down5(x))
		x = self.dropout3(self.down6(x))
		x = self.dropout4(self.down7(x))
		x = torch.flatten(x, 1)
		logits = self.fc(x)
		return logits

class Enc_Conv_fc64(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(3, 64) # 128x128x64
		self.down1 = Down(64, 64) # 64x64x64
		self.down2 = Down(64, 128) # 32x32x128
		self.down3 = Down(128, 128) # 16x16x128
		self.down4 = Down(128, 128) # 8x8x128
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.inc(x)
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.relu(self.down4(x))

		x = torch.transpose(x.view(x.shape[0], 128, 64), 2, 1) # bringing it to dimension B, 64, 128

		return x

class Enc_Conv_Mq16(nn.Module):
	def __init__(self):
		super().__init__()

		self.value_size = 128 #args.value_size
		self.key_size = 128 #args.key_size

		self.inc = DoubleConv(3, 64) # 64x128x128
		self.down0 = Down(64, 64) # 64x64x64
		self.down1 = Down(64, 64) # 64x32x32
		self.down2 = Down(64, 128) # 128x16x16
		self.down3 = Down(128, 128) # 128x8x8
		self.down4 = Down(128, 128) # 128x4x4
		self.down5 = Down(128, 128) # 128x2x2
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(128, self.value_size+self.key_size, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)
		x = self.down0(x)
		x = self.down1(x)
		x = self.down2(x)
		x = self.down3(x)
		x = self.down4(x)
		conv_out = self.down5(x)
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size+self.key_size, -1), 2, 1) # B, C, HW
		z_keys, z_values = z_kv.split([self.key_size, self.value_size], dim=2)

		return z_keys, z_values 

class Enc_Conv_Mq16v1(nn.Module):
	def __init__(self):
		super().__init__()

		self.value_size = 128 #args.value_size
		self.key_size = 128 #args.key_size

		self.inc = DoubleConv(3, 64) # 64x128x128
		self.down0 = Down(64, 64) # 64x64x64
		self.down0s = DoubleConv(64, 64) 
		self.down1 = Down(64, 64) # 64x32x32
		self.down1s = DoubleConv(64, 64)
		self.down2 = Down(64, 128) # 128x16x16
		self.down2s = DoubleConv(128, 128)
		self.down3 = Down(128, 128) # 128x8x8
		self.down3s = DoubleConv(128, 128)
		self.down4 = Down(128, 128) # 128x4x4
		self.down4s = DoubleConv(128, 256)
		self.down5 = DoubleConv(256, 256) # 128x4x4
		self.down5s = DoubleConv(256, 256)
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, self.value_size+self.key_size, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)
		
		x = self.down1s(self.down1(x))
		x = self.down2s(self.down2(x))
		x = self.down3s(self.down3(x))
		x = self.down4s(self.down4(x))
		conv_out = self.down5s(self.down5(x))
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size+self.key_size, -1), 2, 1) # B, C, HW
		z_keys, z_values = z_kv.split([self.key_size, self.value_size], dim=2)

		return z_keys, z_values 

class Enc_Conv_v0_256(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(3, 64) # 64x128x128
		# self.down0 = Down(64, 64) 
		# self.down0s = DoubleConv(64, 64) 
		self.down1 = Down(64, 64) # 64x64x64 
		self.down1s = DoubleConv(64, 64) # 64x64x64
		self.down2 = Down(64, 128) # 128x32x32
		self.down2s = DoubleConv(128, 128)
		self.down3 = Down(128, 128) # 128x16x16
		self.down3s = DoubleConv(128, 128)
		self.down4 = DoubleConv(128, 128) # 128x16x16
		self.down4s = DoubleConv(128, 256)
		self.down5 = DoubleConv(256, 256) # 128x16x16
		self.down5s = DoubleConv(256, 256)
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)
		
		x = self.down1s(self.down1(x))
		x = self.down2s(self.down2(x))
		x = self.down3s(self.down3(x))
		x = self.down4s(self.down4(x))
		conv_out = self.down5s(self.down5(x))
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], 128, -1), 2, 1) # B, C, HW

		return z_kv 

class Enc_Conv_v0_16(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(3, 64) # 64x128x128
		# self.down0 = Down(64, 64) 
		# self.down0s = DoubleConv(64, 64) 
		self.down1 = Down(64, 64) # 64x64x64 
		self.down1s = DoubleConv(64, 64) # 64x64x64
		self.down2 = Down(64, 128) # 128x32x32
		self.down2s = DoubleConv(128, 128)
		self.down3 = Down(128, 128) # 128x16x16
		self.down3s = DoubleConv(128, 128)
		self.down4 = Down(128, 128) # 128x8x8 
		self.down4s = DoubleConv(128, 256)
		self.down5 = Down(256, 256) # 128x4x4
		self.down5s = DoubleConv(256, 256)
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)
		
		x = self.down1s(self.down1(x))
		x = self.down2s(self.down2(x))
		x = self.down3s(self.down3(x))
		x = self.down4s(self.down4(x))
		conv_out = self.down5s(self.down5(x))
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], 128, -1), 2, 1) # B, C, HW

		return z_kv 

class Enc_Conv_v1_16(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(3, 64) # 64x128x128
		self.down0 = Down(64, 64) # 64x64x64
		self.down0s = DoubleConv(64, 64)
		self.down1 = Down(64, 64) # 64x32x32
		self.down1s = DoubleConv(64, 64)
		self.down2 = Down(64, 128) # 128x16x16
		self.down2s = DoubleConv(128, 128)
		self.down3 = Down(128, 128) # 128x8x8
		self.down3s = DoubleConv(128, 128)
		self.down4 = Down(128, 128) # 128x4x4
		self.down4s = DoubleConv(128, 256)
		self.down5 = Down(256, 256) # 128x2x2
		self.down5s = DoubleConv(256, 256)
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)
		x = self.down0s(self.down0(x))
		x = self.down1s(self.down1(x))
		x = self.down2s(self.down2(x))
		x = self.down3s(self.down3(x))
		x = self.down4s(self.down4(x))
		conv_out = self.down5s(self.down5(x))
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], 128, -1), 2, 1) # B, C, HW

		return z_kv 

class Enc_Conv_v1_v5a(nn.Module):
	def __init__(self):
		super().__init__()

		self.inc = DoubleConv(3, 64) # 64x128x128
		self.down0 = Down(64, 64) # 64x64x64
		self.down0s = DoubleConv(64, 64)
		self.down1 = Down(64, 64) # 64x32x32
		self.down1s = DoubleConv(64, 64)
		self.down2 = Down(64, 128) # 128x16x16
		self.down2s = DoubleConv(128, 128)
		self.down3 = Down(128, 128) # 128x8x8
		self.down3s = DoubleConv(128, 128)
		self.down4 = Down(128, 128) # 128x4x4
		self.down4s = DoubleConv(128, 256)
		self.down5 = DoubleConv(256, 256) # 128x2x2
		self.down5s = DoubleConv(256, 256)
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, 128, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x)
		
		x = self.down1s(self.down1(x))
		x = self.down2s(self.down2(x))
		x = self.down3s(self.down3(x))
		x = self.down4s(self.down4(x))
		conv_out = self.down5s(self.down5(x))
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], 128, -1), 2, 1) # B, C, HW

		return z_kv 


class Enc_Conv_Mq32(nn.Module):
	def __init__(self):
		super().__init__()

		self.value_size = 128 #args.value_size
		self.key_size = 128 #args.key_size

		self.inc = DoubleConv(3, 64) # 64x128x128
		self.double0a = DoubleConv(64, 64)
		self.double0b = DoubleConv(64, 64)
		self.down1 = Down(64, 64) # 64x64x64
		self.double1a = DoubleConv(64, 64)
		self.down2 = Down(64, 128) # 64x32x32
		self.double2a = DoubleConv(128, 128)
		self.double2b = DoubleConv(128, 128) # 64x32x32
		self.double2c = DoubleConv(128, 128) 
		self.double2d = DoubleConv(128, 128)
		self.double3a = DoubleConv(128, 256)
		self.double3b = DoubleConv(256, 256)
		self.double3c = DoubleConv(256, 256) # 256x32x32
		self.relu = nn.ReLU()

		self.conv_kv = nn.Conv2d(256, self.value_size+self.key_size, 1, stride=1, padding=0)

	def forward(self, x):
		x = self.inc(x) # 64x128x128
		
		x = self.double0b(self.double0a(x))
		x = self.down1(x) # 64x64x64
		x = self.double1a(x)
		x = self.down2(x) # 64x32x32
		x = self.double2b(self.double2a(x))
		x = self.double2d(self.double2c(x))
		x = self.double3b(self.double3a(x))
		conv_out = self.double3c(x)
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size+self.key_size, -1), 2, 1) # B, C, HW
		z_keys, z_values = z_kv.split([self.key_size, self.value_size], dim=2)

		return z_keys, z_values 


class Resnet_block(nn.Module):
	def __init__(self, pretrained = False):
		super().__init__()
		
		self.value_size = 128 #args.value_size
		self.key_size = 128 #args.key_size
		# Inputs to hidden layer linear transformation
		self.resnet = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-2])
		self.conv_kv = nn.Conv2d(2048, self.value_size+self.key_size, 1, stride=1, padding=0)
		self.relu = nn.ReLU()

		
	def forward(self, x):
		# Pass the input tensor through each of our operations
		x = self.resnet(x) # input = 1024
		z_kv = self.relu(self.conv_kv(x))		# BxCxHxW 'relu' required 
		z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size+self.key_size, -1), 2, 1) # B, C, HW
		z_keys, z_values = z_kv.split([self.key_size, self.value_size], dim=2)

		return z_keys, z_values 
			
class Encoder_Minju_fc32(nn.Module):
	def __init__(self):
		super(Encoder_Minju_fc32, self).__init__()
		# Convolutional layers
		self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) # 64x64x32
		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1) # 32x32x32
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1) # 16x16x32
		self.conv4 = nn.Conv2d(32, 32, 4, stride=2, padding=1) # 8x8x32
		self.conv5 = nn.Conv2d(32, 128, kernel_size=1) 

		# Nonlinearities
		self.relu = nn.ReLU()

		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')
	def forward(self, x):
		# Convolutional layers
		conv_out = self.relu(self.conv1(x)) # 
		conv_out = self.relu(self.conv2(conv_out)) # 
		conv_out = self.relu(self.conv3(conv_out)) # 
		conv_out = self.relu(self.conv4(conv_out)) # 

		# Output
		z = self.relu(self.conv5(conv_out))
		return torch.transpose(z.view(x.shape[0], 128, 64), 2, 1) # bringing it to dimension B, 64, 128


class Encoder_conv_Minju2(nn.Module):
	def __init__(self,):
		super().__init__()
		log.info('Building convolutional encoder...')
		# Convolutional layers
		log.info('Conv layers...')
		
		self.value_size = 128 #args.value_size
		self.key_size = 128 #args.key_size
		self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(32, 32, 4, stride=2, padding=1)

		# h, w = 8, 4
		h, w = 4, 4
		self.conv_kv = nn.Conv2d(32, self.value_size+self.key_size, 1, stride=1, padding=0)

		# Nonlinearities
		self.relu = nn.ReLU()
		# Initialize parameters
		for name, param in self.named_parameters():
			# Initialize all biases to 0
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			# Initialize all pre-ReLU weights using Kaiming normal distribution
			elif 'weight' in name:
				nn.init.kaiming_normal_(param, nonlinearity='relu')

	def forward(self, x):
		# Convolutional layers
		conv_out = self.relu(self.conv1(x))
		conv_out = self.relu(self.conv2(conv_out))
		conv_out = self.relu(self.conv3(conv_out))
		conv_out = self.relu(self.conv4(conv_out))
		z_kv = self.relu(self.conv_kv(conv_out))		# BxCxHxW
		z_kv = torch.transpose(z_kv.view(x.shape[0], self.value_size+self.key_size, -1), 2, 1) # B, C, HW
		z_keys, z_values = z_kv.split([self.key_size, self.value_size], dim=2)

		return z_keys, z_values 
