from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from collections import OrderedDict

from baseline.models.modules import *

import torchvision

import numpy as np
import os
import logging

log = logging.getLogger(__name__)

from skimage.util import random_noise


# https://debuggercafe.com/adding-noise-to-image-data-for-deep-learning-data-augmentation/
def add_noise(img, device, mode, var=.05):
	dtype = img.dtype
	noisy = img.cpu().numpy()
	for i in range(img.shape[0]):
		noisy[i, :, :, :] = random_noise(noisy[i, :, :, :], mode=mode)  # , mean=0, var=var, clip=True)

	noisy_img = torch.tensor(noisy, dtype=dtype, device=device)

	return noisy_img


__all__ = [
	"gamr",
	"gamr_resnet34",
	"gamr_comb",
	"gamr_comb_tr",
	"gamr_comb_r34",
]


class TransformerEncoderLayer_qkv(nn.Module):

	def __init__(self, d_model=128, nhead=8, dim_feedforward=128, dropout=0.1, activation="relu", mlp=False):
		super(TransformerEncoderLayer_qkv, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		self.mlp = mlp
		self.dropout1 = nn.Dropout(dropout)
		self.norm1 = nn.LayerNorm(d_model)
		if mlp:
			self.linear1 = nn.Linear(d_model, dim_feedforward)
			self.norm2 = nn.LayerNorm(d_model)
			self.dropout2 = nn.Dropout(dropout)
			self.activation = _get_activation_fn(activation)

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerEncoderLayer_qkv, self).__setstate__(state)

	def forward(self, src, k, v, src_mask=None, src_key_padding_mask=None):
		r"""Pass the input through the encoder layer.
		Args:
			src: the sequence to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).
		Shape:
			see the docs in Transformer class.
		"""
		# import pdb; pdb.set_trace()
		src2, _ = self.self_attn(src.transpose(1, 0), k.transpose(1, 0), v.transpose(1, 0), attn_mask=src_mask,
								 key_padding_mask=src_key_padding_mask)
		src = src + self.dropout1(src2.transpose(1, 0))
		src = self.norm1(src)
		if self.mlp:
			src2 = self.activation(self.linear1(src))
			src = src + self.dropout2(src2)
			src = self.norm2(src)

		return src


def _get_activation_fn(activation):
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu

	raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class Additive_atn(nn.Module):

    def __init__(self, d_model=128):
        super(Additive_atn, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, src, k, v):
        src2 = v.repeat([1, src.shape[1], 1])
        src = src + src2
        src = self.norm1(src)

        return src
    
# 6650562
class GAMR(nn.Module):
	def __init__(self, time_step=4):
		super(GAMR, self).__init__()
		# Encoder
		log.info('Building encoder...')
		self.encoder = Enc_Conv_v0_16()
		# nn.Sequential(*(list(torchvision.models.resnet50(pretrained=True).children())[:-3]))
		time_step = time_step if time_step is not None else 4
		# LSTM and output layers
		log.info('Building LSTM and output layers...')
		self.z_size = 128
		self.hidden_size = 512
		self.lstm = nn.LSTM(self.z_size, self.hidden_size, batch_first=True)
		self.g_out = nn.Linear(self.hidden_size, self.z_size)
		self.y_out = nn.Linear(4*(self.hidden_size + 256), 4)

		# New addition
		self.query_w_out = nn.Linear(self.hidden_size, self.z_size)

		# time step:
		self.time_step = time_step

		# Transformer
		# log.info('Building transformer encoder...')
		# log.info('number of head is 4...')
		self.ga = Additive_atn(d_model=self.z_size)
		# self.ga = TransformerEncoderLayer_qkv(d_model=self.z_size, nhead=4)

		# RN module terms:
		self.g_theta_hidden = nn.Linear((self.z_size + self.z_size), 512)
		self.g_theta_out = nn.Linear(512, 256)

		# Context normalization
		self.contextnorm = True
		self.gamma1 = nn.Parameter(torch.ones(self.z_size))
		self.beta1 = nn.Parameter(torch.zeros(self.z_size))

		# Nonlinearities
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
		# Initialize parameters
		for name, param in self.named_parameters():
			# Encoder parameters have already been initialized
			if not ('encoder' in name) and not ('confidence' in name):
				# Initialize all biases to 0
				if 'bias' in name:
					nn.init.constant_(param, 0.0)
				else:
					if 'lstm' in name:
						# Initialize gate weights (followed by sigmoid) using Xavier normal distribution
						nn.init.xavier_normal_(param[:self.hidden_size * 2, :])
						nn.init.xavier_normal_(param[self.hidden_size * 3:, :])
						# Initialize input->hidden and hidden->hidden weights (followed by tanh) using Xavier normal distribution with gain =
						nn.init.xavier_normal_(param[self.hidden_size * 2:self.hidden_size * 3, :], gain=5.0 / 3.0)
					elif 'key_w' in name:
						# Initialize weights for key output layer (followed by ReLU) using Kaiming normal distribution
						nn.init.kaiming_normal_(param, nonlinearity='relu')
					elif 'query_w' in name:
						# Initialize weights for query output layer using Kaiming normal distribution
						nn.init.kaiming_normal_(param)
					elif 'g_out' in name:
						# Initialize weights for gate output layer (followed by sigmoid) using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'y_out' in name:
						# Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
						nn.init.xavier_normal_(param)
					elif 'transformer' in name:
						# Initialize attention weights using Xavier normal distribution
						if 'self_attn' in name:
							nn.init.xavier_normal_(param)
						# Initialize feedforward weights (followed by ReLU) using Kaiming normal distribution
						if 'linear' in name:
							nn.init.kaiming_normal_(param, nonlinearity='relu')

	def forward(self, x_seq, device='cuda'):
					
		y_pred_linear = []

		for m in range(4):
			y_pred_linear.append(self.inner_loop(x_seq[m], device))

		y_pred_linear = torch.cat(y_pred_linear, 1)
		# import pdb; pdb.set_trace()
		y_pred_linear = self.y_out(y_pred_linear).squeeze()
		# y_pred = y_pred_linear.argmax(1)
		
		return y_pred_linear


	def inner_loop(self, x_in, device):

		z_img = self.encoder(x_in) # B, 1024, 14, 14 each
		# z_img = torch.transpose(z_img.view(z_img.shape[0], z_img.shape[1], -1), 2, 1) # B, 1024, 196 each
		self.task_seg = [np.arange(z_img.shape[1])]

		# (Mohit addition)
		if self.contextnorm:
			# for keys:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				# import pdb; pdb.set_trace()
				z_seq_all_seg.append(self.apply_context_norm(z_img[:, self.task_seg[seg], :], self.gamma1, self.beta1))
			z_img = torch.cat(z_seq_all_seg, dim=1) 

		# Initialize hidden state
		hidden = torch.zeros(1, x_in.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(1, x_in.shape[0], self.hidden_size).to(device)
		# Initialize retrieved key vector
		key_r = torch.zeros(x_in.shape[0], 1, self.z_size).to(device)

		# Memory model (extra time step to process key retrieved on final time step)
		for t in range(self.time_step):

			# Controller
			# LSTM
			lstm_out, (hidden, cell_state) = self.lstm(key_r, (hidden, cell_state))

			# Key & query output layers
			query_r = self.query_w_out(lstm_out)
			g = self.relu(self.g_out(lstm_out))
			# query_r = z_img + query_r.repeat([1, z_img.shape[1], 1]) # z_img + query
			w_z = self.ga(z_img, query_r, query_r).sum(1).unsqueeze(1)  # [32, 8, 128]
			z_t = (z_img * w_z).sum(1).unsqueeze(1)

			# Read from memory
			if t == 0:
				M_v = z_t
			else:
				M_v = torch.cat([M_v, z_t], dim=1)

			key_r = g * (M_v).sum(1).unsqueeze(1)

		# Task Relation Network layer
		all_g = []
		for z1 in range(M_v.shape[1]):
			for z2 in range(M_v.shape[1]):
				g_hidden = self.relu(self.g_theta_hidden(torch.cat([M_v[:, z1, :], M_v[:, z2, :]], dim=1)))
				g_out = self.relu(self.g_theta_out(g_hidden))
				all_g.append(g_out)  # total length 4

		# Stack and sum all outputs from G_theta
		all_g = torch.stack(all_g, 1).sum(1)  # B, 256

		return torch.cat([lstm_out.squeeze(), all_g], dim=1)

	def apply_context_norm(self, z_seq, gamma, beta):
		eps = 1e-8
		z_mu = z_seq.mean(1)
		z_sigma = (z_seq.var(1) + eps).sqrt()
		z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
		z_seq = (z_seq * gamma) + beta
		return z_seq


def load_ckpt(cfg, model):
	ckpt = cfg.model.params.ckpt
	if os.path.exists(ckpt):
		checkpoint = torch.load(ckpt)
		state_dict = checkpoint["state_dict"]
		# removing module if saved in lightning mode:
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			if 'model' in k:
				name = k[6:]  # remove `model.`
				new_state_dict[name] = v
			else:
				new_state_dict[k] = v
		model.load_state_dict(new_state_dict, strict=True)
		print('loading checkpoints weights \n')
	return model


def gamr(cfg, **kwargs: Any) -> GAMR:
	model = GAMR(time_step=cfg.model.steps)
	model = load_ckpt(cfg, model)
	return model

class GAMR_resnet34(GAMR):
	def __init__(self, time_step=4):
		super(GAMR_resnet34, self).__init__(time_step)
		# Encoder
		log.info('Building encoder...')
		self.encoder = nn.Sequential(*(list(torchvision.models.resnet34(pretrained=True).children())[:-3]))
		self.z_size = 256
		self.lstm = nn.LSTM(self.z_size, self.hidden_size, batch_first=True)
		self.g_out = nn.Linear(self.hidden_size, self.z_size)
		self.y_out = nn.Linear(4*(self.hidden_size + 256), 4)

		# New addition
		self.query_w_out = nn.Linear(self.hidden_size, self.z_size)
		self.ga = Additive_atn(d_model=self.z_size)
		self.g_theta_hidden = nn.Linear((self.z_size + self.z_size), 512)
		self.gamma1 = nn.Parameter(torch.ones(self.z_size))
		self.beta1 = nn.Parameter(torch.zeros(self.z_size))

	def inner_loop(self, x_in, device):

		z_img = self.encoder(x_in) # B, 1024, 14, 14 each
		# import pdb; pdb.set_trace()
		z_img = torch.transpose(z_img.view(z_img.shape[0], z_img.shape[1], -1), 2, 1) # B, 1024, 196 each
		self.task_seg = [np.arange(z_img.shape[1])]

		# (Mohit addition)
		if self.contextnorm:
			# for keys:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				# import pdb; pdb.set_trace()
				z_seq_all_seg.append(self.apply_context_norm(z_img[:, self.task_seg[seg], :], self.gamma1, self.beta1))
			z_img = torch.cat(z_seq_all_seg, dim=1) 

		# Initialize hidden state
		hidden = torch.zeros(1, x_in.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(1, x_in.shape[0], self.hidden_size).to(device)
		# Initialize retrieved key vector
		key_r = torch.zeros(x_in.shape[0], 1, self.z_size).to(device)

		# Memory model (extra time step to process key retrieved on final time step)
		for t in range(self.time_step):

			# Controller
			# LSTM
			lstm_out, (hidden, cell_state) = self.lstm(key_r, (hidden, cell_state))

			# Key & query output layers
			query_r = self.query_w_out(lstm_out)
			g = self.relu(self.g_out(lstm_out))
			# query_r = z_img + query_r.repeat([1, z_img.shape[1], 1]) # z_img + query
			w_z = self.ga(z_img, query_r, query_r).sum(1).unsqueeze(1)  # [32, 8, 128]
			z_t = (z_img * w_z).sum(1).unsqueeze(1)

			# Read from memory
			if t == 0:
				M_v = z_t
			else:
				M_v = torch.cat([M_v, z_t], dim=1)

			key_r = g * (M_v).sum(1).unsqueeze(1)

		# Task Relation Network layer
		all_g = []
		for z1 in range(M_v.shape[1]):
			for z2 in range(M_v.shape[1]):
				g_hidden = self.relu(self.g_theta_hidden(torch.cat([M_v[:, z1, :], M_v[:, z2, :]], dim=1)))
				g_out = self.relu(self.g_theta_out(g_hidden))
				all_g.append(g_out)  # total length 4

		# Stack and sum all outputs from G_theta
		all_g = torch.stack(all_g, 1).sum(1)  # B, 256

		return torch.cat([lstm_out.squeeze(), all_g], dim=1)

def gamr_resnet34(cfg, **kwargs: Any) -> GAMR_resnet34:
	model = GAMR_resnet34(time_step=cfg.model.steps)
	model = load_ckpt(cfg, model)
	return model

class GAMR_comb(GAMR):
	def __init__(self, time_step=4):
		super(GAMR_comb, self).__init__(time_step)
		# Encoder
		log.info('Building encoder...')
		self.encoder = Enc_Conv_v0_16()
		# nn.Sequential(*(list(torchvision.models.resnet34(pretrained=True).children())[:-3]))
		self.z_size = 128
		self.lstm = nn.LSTM(self.z_size, self.hidden_size, batch_first=True)
		self.g_out = nn.Linear(self.hidden_size, self.z_size)
		self.y_out = nn.Linear(self.hidden_size + 256, 4)

		self.query_w_out = nn.Linear(self.hidden_size, self.z_size)
		self.ga = Additive_atn(d_model=self.z_size)
		self.g_theta_hidden = nn.Linear((self.z_size + self.z_size), 512)
		self.gamma1 = nn.Parameter(torch.ones(self.z_size))
		self.beta1 = nn.Parameter(torch.zeros(self.z_size))

		# positional encoding:
		self.pos_embedding = nn.Parameter(torch.randn(1, 4, self.z_size))
	
	def forward(self, x_seq, device='cuda'):
		z_temp_all = []
		for m in range(4):
			z_img_t = self.encoder(x_seq[m])
			# z_img_t = torch.transpose(z_img_t.view(z_img_t.shape[0], z_img_t.shape[1], -1), 2, 1)
			z_img_t = z_img_t + self.pos_embedding[:, m, :]
			z_temp_all.append(z_img_t) 
		
		z_img = torch.cat(z_temp_all, dim=1) # B, 4x196, 256

		self.task_seg = [np.arange(z_img.shape[1])]

		if self.contextnorm:
			# for keys:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				# import pdb; pdb.set_trace()
				z_seq_all_seg.append(self.apply_context_norm(z_img[:, self.task_seg[seg], :], \
						 self.gamma1, self.beta1))
			z_img = torch.cat(z_seq_all_seg, dim=1) 

		# Initialize hidden state
		hidden = torch.zeros(1, z_img.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(1, z_img.shape[0], self.hidden_size).to(device)
		# Initialize retrieved key vector
		key_r = torch.zeros(z_img.shape[0], 1, self.z_size).to(device)

		# Memory model (extra time step to process key retrieved on final time step)
		for t in range(self.time_step):

			# Controller
			# LSTM
			lstm_out, (hidden, cell_state) = self.lstm(key_r, (hidden, cell_state))

			# Key & query output layers
			query_r = self.query_w_out(lstm_out)
			g = self.relu(self.g_out(lstm_out))
			# query_r = z_img + query_r.repeat([1, z_img.shape[1], 1]) # z_img + query
			w_z = self.ga(z_img, query_r, query_r).sum(1).unsqueeze(1)  # [32, 8, 128]
			z_t = (z_img * w_z).sum(1).unsqueeze(1)

			# Read from memory
			if t == 0:
				M_v = z_t
			else:
				M_v = torch.cat([M_v, z_t], dim=1)

			key_r = g * (M_v).sum(1).unsqueeze(1)

		# Task Relation Network layer
		all_g = []
		for z1 in range(M_v.shape[1]):
			for z2 in range(M_v.shape[1]):
				g_hidden = self.relu(self.g_theta_hidden(torch.cat([M_v[:, z1, :], M_v[:, z2, :]], dim=1)))
				g_out = self.relu(self.g_theta_out(g_hidden))
				all_g.append(g_out)  # total length 4

		# Stack and sum all outputs from G_theta
		all_g = torch.stack(all_g, 1).sum(1)  # B, 256

		y_pred_linear = self.y_out(torch.cat([lstm_out.squeeze(), all_g], dim=1)).squeeze()
		
		return y_pred_linear 

def gamr_comb(cfg, **kwargs: Any) -> GAMR_comb:
	model = GAMR_comb(time_step=cfg.model.steps)
	model = load_ckpt(cfg, model)
	return model

class GAMR_comb_tr(GAMR_comb):
	def __init__(self, time_step=4):
		super(GAMR_comb_tr, self).__init__(time_step)

		self.ga = TransformerEncoderLayer_qkv(d_model=self.z_size, nhead=4)
	
	def forward(self, x_seq, device='cuda'):
		z_temp_all = []
		for m in range(4):
			z_img_t = self.encoder(x_seq[m])
			# z_img_t = torch.transpose(z_img_t.view(z_img_t.shape[0], z_img_t.shape[1], -1), 2, 1)
			z_img_t = z_img_t + self.pos_embedding[:, m, :]
			z_temp_all.append(z_img_t) 
		
		z_img = torch.cat(z_temp_all, dim=1) # B, 4x196, 256

		self.task_seg = [np.arange(z_img.shape[1])]

		if self.contextnorm:
			# for keys:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				# import pdb; pdb.set_trace()
				z_seq_all_seg.append(self.apply_context_norm(z_img[:, self.task_seg[seg], :], \
						 self.gamma1, self.beta1))
			z_img = torch.cat(z_seq_all_seg, dim=1) 

		# Initialize hidden state
		hidden = torch.zeros(1, z_img.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(1, z_img.shape[0], self.hidden_size).to(device)
		# Initialize retrieved key vector
		key_r = torch.zeros(z_img.shape[0], 1, self.z_size).to(device)

		# Memory model (extra time step to process key retrieved on final time step)
		for t in range(self.time_step):

			# Controller
			# LSTM
			lstm_out, (hidden, cell_state) = self.lstm(key_r, (hidden, cell_state))

			# Key & query output layers
			query_r = self.query_w_out(lstm_out)
			g = self.relu(self.g_out(lstm_out))
			query_r = z_img + query_r.repeat([1, z_img.shape[1], 1]) # z_img + query
			w_z = self.ga(z_img, query_r, query_r).sum(1).unsqueeze(1)  # [32, 8, 128]
			z_t = (z_img * w_z).sum(1).unsqueeze(1)

			# Read from memory
			if t == 0:
				M_v = z_t
			else:
				M_v = torch.cat([M_v, z_t], dim=1)

			key_r = g * (M_v).sum(1).unsqueeze(1)

		# Task Relation Network layer
		all_g = []
		for z1 in range(M_v.shape[1]):
			for z2 in range(M_v.shape[1]):
				g_hidden = self.relu(self.g_theta_hidden(torch.cat([M_v[:, z1, :], M_v[:, z2, :]], dim=1)))
				g_out = self.relu(self.g_theta_out(g_hidden))
				all_g.append(g_out)  # total length 4

		# Stack and sum all outputs from G_theta
		all_g = torch.stack(all_g, 1).sum(1)  # B, 256

		y_pred_linear = self.y_out(torch.cat([lstm_out.squeeze(), all_g], dim=1)).squeeze()
		
		return y_pred_linear 

def gamr_comb_tr(cfg, **kwargs: Any) -> GAMR_comb_tr:
	model = GAMR_comb_tr(time_step=cfg.model.steps)
	model = load_ckpt(cfg, model)
	return model

class GAMR_comb_r34(GAMR):
	def __init__(self, time_step=4):
		super(GAMR_comb_r34, self).__init__(time_step)
		# Encoder
		log.info('Building encoder...')
		self.encoder = nn.Sequential(*(list(torchvision.models.resnet34(pretrained=True).children())[:-3]))
		self.z_size = 256
		self.lstm = nn.LSTM(self.z_size, self.hidden_size, batch_first=True)
		self.g_out = nn.Linear(self.hidden_size, self.z_size)
		self.y_out = nn.Linear(self.hidden_size + 256, 4)

		self.query_w_out = nn.Linear(self.hidden_size, self.z_size)
		self.ga = Additive_atn(d_model=self.z_size)
		self.g_theta_hidden = nn.Linear((self.z_size + self.z_size), 512)
		self.gamma1 = nn.Parameter(torch.ones(self.z_size))
		self.beta1 = nn.Parameter(torch.zeros(self.z_size))

		# positional encoding:
		self.pos_embedding = nn.Parameter(torch.randn(1, 4, self.z_size))
	
	def forward(self, x_seq, device='cuda'):
		z_temp_all = []
		for m in range(4):
			z_img_t = self.encoder(x_seq[m])
			z_img_t = torch.transpose(z_img_t.view(z_img_t.shape[0], z_img_t.shape[1], -1), 2, 1)
			z_img_t = z_img_t + self.pos_embedding[:, m, :]
			z_temp_all.append(z_img_t) 
		
		z_img = torch.cat(z_temp_all, dim=1) # B, 4x196, 256

		self.task_seg = [np.arange(z_img.shape[1])]

		if self.contextnorm:
			# for keys:
			z_seq_all_seg = []
			for seg in range(len(self.task_seg)):
				# import pdb; pdb.set_trace()
				z_seq_all_seg.append(self.apply_context_norm(z_img[:, self.task_seg[seg], :], \
						 self.gamma1, self.beta1))
			z_img = torch.cat(z_seq_all_seg, dim=1) 

		# Initialize hidden state
		hidden = torch.zeros(1, z_img.shape[0], self.hidden_size).to(device)
		cell_state = torch.zeros(1, z_img.shape[0], self.hidden_size).to(device)
		# Initialize retrieved key vector
		key_r = torch.zeros(z_img.shape[0], 1, self.z_size).to(device)

		# Memory model (extra time step to process key retrieved on final time step)
		for t in range(self.time_step):

			# Controller
			# LSTM
			lstm_out, (hidden, cell_state) = self.lstm(key_r, (hidden, cell_state))

			# Key & query output layers
			query_r = self.query_w_out(lstm_out)
			g = self.relu(self.g_out(lstm_out))
			# query_r = z_img + query_r.repeat([1, z_img.shape[1], 1]) # z_img + query
			w_z = self.ga(z_img, query_r, query_r).sum(1).unsqueeze(1)  # [32, 8, 128]
			z_t = (z_img * w_z).sum(1).unsqueeze(1)

			# Read from memory
			if t == 0:
				M_v = z_t
			else:
				M_v = torch.cat([M_v, z_t], dim=1)

			key_r = g * (M_v).sum(1).unsqueeze(1)

		# Task Relation Network layer
		all_g = []
		for z1 in range(M_v.shape[1]):
			for z2 in range(M_v.shape[1]):
				g_hidden = self.relu(self.g_theta_hidden(torch.cat([M_v[:, z1, :], M_v[:, z2, :]], dim=1)))
				g_out = self.relu(self.g_theta_out(g_hidden))
				all_g.append(g_out)  # total length 4

		# Stack and sum all outputs from G_theta
		all_g = torch.stack(all_g, 1).sum(1)  # B, 256

		y_pred_linear = self.y_out(torch.cat([lstm_out.squeeze(), all_g], dim=1)).squeeze()
		
		return y_pred_linear 

def gamr_comb_r34(cfg, **kwargs: Any) -> GAMR_comb_r34:
	model = GAMR_comb_r34(time_step=cfg.model.steps)
	model = load_ckpt(cfg, model)
	return model
	
if __name__ == "__main__":
	from .utils import print_info_net

	cfg = DictConfig(
		{
			"models": {
				"nclasses": 4,
			}
		}
	)

	for net_name in __all__:
		if net_name.startswith("gamr"):
			print(net_name)
			print_info_net(globals()[net_name](cfg))
			print()