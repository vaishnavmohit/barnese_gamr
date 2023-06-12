"""
Base Pytorch Lightning classification model
"""
from typing import Any, Callable, List, Optional, Type, Union

import logging
import pdb

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics as metrics

import matplotlib.pyplot as plt
import seaborn as sn

import json
import pandas as pd

import numpy as np
from pathlib import Path
from pytorch_lightning.loggers.neptune import NeptuneLogger

from sklearn.metrics import classification_report

from pytorch_lightning.metrics import MetricCollection, Accuracy, Precision, Recall

import os

import pickle

from pytorch_lightning.metrics import Accuracy

from baseline.utils import load_obj
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import neptune.new as neptune
from neptune.new.types import File

import csv

class LitClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.lr_scheduler = cfg.scheduler
        self.lr = cfg.training.lr
        # self.weight_decay = cfg.training.weight_decay
        self.optimizer = cfg.optimizer
        self.criterion = F.cross_entropy
        self.best_val_acc1 = torch.tensor(0.0)
        self.best_val_acc3 = torch.tensor(0.0)
        self.best_val_acc5 = torch.tensor(0.0)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        return self.criterion(outputs, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        loss = self.loss(out, y)
        opt = self.optimizers()

        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            
            for pg in opt.param_groups: #opt.param_groups[0]['lr']
                pg['lr'] = lr_scale * self.lr
                
        self.log("lr", opt.param_groups[0]['lr'], prog_bar=True)

        # self.log("train_loss", loss)
        # self.log("train_acc_s", self.train_accuracy(out, y))

        acc1, acc3, acc5 = self.__accuracy(out, y, topk=(1, 3, 5))
        # [self.accuracy1(out, y), self.accuracy3(out, y), self.accuracy5(out, y)]
        self.log('train_loss', loss, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
        self.log('train_acc1', acc1, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
        self.log('train_acc3', acc3, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
        self.log('train_acc5', acc5, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)


        results = {"loss": loss, "train_acc1": acc1, 'train_acc3': acc3, 'train_acc5': acc5}

        return results

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        loss = self.loss(out, y)

        # self.log("val_loss", val_loss)
        self.log('val_loss', loss, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
        acc1, acc3, acc5 = self.__accuracy(out, y, topk=(1, 3, 5))

        results = {"loss": loss, "val_acc1": acc1, 'val_acc3': acc3, 'val_acc5': acc5}
        return results

    def validation_epoch_end(self, outputs):
        # outs is a list of whatever you returned in `validation_step`
        acc1 = torch.stack([x['val_acc1'] for x in outputs]).mean()
        acc3 = torch.stack([x['val_acc3'] for x in outputs]).mean()
        acc5 = torch.stack([x['val_acc5'] for x in outputs]).mean()
        # need to re-write the function
        if self.best_val_acc1 < acc1.cpu():
            logger.debug(f"Previous best val acc1: {self.best_val_acc1}")
            self.best_val_acc1 = acc1.cpu()
            self.best_val_acc3 = acc3.cpu()
            self.best_val_acc5 = acc5.cpu()
            logger.debug(f"New best val acc1: {self.best_val_acc1}")

        self.log('val_acc1', acc1, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_acc3', acc3, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_acc5', acc5, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        acc1, acc3, acc5 = self.__accuracy(out, y, topk=(1, 3, 5))

        results = {"test_acc1": acc1, 'test_acc3': acc3, 'test_acc5': acc5}
        return results

    def test_epoch_end(self, outputs):

        acc1 = torch.stack([x['test_acc1'] for x in outputs]).mean()
        acc3 = torch.stack([x['test_acc3'] for x in outputs]).mean()
        acc5 = torch.stack([x['test_acc5'] for x in outputs]).mean()

        self.log('test_acc1', acc1, on_step=False, prog_bar=True, on_epoch=True)
        self.log('test_acc3', acc3, on_step=False, prog_bar=True, on_epoch=True)
        self.log('test_acc5', acc5, on_step=False, prog_bar=True, on_epoch=True)

        results = {"test_acc1": acc1, 'test_acc3': acc3, 'test_acc5': acc5}
        return results

    def configure_optimizers(self):
        optimizer = load_obj(self.optimizer.class_name)(
            self.parameters(), **self.optimizer.params
        )
        '''if self.lr_scheduler.use:
            if self.lr_scheduler.name == 'LambdaLR':
                scheduler = lr_scheduler.LambdaLR(optimizer, 
                            lambda epoch: 0.1**(epoch // self.lr_scheduler.params.step))
                
            else:
                scheduler = load_obj(self.lr_scheduler.class_name)(
                    optimizer, **self.optimizer.params)
            return [optimizer] , [scheduler]
        else:'''
        return [optimizer]

    @staticmethod
    def __accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def count_params(self) -> list:
        res = []
        counter = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                count = param.numel()
                res.append(f"\tParam {name} : {count}")
                counter += count
        res.append(f"Total number trainable params : {counter}")
        return res

    def __repr__(self):
        msg = super().__repr__()
        msg += "\n".join(self.count_params())
        return msg

    # Rework the progress_bar_dict
    def get_progress_bar_dict(self) -> dict:
        # Get the running_loss
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = None
        if running_train_loss is not None:
            avg_training_loss = running_train_loss.cpu().item()
        elif self.trainer.train_loop.automatic_optimization:
            avg_training_loss = float("NaN")

        tqdm_dict = {}
        if avg_training_loss is not None:
            tqdm_dict["loss"] = f"{avg_training_loss:.3e}"
        return tqdm_dict


class LitClassifier_op(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.lr_scheduler = cfg.scheduler
        self.lr = cfg.training.lr
        self.wd_start = 1
        # self.weight_decay = cfg.training.weight_decay
        self.optimizer = cfg.optimizer
        self.criterion = F.cross_entropy
        self.num_classes = cfg.model.nclasses
        def get_scalar_metrics(num_classes: int, average: str='macro', prefix: str=''):
            default = { 'acc/top1': metrics.Accuracy(top_k=1),
                        'acc/top3': metrics.Accuracy(top_k=3),
                        'precision/top1': metrics.Precision(num_classes=num_classes, top_k=1, average=average),
                        'precision/top3': metrics.Precision(num_classes=num_classes, top_k=3, average=average),
                        'recall/top1': metrics.Recall(num_classes=num_classes, top_k=1, average=average),
                        'recall/top3': metrics.Recall(num_classes=num_classes, top_k=3, average=average),
                        'F1/top1': metrics.F1(num_classes=num_classes, average=average),
                        # 'F1/top3': metrics.F1(num_classes=num_classes, top_k=3, average=average)
                        }
            default = {prefix + str(key): val for key, val in default.items()}

            return metrics.MetricCollection(default)

        self.train_metrics = get_scalar_metrics(num_classes=cfg.model.nclasses, prefix='train_')
        self.val_metrics = get_scalar_metrics(num_classes=cfg.model.nclasses, prefix='val_')
        self.test_metrics = get_scalar_metrics(num_classes=cfg.model.nclasses, prefix='test_')
        self.softmax = nn.Softmax(dim=1)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        return self.criterion(outputs, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        loss = self.loss(out, y)

        acc1, acc3, acc5 = self.__accuracy(out, y, topk=(1, 3, 5))
        self.train_metrics(self.softmax(out), y)

        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
        self.log('train_acc1', acc1, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
        self.log('train_acc3', acc3, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)
        self.log('train_acc5', acc5, on_step=True, prog_bar=True, on_epoch=False, sync_dist=True)


        results = {"loss": loss, "train_acc1": acc1, 'train_acc3': acc3, 'train_acc5': acc5}

        return results


    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        loss = self.loss(out, y)

        acc1, acc3, acc5 = self.__accuracy(out, y, topk=(1, 3, 5))

        self.val_metrics(self.softmax(out), y)
        self.log('val_loss', loss, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        results = {"loss": loss, "val_acc1": acc1, 'val_acc3': acc3, 'val_acc5': acc5}

        return results

    def validation_epoch_end(self, outputs):
        acc1 = torch.stack([x['val_acc1'] for x in outputs]).mean()
        acc3 = torch.stack([x['val_acc3'] for x in outputs]).mean()
        acc5 = torch.stack([x['val_acc5'] for x in outputs]).mean()
        
        self.log('val_acc1', acc1, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_acc3', acc3, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_acc5', acc5, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        acc1, acc3, acc5 = self.__accuracy(out, y, topk=(1, 3, 5))
        self.test_metrics(self.softmax(out), y)
        
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        results = {"test_acc1": acc1, 'test_acc3': acc3, 'test_acc5': acc5} # accumulating output
        return results

    def test_epoch_end(self, outputs):

        acc1 = torch.stack([x['test_acc1'] for x in outputs]).mean()
        acc3 = torch.stack([x['test_acc3'] for x in outputs]).mean()
        acc5 = torch.stack([x['test_acc5'] for x in outputs]).mean()

        self.log('test_acc1', acc1, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_acc3', acc3, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_acc5', acc5, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)

   
    @staticmethod
    def __accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #     optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #     # warm up lr
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / float(500))
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.lr
    
    def configure_optimizers(self):
        optimizer = load_obj(self.optimizer.class_name)(
            self.parameters(), **self.optimizer.params
        )
        
        self.reduce_lr_on_plateau = load_obj(self.lr_scheduler.class_name)(optimizer, **self.lr_scheduler.params)

        lr_schedulers  = {'scheduler': self.reduce_lr_on_plateau, 'interval': 'epoch', 'monitor': 'val_acc1'}

        return [optimizer] , [lr_schedulers]

# importing model to be used with optuna:

class LitClassifier_optuna(pl.LightningModule):
    def __init__(self, 
                lr: float = 0.001,
                weight_decay: float = 0.0005,
                steps: int = 1500):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.cfg = OmegaConf.load(os.path.join(os.getcwd(),'config.yaml'))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train_acc1": [],
            "val_acc1": [],
            "train_loss": [],
            "val_loss": [],
        }
        self.model = load_obj(self.cfg.model.class_name)(self.cfg)
        logger.debug(f"using alphas: ")

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        return self.criterion(outputs, targets)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        acc1 = 100*self.train_accuracy(out, y)
        # [self.accuracy1(out, y), self.accuracy3(out, y), self.accuracy5(out, y)]
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
        self.log('train_acc1', acc1, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)


        results = {"loss": loss, "train_acc1": acc1}

        return results
    
    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train_acc1"].append(self.trainer.callback_metrics["train_acc1"])
        self.metric_hist["train_loss"].append(self.trainer.callback_metrics["train_loss"])

        self.log("train/acc_best", max(self.metric_hist["train_acc1"]), prog_bar=False, \
                sync_dist=True, sync_dist_op='mean')
        self.log("train/loss_best", min(self.metric_hist["train_loss"]), prog_bar=False, \
                sync_dist=True, sync_dist_op='mean')

    def validation_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        acc1 = 100*self.val_accuracy(out, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("val_acc1", acc1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        results = {"loss": loss, "val_acc1": acc1}
        return results

    def validation_epoch_end(self, outputs):
        # outs is a list of whatever you returned in `validation_step`

        self.metric_hist["val_acc1"].append(self.trainer.callback_metrics["val_acc1"])
        self.metric_hist["val_loss"].append(self.trainer.callback_metrics["val_loss"])
        
        self.log("val/acc_best", max(self.metric_hist["val_acc1"]), prog_bar=True, sync_dist=True)
        self.log("val/loss_best", min(self.metric_hist["val_loss"]), prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        acc1 = 100*self.test_accuracy(out, y) # self.__accuracy(out, y, topk=(1, 3, 5)) 
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("test_acc1", acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        results = {"loss": loss, "test_acc1": acc1}
        return results

    def test_epoch_end(self, outputs: List[Any]):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay #training.lr/wd
        )
        '''self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                mode='min',
                                                                                factor=0.2,
                                                                                patience=2,
                                                                                min_lr=1e-6,
                                                                                verbose=True
                                                                            )'''

        return [optimizer]

    
class LitClassifier_emb(pl.LightningModule):
    def __init__(self, 
                lr: float = 0.001,
                weight_decay: float = 0.0005
                ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.cfg = OmegaConf.load(os.path.join(os.getcwd(),'config.yaml'))
        self.lr_scheduler = self.cfg.scheduler

        def get_scalar_metrics(num_classes: int, average: str='macro', prefix: str=''):
            default = { 'acc_top1': metrics.Accuracy(),
                        'precision_top1': metrics.Precision(num_classes=num_classes, average=average),
                        'recall_top1': metrics.Recall(num_classes=num_classes, average=average),
                        'F1_top1': metrics.F1(num_classes=num_classes, average=average),
                        }
            default = {prefix + str(key): val for key, val in default.items()}

            return metrics.MetricCollection(default)

        self.train_metrics = get_scalar_metrics(num_classes=self.cfg.model.nclasses, prefix='train_')
        self.val_metrics = get_scalar_metrics(num_classes=self.cfg.model.nclasses, prefix='val_')
        self.test_metrics = get_scalar_metrics(num_classes=self.cfg.model.nclasses, prefix='test_')

        self.metric_hist = {
            "train_acc1": [],
            "val_acc1": [],
            "train_loss": [],
            "val_loss": [],
        }
        self.model = load_obj(self.cfg.model.class_name)(self.cfg)

    def forward(self, x: torch.Tensor, y):
        return self.model(x, y)

    def step(self, batch: Any):
        x, y = batch
        logits, loss = self.forward(x, y)
        return loss, logits, y

    def t_step(self, batch: Any):
        x, y = batch
        logits, loss = self.forward(x, y=None)
        return loss, logits, y


    def training_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        self.train_metrics(out, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        results = {"loss": loss}
        return results
        
    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train_acc1"].append(self.trainer.callback_metrics["train_acc_top1"])
        self.metric_hist["train_loss"].append(self.trainer.callback_metrics["train_loss"])

        self.log("train/acc_best", max(self.metric_hist["train_acc1"]), prog_bar=True, \
                sync_dist=True, sync_dist_op='mean')
        self.log("train/loss_best", min(self.metric_hist["train_loss"]), prog_bar=True, \
                sync_dist=True, sync_dist_op='mean')

    def validation_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        self.val_metrics(out, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        results = {"loss": loss}
        return results

    def validation_epoch_end(self, outputs):
        # outs is a list of whatever you returned in `validation_step`

        self.metric_hist["val_acc1"].append(self.trainer.callback_metrics["val_acc_top1"])
        self.metric_hist["val_loss"].append(self.trainer.callback_metrics["val_loss"])
        
        self.log("val/acc_best", max(self.metric_hist["val_acc1"]), prog_bar=True, sync_dist=True)
        self.log("val/loss_best", min(self.metric_hist["val_loss"]), prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, out, y = self.t_step(batch)

        self.test_metrics(out, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_epoch_end(self, outputs: List[Any]):
        pass

    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay #training.lr/wd
        )

        self.reduce_lr_on_plateau = load_obj(self.lr_scheduler.class_name)(optimizer, **self.lr_scheduler.params)
        lr_schedulers  = {'scheduler': self.reduce_lr_on_plateau, 'interval': 'epoch', 'monitor': 'val_acc_top1'}

        return [optimizer] , [lr_schedulers]


class LitClassifier_v3(pl.LightningModule):
    def __init__(self, 
                lr: float = 0.001,
                weight_decay: float = 0.0005
                ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.cfg = OmegaConf.load(os.path.join(os.getcwd(),'config.yaml'))
        self.lr_scheduler = self.cfg.scheduler
        self.optimizer = self.cfg.optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        def get_scalar_metrics(num_classes: int, average: str='macro', prefix: str=''):
            default = { 'acc_top1': metrics.Accuracy(top_k=1)}
            default = {prefix + str(key): val for key, val in default.items()}

            return metrics.MetricCollection(default)

        self.train_metrics = get_scalar_metrics(num_classes=self.cfg.model.nclasses, prefix='train_')
        self.val_metrics = get_scalar_metrics(num_classes=self.cfg.model.nclasses, prefix='val_')
        self.test_metrics = get_scalar_metrics(num_classes=self.cfg.model.nclasses, prefix='test_')

        self.metric_hist = {
            "train_acc1": [],
            "val_acc1": [],
            "train_loss": [],
            "val_loss": [],
        }
        self.model = load_obj(self.cfg.model.class_name)(self.cfg)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        return self.criterion(outputs, targets)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = self.softmax(logits)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        self.train_metrics(out, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        results = {"loss": loss}
        return results

    def validation_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        self.val_metrics(out, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        results = {"loss": loss}
        return results

    def validation_epoch_end(self, outputs):
        # outs is a list of whatever you returned in `validation_step`

        self.metric_hist["val_acc1"].append(self.trainer.callback_metrics["val_acc_top1"])
        # self.metric_hist["val_loss"].append(self.trainer.callback_metrics["val_loss"])
        
        self.log("val/acc_best", max(self.metric_hist["val_acc1"]), prog_bar=True, sync_dist=True)
        # self.log("val/loss_best", min(self.metric_hist["val_loss"]), prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, out, y = self.step(batch)

        self.test_metrics(out, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log("conf", self.trainer.callback_metrics["test_conf"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # return self.trainer.callback_metrics["test_acc_top1"]

    def configure_optimizers(self):
        
        optimizer = load_obj(self.optimizer.class_name)(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay #training.lr/wd
        )

        self.schedular = load_obj(self.lr_scheduler.class_name)(optimizer, **self.lr_scheduler.params)
        lr_schedulers  = {'scheduler': self.schedular, 'interval': 'epoch', 'monitor': 'val_acc_top1'}

        return [optimizer] , [lr_schedulers]

class LitClassifier_freeze(LitClassifier_v3):
    def __init__(self, 
                lr: float = 0.001,
                weight_decay: float = 0.0005,
                ):
        super().__init__(lr, weight_decay)
        self.save_hyperparameters()
        self.model = load_obj(self.cfg.model.class_name)(self.cfg)

    def configure_optimizers(self):
        '''
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        '''
        params_gamr = []
        params_encoder = []
        for name, param in self.named_parameters():
            if 'encoder' in name:
                params_encoder.append(param)
                param.requires_grad = False
            else:
                params_gamr.append(param)
        optimizer = load_obj(self.optimizer.class_name)(
            [
				{"params": params_encoder, "lr": 0},
				{"params": params_gamr},
			], 
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay #training.lr/wd
        )

        self.reduce_lr_on_plateau = load_obj(self.lr_scheduler.class_name)(optimizer, **self.lr_scheduler.params)
        lr_schedulers  = {'scheduler': self.reduce_lr_on_plateau, 'interval': 'epoch', 'monitor': 'val_acc_top1'}

        return [optimizer] , [lr_schedulers]


