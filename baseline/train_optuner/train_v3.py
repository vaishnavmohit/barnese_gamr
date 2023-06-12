"""
File to train a model for image classificaton
"""
from typing import List, Optional

import logging
import os
import warnings
import hydra 
import pdb 

from memory_profiler import profile

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers.neptune import NeptuneLogger
from neptune.new.types import File

from csv import writer, DictWriter

from baseline.data import *

from pathlib import Path

# taken from 
# https://github.com/optuna/optuna/blob/master/examples/pytorch/pytorch_lightning_simple.py
import optuna

from baseline.utils import template_utils

log = template_utils.get_logger(__name__)

logger1 = logging.getLogger(__name__)
logger1.setLevel(logging.DEBUG)

warnings.simplefilter(action="ignore", category=FutureWarning)

def train(config: DictConfig) -> Optional[float]:
    """
    Train a model for image classification

    Args:
        cfg: configuration
    """

    logger1.info(f"Current working directory: {os.getcwd()}")

    # Seed everything
    # removing seed as cuda initialization error
    seed_everything(config.seed)

    OmegaConf.save(config=config, f="config.yaml")
    logger1.info(OmegaConf.to_yaml(config))

    # reading and updating model lr and w_d:
    # if not config.training.test and not config.training.optuna:
    if 'gamr_add' in config.model.architecture and config.training.optuna is False:
        csv_name = '/users/mvaishn1/data/data/mvaishn1/svrt/code/esbn_related/'
        csv_name+='esbn_qr_m_v5c'+'_'+str(4)+'_grp_'+str(config.training.grp)+'.csv' # config.model.architecture
        # csv_name+=str(config.model.steps)+'_grp_'+str(config.training.grp)+'.csv'
        if os.path.exists(csv_name):
            df=pd.read_csv(csv_name, names =('task', 'lr', 'w_d'))
            config.model_optuna.lr = float(df[df.task==int(config.training.task)].lr)
            config.model_optuna.weight_decay = float(df[df.task==config.training.task].w_d)
            OmegaConf.save(config=config, f="config.yaml")

    logger1.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    if not config.training.test:
        # checkpoint path
        ckpt_path = os.path.join(config.training.ckpt, "checkpoints/last.ckpt")
    else:
        ckpt_path = config.training.ckpt

    if os.path.exists(ckpt_path):
        logging.info(f"Loading existing checkpoint @ {ckpt_path}")
    else:
        logging.info("No existing ckpt found. Training from scratch")
        ckpt_path = None

    # Init Lightning model
    logger1.info(f"Instantiating model <{config.model_optuna._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model_optuna)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {pytorch_total_params}')
    
    # import pdb; pdb.set_trace()

    logging.info(model)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Neptune:
    if config.training.neptune:
        T1 = True if config.model.params.pretrained else False
        # naming tags and experiment
        exp_name = config.model.architecture+'_data'+config.datamodule.data_name+'_bs'+str(config.datamodule.batch_size)+'_lr'+str(config.model_optuna.lr)+'_wd'+str(config.model_optuna.weight_decay)
        tags=[config.model.architecture, 'pytorch-light', 'ncls-'+str(config.model.nclasses), \
        'grp-'+str(config.training.grp), 'wd-'+str(config.model_optuna.weight_decay), 'data-'+config.datamodule.data_name, 'task-'+str(config.training.task), \
        'imgpre-'+str(T1)]

        trainer_logger = NeptuneLogger(project_name="Serre-Lab/visreas", experiment_name=exp_name, tags=tags, upload_source_files=["*.yaml"] )

    else:
        trainer_logger = logger

    # Init Lightning trainer
    logger1.info(f"Instantiating trainer <{config.trainer._target_}>")
    
    # enabling resume function when not using optuna framework:
    if not config.training.optuna:
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, \
            resume_from_checkpoint=ckpt_path, \
            callbacks=callbacks, \
            plugins=DDPPlugin(find_unused_parameters=True), \
            logger=trainer_logger, _convert_="partial" # changing logger
        )
    else:
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, \
            callbacks=callbacks, \
            plugins=DDPPlugin(find_unused_parameters=True), \
            logger=trainer_logger, _convert_="partial" # changing logger
        )
    # move_metrics_to_cpu=True
    # Send some parameters from config to all lightning loggers
    logger1.info("Logging hyperparameters!")
    template_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if not config.training.test:
        # Train the model
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    if not config.training.optuna:
        trainer.test(model=model, datamodule=datamodule)
    # if not config.trainer.get("fast_dev_run"):
    #     log.info("Starting testing!")
    #     trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    template_utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if config.training.test:
        val_acc = 0
    else:
        val_acc = trainer.callback_metrics["val/acc_best"].item()

    if not config.training.optuna:
        
        # saving the results:
        
        dict = {}
        dict['test_acc'] = trainer.callback_metrics["test_acc_top1"].item()
        dict['val_acc'] = val_acc
        dict['model_name'] = config.model.architecture
        dict['task'] = config.training.task
        dict['grp'] = config.training.grp
        dict['epoch'] = config.trainer.max_epochs
        dict['batch_size'] = config.training.batch_size
        dict['lr'] = config.model_optuna.lr
        dict['weight_decay'] = config.model_optuna.weight_decay
        dict['checkpoint_path'] = trainer.checkpoint_callback.best_model_path
        # if 'esbn' in config.model.architecture:
        dict['step'] = config.model.steps  
        dict['model_ckpt'] = config.model.params.ckpt
        # else:
        dict['pretrained'] = config.model.params.pretrained
        dict['nhead'] = config.model.nhead
        dict['seed'] = config.seed
        if 'noise' in config.model.architecture:
            dict['var'] = config.training.var

        # list of column names 	
        field_names = dict.keys()
        
        # Open your CSV file in append mode
        # Create a file object for this file
        csvname = '/gpfs/data/tserre/data/mvaishn1/svrt/code/'
        csvname+= config.training.csv +'.csv'
        if 'deit' in dict['model_name']:
            csvname = '/gpfs/data/tserre/data/mvaishn1/svrt/code/result_deit.csv'
        with open(csvname, 'a') as f_object:
            
            # Pass the file object and a list 
            # of column names to DictWriter()
            # You will get a object of DictWriter
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        
            #Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(dict)
        
            #Close the file object
            f_object.close()

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")


    # return trainer.callback_metrics["val_F1_top1"].item()
    return val_acc #trainer.callback_metrics["val_acc_top1"].item()