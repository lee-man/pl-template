import yaml
import os
import argparse
import logging

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

import data
import models
from experiment import ClsExperiment


class ClassificationModel(pl.LightningModule):
    '''
    Generic PyTorch-Lightning model for classfication task
    '''
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
    
    def forward(self, x):
        x = self.model(x)
        return x
    



def main():
    parser = argparse.ArgumentParser(description='Generic runner for your deep learning/machine learning model.')
    parser.add_argument('--config',  '-c',
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/setup.yaml')
    args = parser.parse_args()

    # Parse the arguments
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    for k,v in config.items():   # Make the configuration easier to manipulate.
        setattr(args, k, v)

    # Create logger
    tt_logger = pl.loggers.TestTubeLogger(
        save_dir=args.logging_params['save_dir'],
        name=args.logging_params['name'],
        debug=args.logging_params['debug'],
        create_git_tag=False,
    )

    # Create data module
    assert args.exp_params['dataset'].lower() in data.data_modules, 'The dataset is not supported yet.'
    dm = data.data_modules[args.exp_params['dataset']](**args.exp_params)

    # Create model
    assert args.model_params['name'].lower() in models.cnn_models, 'The model is not implemented yet.'
    model = models.cnn_models[args.model_params['name']](num_classes=dm.num_classes, **args.model_params)

    # Create experiment instant
    experiment = ClsExperiment(model, **args.exp_params)

    runner = pl.Trainer(default_save_path=f"{tt_logger.save_dir}",
                 min_nb_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **args.trainer_params)
    
    runner.fit(experiment)








     
    
    

