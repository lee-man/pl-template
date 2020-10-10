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

    


def main():
    parser = argparse.ArgumentParser(description='Generic runner for your deep learning/machine learning model.')
    parser.add_argument('--config',  '-c',
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/resnet.yaml')
    parser.add_argument('--debug', action='store_true', help='debug mode.')
    args = parser.parse_args()

    # Parse the arguments
    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    for k,v in config.items():   # Make the configuration easier to manipulate.
        setattr(args, k, v)

    # Set the seed
    pl.seed_everything(args.logger_params['manual_seed'])
    # Create logger
    tt_logger = pl.loggers.TestTubeLogger(
        save_dir=args.logger_params['save_dir'],
        name=args.logger_params['name'],
        debug=args.debug,
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

    if args.debug:
        runner = pl.Trainer(
            default_root_dir=f"{tt_logger.save_dir}",
            logger=tt_logger,
            benchmark=True,
            deterministic=True,
            overfit_batches=0.01,
            **args.trainer_params
        )

    else:
        # Setting checkpointing callback function
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_last=True
        )

        runner = pl.Trainer(default_root_dir=f"{tt_logger.save_dir}",
                            logger=tt_logger,
                            benchmark=True,
                            deterministic=True,
                            # distributed_backend='ddp',
                            checkpoint_callback=checkpoint_callback,
                            **args.trainer_params)
    
    runner.fit(experiment, datamodule=dm)

if __name__ == "__main__":
    main()






     
    
    

