import yaml
import os
import argparse
import logging

from pytorch_lightning import loggers as pl_loggers


def main():
    parser = argparse.ArgumentParser(description='Generic runner for your deep learning/machine learning model.')
    parser.add_argument('--config',  '-c',
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/setup.yaml')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    for k,v in config.items():   # make the configuration easier to manipulate.
        setattr(args, k, v)

    tt_logger = pl_loggers.TestTubeLogger(
        save_dir=args.logging_params['save_dir'],
        name=args.logging_params['name'],
        debug=args.logging_params['debug'],
        create_git_tag=False,
    )
    

