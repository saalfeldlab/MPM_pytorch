#!/usr/bin/env python3
"""
Standalone INR training script for subprocess execution.

This script is called by run_MPM.py as a subprocess to ensure that any code modifications
to Siren_Network.py or graph_trainer.py are reloaded for each iteration.

Usage:
    python train_INR_subprocess.py --config CONFIG_PATH --field_name FIELD --device DEVICE [--erase]
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from MPM_pytorch.config import MPM_pytorchConfig
from MPM_pytorch.models.graph_trainer import data_train_INR
from MPM_pytorch.models.utils import set_device


def main():
    parser = argparse.ArgumentParser(description='Train INR network on MPM fields')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--field_name', type=str, default='Jp', help='Field to train on (Jp, F, S, C)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--erase', action='store_true', help='Erase existing log files')
    parser.add_argument('--log_file', type=str, default=None, help='Path to analysis log file')
    parser.add_argument('--config_file', type=str, default=None, help='Config file name for log directory (e.g., multimaterial/config_name)')

    args = parser.parse_args()

    # Load config
    config = MPM_pytorchConfig.from_yaml(args.config)

    # Set config_file if provided (needed for proper log directory path)
    if args.config_file:
        config.config_file = args.config_file

    # Set device
    device = set_device(args.device)

    # Open log file if specified
    log_file = None
    if args.log_file:
        log_file = open(args.log_file, 'w')

    try:
        # Run training - this will reload any modified code
        data_train_INR(
            config=config,
            device=device,
            field_name=args.field_name,
            erase=args.erase,
            log_file=log_file
        )
    finally:
        if log_file:
            log_file.close()


if __name__ == '__main__':
    main()
