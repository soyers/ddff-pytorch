#! /usr/bin/python3

import argparse
import torch
import random
import numpy as np
import ddff.trainers.DDFFTrainer as DDFFTrainer

if __name__ == "__main__":
    #Add command line parser arguments
    parser = argparse.ArgumentParser(description='Train ddff net on specified h5 dataset.')
    parser.add_argument('--dataset', default="ddff-dataset-trainval.h5", help='h5 file that contains the training and validation data (default: ddff-dataset-trainval.h5)')
    parser.add_argument('--epochs', default=200, type=int, help='number of training epochs (default: 200)')
    parser.add_argument('--checkpoint', default="ddff_cc3_checkpoint.pt", help='Checkpoint file to be created during training (default: ddff_cc3_checkpoint.pt)')
    parser.add_argument('--checkpoint_frequency', default=5, type=int, help='Checkpoint frequency to save intermediate models. (default: 5)')
    parser.add_argument('--workers', default=0, type=int, help='Number of threads reading the dataset. (default: 0)')
    parser.add_argument('--batchsize', default=2, type=int, help='batch size during training (default: 2)')
    parser.add_argument('--pretrained', default="bn", help='Either specify a npy file to load tensorflow weights or use "bn" or "no_bn" to use pretrained weights from torchvision package (default: bn)')

    #Parse arguments
    args = parser.parse_args()

    #Finetune tensorflow vgg16 model
    ddff_trainer = DDFFTrainer.DDFFTrainer.from_h5_data(args.dataset,
                    learning_rate=0.001,
                    max_gradient=5.0,
                    cc1_enabled=False,
                    cc2_enabled=False,
                    cc3_enabled=True,
                    cc4_enabled=False,
                    cc5_enabled=False,
                    training_crop_size=None,
                    validation_crop_size=None,
                    pretrained=args.pretrained,
                    normalize_mean=None, normalize_std=None,
                    epochs=args.epochs,
                    checkpoint_file=args.checkpoint,
                    checkpoint_frequency=args.checkpoint_frequency,
                    batch_size=args.batchsize,
                    num_workers=args.workers,
                    deterministic=True)
