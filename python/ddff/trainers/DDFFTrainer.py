#! /usr/bin/python3

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader
import ddff.models.DDFFNet as DDFFNet
import ddff.dataproviders.datareaders.FocalStackDDFFH5Reader as FocalStackDDFFH5Reader
from ddff.trainers.BaseTrainer import BaseTrainer

class DDFFTrainer(BaseTrainer):
    def __init__(self, stack_size, learning_rate=0.001, cliprange=[0.0202, 0.2825],
                        cc1_enabled=False, 
                        cc2_enabled=False, 
                        cc3_enabled=True, 
                        cc4_enabled=False, 
                        cc5_enabled=False, 
                        pretrained='no_bn', 
                        sequential_weight_sharing=False, 
                        scheduler_step_size=4, 
                        scheduler_gama=0.9, 
                        deterministic=False, 
                        optimizer='sgd'):
        #Define model
        net = DDFFNet.DDFFNet(stack_size, cc1_enabled=cc1_enabled, cc2_enabled=cc2_enabled, cc3_enabled=cc3_enabled, cc4_enabled=cc4_enabled, cc5_enabled=cc5_enabled, sequential_weight_sharing=sequential_weight_sharing, pretrained=pretrained)
        #Define optimizer
        if optimizer == 'sgd':
            opt = self.create_optimizer(net, {"algorithm":'sgd', "learning_rate":learning_rate,  "weight_decay": 0.0005, "momentum":0.9})
        else:
            opt = self.create_optimizer(net, {"algorithm":'adam', "learning_rate":learning_rate,  "weight_decay": 0.0005})
        #Define scheduler
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=scheduler_step_size, gamma=scheduler_gama)
        #Define training loss
        training_loss = self.MaskedLoss(nn.MSELoss(), valid_cond=lambda x : x >= cliprange[0])

        #Call parent constructor
        super(DDFFTrainer, self).__init__(net, opt, training_loss, deterministic, scheduler=scheduler)

    @classmethod
    def from_h5_data(cls,root_dir,
                        learning_rate=0.001, 
                        cc1_enabled=False, 
                        cc2_enabled=False, 
                        cc3_enabled=True, 
                        cc4_enabled=False, 
                        cc5_enabled=False, 
                        training_crop_size=None, 
                        validation_crop_size=None, 
                        pretrained='no_bn', 
                        normalize_mean=[0.485, 0.456, 0.406], 
                        normalize_std=[0.229, 0.224, 0.225],
                        sequential_weight_sharing=False, 
                        scheduler_step_size=4, 
                        scheduler_gama=0.9, 
                        deterministic=False, 
                        optimizer='sgd', 
                        epochs=20, 
                        batch_size=2, 
                        num_workers=4, 
                        checkpoint_file=None, 
                        checkpoint_frequency=50):
        #Create data loaders
        transform_train = cls.__create_preprocessing(cls, crop_size=training_crop_size, mean=normalize_mean, std=normalize_std)
        transform_validation = cls.__create_preprocessing(cls, crop_size=validation_crop_size, mean=normalize_mean, std=normalize_std)
        #Create h5 reader
        dataset_train = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(root_dir, transform=transform_train, stack_key="stack_train", disp_key="disp_train")
        dataset_validation = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(root_dir, transform=transform_validation, stack_key="stack_val", disp_key="disp_val")
        #Create data loader
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dataloader_validation = DataLoader(dataset_validation, batch_size=1, shuffle=True, num_workers=0)
        #Call constructor
        instance = cls(dataset_train.get_stack_size(), learning_rate=learning_rate, 
                        cc1_enabled=cc1_enabled, 
                        cc2_enabled=cc2_enabled, 
                        cc3_enabled=cc3_enabled, 
                        cc4_enabled=cc4_enabled, 
                        cc5_enabled=cc5_enabled, 
                        pretrained=pretrained, 
                        sequential_weight_sharing=sequential_weight_sharing, 
                        scheduler_step_size=scheduler_step_size, 
                        scheduler_gama=scheduler_gama, 
                        deterministic=deterministic, 
                        optimizer=optimizer)

        #Save instance variables
        instance.dataloader_validation = dataloader_validation

        #Fit instance
        epoch_losses = instance.train(dataloader_train, epochs, checkpoint_file=checkpoint_file, checkpoint_frequency=checkpoint_frequency)
        print("Losses per epoch: " + str(epoch_losses))

        return instance

    @classmethod
    def from_checkpoint(cls, checkpoint_file, stack_size,
                        cc1_enabled=False, 
                        cc2_enabled=False, 
                        cc3_enabled=True, 
                        cc4_enabled=False, 
                        cc5_enabled=False, 
                        sequential_weight_sharing=False, 
                        deterministic=False, 
                        optimizer='sgd'):
        #Call constructor
        instance = cls(stack_size,
                        cc1_enabled=cc1_enabled, 
                        cc2_enabled=cc2_enabled, 
                        cc3_enabled=cc3_enabled, 
                        cc4_enabled=cc4_enabled, 
                        cc5_enabled=cc5_enabled, 
                        sequential_weight_sharing=sequential_weight_sharing, 
                        deterministic=deterministic, 
                        optimizer=optimizer)

        #Load checkpoint
        instance.load_checkpoint(checkpoint_file)

        return instance

    @classmethod
    def from_tflearn(cls, checkpoint_file, stack_size,
                        cc1_enabled=False, 
                        cc2_enabled=False, 
                        cc3_enabled=True, 
                        cc4_enabled=False, 
                        cc5_enabled=False, 
                        sequential_weight_sharing=False, 
                        deterministic=False, 
                        optimizer='sgd'):
        #Call constructor
        instance = cls(stack_size,
                        cc1_enabled=cc1_enabled, 
                        cc2_enabled=cc2_enabled, 
                        cc3_enabled=cc3_enabled, 
                        cc4_enabled=cc4_enabled, 
                        cc5_enabled=cc5_enabled, 
                        sequential_weight_sharing=sequential_weight_sharing, 
                        deterministic=deterministic, 
                        optimizer=optimizer,
                        pretrained=None)

        #Load checkpoint
        instance.load_tflearn(checkpoint_file)

        return instance

    def load_tflearn(self, checkpoint_file):
        #Load dict
        pretrained_dict = np.load(checkpoint_file)
        #Update according to generated mapping
        pretrained_dict = {self.__translate_tflearn_key(k): v for k, v in pretrained_dict.items()}
        #Transpose all weight tensors since tflearn stores them transposed
        # Tensorflow 2D Conv layer: h * w * in_channels * out_channels
        # PyTorch 2D Conv layer: out_channels * in_channels * h * w
        #Same logic was also implemented in https://github.com/ruotianluo/pytorch-mobilenet-from-tf/blob/master/convert.py
        pretrained_dict = {k:(v.transpose((3, 2, 0, 1)) if (k.startswith("conv") or k.startswith("upconv")) and v.ndim == 4 else v) for k, v in pretrained_dict.items()}
        pretrained_dict = {("scoring" + k[len("conv_disp"):] if k.startswith("conv_disp") else "autoencoder." + k):v for k, v in pretrained_dict.items()}
        #Convert weight arrays to tirch tensors
        pretrained_dict = {k:torch.from_numpy(v).float() for k, v in pretrained_dict.items()}
        #Load weights
        self.model.load_state_dict(pretrained_dict)

    def __translate_tflearn_key(self, key):
        if key.endswith("/W:0"):
            return key[:-len("/W:0")] + ".weight"
        if key.endswith("/up_filter:0"):
            return key[:-len("/up_filter:0")] + ".weight"
        if key.endswith("/gamma:0"):
            return key[:-len("/gamma:0")] + ".weight"
        if key.endswith("/beta:0"):
            return key[:-len("/beta:0")] + ".bias"
        if key.endswith("/moving_mean:0"):
            return key[:-len("/moving_mean:0")] + ".running_mean"
        if key.endswith("/moving_variance:0"):
            return key[:-len("/moving_variance:0")] + ".running_var"

    def __create_preprocessing(self, crop_size=None, cliprange=[0.0202, 0.2825], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        transform = [FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.ToTensor()]
        if cliprange is not None:
            transform += [FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.ClipGroundTruth(cliprange[0], cliprange[1])]
        if crop_size is not None:
            transform += [FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.RandomCrop(crop_size)]
        transform += [FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.Normalize(mean_input=mean, std_input=std)]
        transform = torchvision.transforms.Compose(transform)
        return transform

    def create_validation_loader(self):
        try:
            return self.dataloader_validation
        except AttributeError:
            return None

    class MaskedLoss(nn.Module):
        def __init__(self, loss, valid_cond=lambda x : x > 0.0):
            super(DDFFTrainer.MaskedLoss, self).__init__()
            self.loss = loss
            self.valid_cond = valid_cond

        def forward(self, inputs, outputs):
            mask = self.valid_cond(outputs)#(outputs != self.invalid_value)
            return self.loss(inputs[mask], outputs[mask])
