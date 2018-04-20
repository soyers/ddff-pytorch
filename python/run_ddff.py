import torch
import random
import numpy as np
import ddff.trainers.DDFFTrainer as DDFFTrainer

if __name__ == "__main__":

    root_dir_h5 = "/usr/data/soyers/Original_Dataset/dataset-trainval.h5"
    epochs = 200
    split_ratio = 0.8

    #Uncomment to finetune pretrained vgg16 net from torchvision package
    #ddff_trainer = DDFFTrainer.DDFFTrainer.from_h5_data(root_dir_h5, learning_rate=0.001, cc1_enabled=False, cc2_enabled=False, cc3_enabled=True, cc4_enabled=False, cc5_enabled=False, training_crop_size=None, validation_crop_size=None, pretrained='no_bn', epochs=epochs, checkpoint_file="ddff_checkpoint_cc3_orig_data_refac.pt")
 
    #Finetune tensorflow vgg16 model
    ddff_trainer = DDFFTrainer.DDFFTrainer.from_h5_data(root_dir_h5, learning_rate=0.001, cc1_enabled=False, cc2_enabled=False, cc3_enabled=True, cc4_enabled=False, cc5_enabled=False, training_crop_size=None, validation_crop_size=None, pretrained='/usr/data/soyers/vgg16.npy', normalize_mean=[103.939/255.0, 116.779/255.0, 123.68/255.0], normalize_std=[1.0, 1.0, 1.0], epochs=epochs, checkpoint_file="ddff_cc3_checkpoint.pt", batch_size=2, deterministic=True)
