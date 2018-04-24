#! /usr/bin/python3

import numpy as np
import ddff.dataproviders.datareaders.FocalStackDDFFH5Reader as FocalStackDDFFH5Reader
import ddff.trainers.DDFFTrainer as DDFFTrainer
from ddff.metricseval.BaseDDFFEval import BaseDDFFEval
import torchvision
from torch.utils.data import DataLoader

class DDFFTFLearnEval(BaseDDFFEval):
    def __init__(self, checkpoint, focal_stack_size=10, norm_mean=None, norm_std=None):
        trainer = DDFFTrainer.DDFFTrainer.from_tflearn(checkpoint, focal_stack_size)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        super(DDFFTFLearnEval, self).__init__(trainer)

    def evaluate(self, filename_testset, stack_key="stack_val", disp_key="disp_val", image_size=(383,552)):
        #Calculate pad size for images
        test_pad_size = (np.ceil((image_size[0] / 32)) * 32, np.ceil((image_size[1] / 32)) * 32) #32=2**numPoolings(=5)
        #Create test set transforms
        transform_test = [FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.ToTensor(), 
                            FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.PadSamples(test_pad_size)]
        if self.norm_mean is not None and self.norm_std is not None:
            transform_test += [FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.Normalize(mean_input=self.norm_mean, std_input=self.norm_std)]
        transform_test = torchvision.transforms.Compose(transform_test)
        #Create dataloader
        datareader = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(filename_testset, transform=transform_test, stack_key=stack_key, disp_key=disp_key)
        dataloader = DataLoader(datareader, batch_size=1, shuffle=False, num_workers=0)
        return super(DDFFTFLearnEval, self).evaluate(dataloader)
