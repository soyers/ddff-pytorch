#! /usr/bin/python3

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
import h5py

class FocalStackDDFFH5Reader(Dataset):

    def __init__(self, hdf5_filename, transform=None, stack_key="stack_test", disp_key="disp_test"):
        """
        Args:
            root_dir_fs (string): Directory with all focal stacks of all image datasets.
            root_dir_depth (string): Directory with all depth images of all image datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #Disable opencv threading since it leads to deadlocks in PyTorch DataLoader
        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.transform = transform

    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        #Create sample dict
        sample = {'input': self.hdf5[self.stack_key][idx].astype(float), 'output': self.hdf5[self.disp_key][idx]}

        #Transform sample with data augmentation transformers
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]

    class ToTensor(object):
        """Convert ndarrays in sample to Tensors."""
        def __call__(self, sample):
            #Add color dimension to depth map
            sample['output'] = np.expand_dims(sample['output'], axis=0)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            sample['input'] = torch.from_numpy(sample['input'].transpose((0,3,1,2))).float()
            sample['output'] = torch.from_numpy(sample['output']).float()
            return sample

    class Normalize(object):
        def __init__(self, mean_input, std_input, mean_output=None, std_output=None):
            self.mean_input = mean_input
            self.std_input = std_input
            self.mean_output = mean_output
            self.std_output = std_output

        def __call__(self, sample):
            input_images = torch.stack([torchvision.transforms.functional.normalize(sample_input, mean=self.mean_input, std=self.std_input) for sample_input in sample['input']])
            if self.mean_output is None or self.std_output is None:
                output_image = sample['output']
            else:
                output_image = torchvision.transforms.functional.normalize(sample['output'], mean=self.mean_output, std=self.std_output)
            return {'input': input_images, 'output': output_image}

    class ClipGroundTruth(object):
        def __init__(self, lower_bound, upper_bound):
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        def __call__(self, sample):
            sample['output'][sample['output'] < self.lower_bound] = 0.0
            sample['output'][sample['output'] > self.upper_bound] = 0.0
            return sample

    class RandomCrop(object):
        def __init__(self, output_size, valid_crop_threshold=0.8):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            self.valid_crop_threshold = valid_crop_threshold

        def __is_valid_crop(self, output_image, valid_pixel_cond=lambda x : x >= 0.01):
            valid_occurrances = valid_pixel_cond(output_image).sum()
            all_occurances = np.prod(output_image.shape)
            return (float(valid_occurrances) / float(all_occurances)) >= self.valid_crop_threshold

        def __call__(self, sample):
            h, w = sample['input'].shape[2:4]
            new_h, new_w = self.output_size

            #Generate list of possible random crops
            candidates = np.asarray([(x,y) for y in range(h - new_h) for x in range(w - new_w)])
            np.random.shuffle(candidates)
            
            #Iterate through candidates and choose forst valid crop
            for x,y in candidates:
                output_image = sample['output'][:,y:(y + new_h),x:(x + new_w)]
                if self.__is_valid_crop(output_image):
                    input_images = torch.stack([sample_input[:,y:(y + new_h),x:(x + new_w)] for sample_input in sample['input']])
                    return {'input': input_images, 'output': output_image}

            #No valid crop found. Return any crop
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            input_images = torch.stack([sample_input[:,top:(top + new_h),left:(left + new_w)] for sample_input in sample['input']])
            output_image = sample['output'][:,top:(top + new_h),left:(left + new_w)]
            return {'input': input_images, 'output': output_image}

    class PadSamples(object):
        def __init__(self, output_size, ground_truth_pad_value=0.0):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            self.ground_truth_pad_value = ground_truth_pad_value

        def __call__(self, sample):
            h, w = sample['input'].shape[2:4]
            new_h, new_w = self.output_size
            padh = np.int32(new_h - h)
            padw = np.int32(new_w - w)
            sample['input'] = torch.stack([torch.from_numpy(np.pad(sample_input.numpy(), ((0,0),(0,padh),(0,padw)), mode="reflect")).float() for sample_input in sample['input']])
            sample['output'] = torch.from_numpy(np.pad(sample['output'].numpy(), ((0,0),(0,padh),(0,padw)), mode="constant", constant_values=self.ground_truth_pad_value)).float()

            return sample

    class RandomSubStack(object):
        def __init__(self, output_size):
            self.output_size = output_size

        def __call__(self, sample):
            sample['input'] = torch.stack([sample['input'][i] for i in np.random.choice(sample['input'].shape[0], self.output_size, replace=False)])
            return sample
