#! /usr/bin/python3

import torch.nn as nn
import torchvision
import torch
import numpy as np

class DDFFNet(nn.Module):
    def __init__(self, focal_stack_size, output_dims=1, cc1_enabled=False, cc2_enabled=False, cc3_enabled=True, cc4_enabled=False, cc5_enabled=False, bias=False, pretrained='no_bn'):
        super(DDFFNet, self).__init__()
        self.autoencoder = DDFFAutoEncoder(output_dims, cc1_enabled, cc2_enabled, cc3_enabled, cc4_enabled, cc5_enabled, bias=bias)
        self.scoring = nn.Conv2d(focal_stack_size*output_dims, output_dims, 1, bias=False)
        #Init weights
        self.apply(self.weights_init)
        #Update pretrained weights
        if pretrained == 'no_bn':
            autoencoder_state_dict = self.autoencoder.state_dict()
            #Load pretrained dict
            pretrained_dict = torchvision.models.vgg16(pretrained=True).features.state_dict()
            #Filter and map pretrained dict
            pretrained_dict = self.__map_state_dict(pretrained_dict, bias=bias)
            #Update model dict
            autoencoder_state_dict.update(pretrained_dict)
            #Load updated state dict
            self.autoencoder.load_state_dict(autoencoder_state_dict)
        elif pretrained == 'bn':
            autoencoder_state_dict = self.autoencoder.state_dict()
            #Load pretrained dict
            pretrained_dict = torchvision.models.vgg16_bn(pretrained=True).features.state_dict()
            #Filter and map pretrained dict
            pretrained_dict = self.__map_state_dict_bn(pretrained_dict, bias=bias)
            #Update model dict
            autoencoder_state_dict.update(pretrained_dict)
            #Load updated state dict
            self.autoencoder.load_state_dict(autoencoder_state_dict)
        elif pretrained is not None:
            autoencoder_state_dict = self.autoencoder.state_dict()
            #Load pretrained dict
            pretrained_weights = np.load(pretrained, encoding="latin1").item()
            #Filter and map pretrained dict
            pretrained_dict = self.__map_state_dict_tf(pretrained_weights, bias=bias)
            #Update model dict
            autoencoder_state_dict.update(pretrained_dict)
            #Load updated state dict
            self.autoencoder.load_state_dict(autoencoder_state_dict)

    def forward(self, images):
        #Encode stacks in batch dimension and calculate features
        image_features = self.autoencoder(images.view(-1, *images.shape[2:]))
        #Encode stacks in feature dimension again
        image_features = image_features.view(images.shape[0], -1, *image_features.shape[2:])
        #Score extracted features
        result = self.scoring(image_features)

        return result

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0, 1.0)
            m.running_var.normal_(0, 1.0)
            m.running_mean.fill_(0)
            m.bias.data.fill_(0)

    def __map_state_dict(self, vgg16_features_dict, bias):
        layer_mappings = {'0.weight': 'conv1_1.weight',
                '2.weight': 'conv1_2.weight',
                '5.weight': 'conv2_1.weight',
                '7.weight': 'conv2_2.weight',
                '10.weight': 'conv3_1.weight',
                '12.weight': 'conv3_2.weight',
                '14.weight': 'conv3_3.weight',
                '17.weight': 'conv4_1.weight',
                '19.weight': 'conv4_2.weight',
                '21.weight': 'conv4_3.weight',
                '24.weight': 'conv5_1.weight',
                '26.weight': 'conv5_2.weight',
                '28.weight': 'conv5_3.weight'}
        if bias:
            layer_mappings.update({'0.bias': 'conv1_1.bias',
                '2.bias': 'conv1_2.bias',
                '5.bias': 'conv2_1.bias',
                '7.bias': 'conv2_2.bias',
                '10.bias': 'conv3_1.bias',
                '12.bias': 'conv3_2.bias',
                '14.bias': 'conv3_3.bias',
                '17.bias': 'conv4_1.bias',
                '19.bias': 'conv4_2.bias',
                '21.bias': 'conv4_3.bias',
                '24.bias': 'conv5_1.bias',
                '26.bias': 'conv5_2.bias',
                '28.bias': 'conv5_3.bias'})
        #Update according to generated mapping
        pretrained_dict = {layer_mappings[k]: v for k, v in vgg16_features_dict.items() if k in layer_mappings}
        return pretrained_dict

    def __map_state_dict_bn(self, vgg16_features_dict, bias):
        layer_mappings = {'0.weight': 'conv1_1.weight',
                '1.weight': 'conv1_1_bn.weight', '1.bias': 'conv1_1_bn.bias', '1.running_mean': 'conv1_1_bn.running_mean', '1.running_var': 'conv1_1_bn.running_var',
                '3.weight': 'conv1_2.weight',
                '4.weight': 'conv1_2_bn.weight', '4.bias': 'conv1_2_bn.bias', '4.running_mean': 'conv1_2_bn.running_mean', '4.running_var': 'conv1_2_bn.running_var',
                '7.weight': 'conv2_1.weight',
                '8.weight': 'conv2_1_bn.weight', '8.bias': 'conv2_1_bn.bias', '8.running_mean': 'conv2_1_bn.running_mean', '8.running_var': 'conv2_1_bn.running_var',
                '10.weight': 'conv2_2.weight',
                '11.weight': 'conv2_2_bn.weight', '11.bias': 'conv2_2_bn.bias', '11.running_mean': 'conv2_2_bn.running_mean', '11.running_var': 'conv2_2_bn.running_var',
                '14.weight': 'conv3_1.weight',
                '15.weight': 'conv3_1_bn.weight', '15.bias': 'conv3_1_bn.bias', '15.running_mean': 'conv3_1_bn.running_mean', '15.running_var': 'conv3_1_bn.running_var',
                '17.weight': 'conv3_2.weight',
                '18.weight': 'conv3_2_bn.weight', '18.bias': 'conv3_2_bn.bias', '18.running_mean': 'conv3_2_bn.running_mean', '18.running_var': 'conv3_2_bn.running_var',
                '20.weight': 'conv3_3.weight',
                '21.weight': 'conv3_3_bn.weight', '21.bias': 'conv3_3_bn.bias', '21.running_mean': 'conv3_3_bn.running_mean', '21.running_var': 'conv3_3_bn.running_var',
                '24.weight': 'conv4_1.weight',
                '25.weight': 'conv4_1_bn.weight', '25.bias': 'conv4_1_bn.bias', '25.running_mean': 'conv4_1_bn.running_mean', '25.running_var': 'conv4_1_bn.running_var',
                '27.weight': 'conv4_2.weight',
                '28.weight': 'conv4_2_bn.weight', '28.bias': 'conv4_2_bn.bias', '28.running_mean': 'conv4_2_bn.running_mean', '28.running_var': 'conv4_2_bn.running_var',
                '30.weight': 'conv4_3.weight',
                '31.weight': 'conv4_3_bn.weight', '31.bias': 'conv4_3_bn.bias', '31.running_mean': 'conv4_3_bn.running_mean', '31.running_var': 'conv4_3_bn.running_var',
                '34.weight': 'conv5_1.weight',
                '35.weight': 'conv5_1_bn.weight', '35.bias': 'conv5_1_bn.bias', '35.running_mean': 'conv5_1_bn.running_mean', '35.running_var': 'conv5_1_bn.running_var',
                '37.weight': 'conv5_2.weight',
                '38.weight': 'conv5_2_bn.weight', '38.bias': 'conv5_2_bn.bias', '38.running_mean': 'conv5_2_bn.running_mean', '38.running_var': 'conv5_2_bn.running_var',
                '40.weight': 'conv5_3.weight',
                '41.weight': 'conv5_3_bn.weight', '41.bias': 'conv5_3_bn.bias', '41.running_mean': 'conv5_3_bn.running_mean', '41.running_var': 'conv5_3_bn.running_var'}
        if bias:
            layer_mappings.update({'0.bias': 'conv1_1.bias',
                '3.bias': 'conv1_2.bias',
                '7.bias': 'conv2_1.bias',
                '10.bias': 'conv2_2.bias',
                '14.bias': 'conv3_1.bias',
                '17.bias': 'conv3_2.bias',
                '20.bias': 'conv3_3.bias',
                '24.bias': 'conv4_1.bias',
                '27.bias': 'conv4_2.bias',
                '30.bias': 'conv4_3.bias',
                '34.bias': 'conv5_1.bias',
                '37.bias': 'conv5_2.bias',
                '40.bias': 'conv5_3.bias'
            })
        #Update according to generated mapping
        pretrained_dict = {layer_mappings[k]: v for k, v in vgg16_features_dict.items() if k in layer_mappings}
        return pretrained_dict

    def __map_state_dict_tf(self, vgg16_features, bias):
        pretrained_dict = {
            'conv1_1.weight': torch.from_numpy(vgg16_features['conv1_1'][0].transpose((3, 2, 0, 1))).float(),
            'conv1_2.weight': torch.from_numpy(vgg16_features['conv1_2'][0].transpose((3, 2, 0, 1))).float(),
            'conv2_1.weight': torch.from_numpy(vgg16_features['conv2_1'][0].transpose((3, 2, 0, 1))).float(),
            'conv2_2.weight': torch.from_numpy(vgg16_features['conv2_2'][0].transpose((3, 2, 0, 1))).float(),
            'conv3_1.weight': torch.from_numpy(vgg16_features['conv3_1'][0].transpose((3, 2, 0, 1))).float(),
            'conv3_2.weight': torch.from_numpy(vgg16_features['conv3_2'][0].transpose((3, 2, 0, 1))).float(),
            'conv3_3.weight': torch.from_numpy(vgg16_features['conv3_3'][0].transpose((3, 2, 0, 1))).float(),
            'conv4_1.weight': torch.from_numpy(vgg16_features['conv4_1'][0].transpose((3, 2, 0, 1))).float(),
            'conv4_2.weight': torch.from_numpy(vgg16_features['conv4_2'][0].transpose((3, 2, 0, 1))).float(),
            'conv4_3.weight': torch.from_numpy(vgg16_features['conv4_3'][0].transpose((3, 2, 0, 1))).float(),
            'conv5_1.weight': torch.from_numpy(vgg16_features['conv5_1'][0].transpose((3, 2, 0, 1))).float(),
            'conv5_2.weight': torch.from_numpy(vgg16_features['conv5_2'][0].transpose((3, 2, 0, 1))).float(),
            'conv5_3.weight': torch.from_numpy(vgg16_features['conv5_3'][0].transpose((3, 2, 0, 1))).float(),
        }
        if bias:
            pretrained_dict.update({
                'conv1_1.bias': torch.from_numpy(vgg16_features['conv1_1'][1]).float(),
                'conv1_2.bias': torch.from_numpy(vgg16_features['conv1_2'][1]).float(),
                'conv2_1.bias': torch.from_numpy(vgg16_features['conv2_1'][1]).float(),
                'conv2_2.bias': torch.from_numpy(vgg16_features['conv2_2'][1]).float(),
                'conv3_1.bias': torch.from_numpy(vgg16_features['conv3_1'][1]).float(),
                'conv3_2.bias': torch.from_numpy(vgg16_features['conv3_2'][1]).float(),
                'conv3_3.bias': torch.from_numpy(vgg16_features['conv3_3'][1]).float(),
                'conv4_1.bias': torch.from_numpy(vgg16_features['conv4_1'][1]).float(),
                'conv4_2.bias': torch.from_numpy(vgg16_features['conv4_2'][1]).float(),
                'conv4_3.bias': torch.from_numpy(vgg16_features['conv4_3'][1]).float(),
                'conv5_1.bias': torch.from_numpy(vgg16_features['conv5_1'][1]).float(),
                'conv5_2.bias': torch.from_numpy(vgg16_features['conv5_2'][1]).float(),
                'conv5_3.bias': torch.from_numpy(vgg16_features['conv5_3'][1]).float()
            })
        return pretrained_dict

class DDFFAutoEncoder(nn.Module):
    """Create model from VGG_16 by deleting the classifier layer."""
    def __init__(self, output_dims, cc1_enabled, cc2_enabled, cc3_enabled, cc4_enabled, cc5_enabled, bias=False):
        super(DDFFAutoEncoder, self).__init__()
        #Save parameters
        self.output_dims = output_dims
        self.cc1_enabled = cc1_enabled
        self.cc2_enabled = cc2_enabled
        self.cc3_enabled = cc3_enabled
        self.cc4_enabled = cc4_enabled
        self.cc5_enabled = cc5_enabled

        #Encoder
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1, bias=bias)
        self.conv1_1_bn = nn.BatchNorm2d(64, eps=0.001)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, bias=bias)
        self.conv1_2_bn = nn.BatchNorm2d(64, eps=0.001)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, bias=bias)
        self.conv2_1_bn = nn.BatchNorm2d(128, eps=0.001)
        self.conv2_2 = nn.Conv2d(128, 128 , 3, padding=1, bias=bias)
        self.conv2_2_bn = nn.BatchNorm2d(128, eps=0.001)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, bias=bias)
        self.conv3_1_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_2_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_3_bn = nn.BatchNorm2d(256, eps=0.001)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.encdrop3 = nn.Dropout(p=0.5)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, bias=bias)
        self.conv4_1_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_3_bn = nn.BatchNorm2d(512, eps=0.001)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.encdrop4 = nn.Dropout(p=0.5)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_1_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_2_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_3_bn = nn.BatchNorm2d(512, eps=0.001)
        self.pool5 = nn.MaxPool2d(2, stride=2)
        self.encdrop5 = nn.Dropout(p=0.5)

        #Decoder
        self.upconv5 = nn.ConvTranspose2d(512, 512, 4, padding=1, stride=2, bias=False)
        if self.cc5_enabled:
            self.conv5_3_D = nn.Conv2d(1024, 512, 3, padding=1, bias=bias)
        else:
            self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_3_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_2_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv5_1_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.decdrop5 = nn.Dropout(p=0.5)

        self.upconv4 = nn.ConvTranspose2d(512, 512, 4, padding=1, stride=2, bias=False)
        if self.cc4_enabled:
            self.conv4_3_D = nn.Conv2d(1024, 512, 3, padding=1, bias=bias)
        else:
            self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_3_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1, bias=bias)
        self.conv4_2_D_bn = nn.BatchNorm2d(512, eps=0.001)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1, bias=bias)
        self.conv4_1_D_bn = nn.BatchNorm2d(256, eps=0.001)
        self.decdrop4 = nn.Dropout(p=0.5)

        self.upconv3 = nn.ConvTranspose2d(256, 256, 4, padding=1, stride=2, bias=False)
        if self.cc3_enabled:
            self.conv3_3_D = nn.Conv2d(512, 256, 3, padding=1, bias=bias)
        else:
            self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_3_D_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1, bias=bias)
        self.conv3_2_D_bn = nn.BatchNorm2d(256, eps=0.001)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1, bias=bias)
        self.conv3_1_D_bn = nn.BatchNorm2d(128, eps=0.001)
        self.decdrop3 = nn.Dropout(p=0.5)

        self.upconv2 = nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2, bias=False)
        if self.cc2_enabled:
            self.conv2_2_D = nn.Conv2d(256, 128, 3, padding=1, bias=bias)
        else:
            self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1, bias=bias)
        self.conv2_2_D_bn = nn.BatchNorm2d(128, eps=0.001)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1, bias=bias)
        self.conv2_1_D_bn = nn.BatchNorm2d(64, eps=0.001)

        self.upconv1 = nn.ConvTranspose2d(64, 64, 4, padding=1, stride=2, bias=False)
        if self.cc1_enabled:
            self.conv1_2_D = nn.Conv2d(128, 64, 3, padding=1, bias=bias)
        else:
            self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1, bias=bias)
        self.conv1_2_D_bn = nn.BatchNorm2d(64, eps=0.001)
        self.conv1_1_D = nn.Conv2d(64, self.output_dims, 3, padding=1, bias=bias)
        self.conv1_1_D_bn = nn.BatchNorm2d(self.output_dims, eps=0.001)

    def forward(self, x):
        #Encoder
        x = nn.functional.relu(self.conv1_1_bn(self.conv1_1(x)))
        cc1 = nn.functional.relu(self.conv1_2_bn(self.conv1_2(x)))
        x = self.pool1(cc1)
        x = nn.functional.relu(self.conv2_1_bn(self.conv2_1(x)))
        cc2 = nn.functional.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(cc2)
        x = nn.functional.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = nn.functional.relu(self.conv3_2_bn(self.conv3_2(x)))
        cc3 = nn.functional.relu(self.conv3_3_bn(self.conv3_3(x)))
        x = self.pool3(cc3)
        x = self.encdrop3(x)
        x = nn.functional.relu(self.conv4_1_bn(self.conv4_1(x)))
        x = nn.functional.relu(self.conv4_2_bn(self.conv4_2(x)))
        cc4 = nn.functional.relu(self.conv4_3_bn(self.conv4_3(x)))
        x = self.pool4(cc4)
        x = self.encdrop4(x)
        x = nn.functional.relu(self.conv5_1_bn(self.conv5_1(x)))
        x = nn.functional.relu(self.conv5_2_bn(self.conv5_2(x)))
        cc5 = nn.functional.relu(self.conv5_3_bn(self.conv5_3(x)))
        x = self.pool5(cc5)
        x = self.encdrop5(x)

        #Decoder
        x = self.upconv5(x)
        if self.cc5_enabled:
            x = torch.cat([x, cc5], 1)
        x = nn.functional.relu(self.conv5_3_D_bn(self.conv5_3_D(x)))
        x = nn.functional.relu(self.conv5_2_D_bn(self.conv5_2_D(x)))
        x = nn.functional.relu(self.conv5_1_D_bn(self.conv5_1_D(x)))
        x = self.decdrop5(x)
        x = self.upconv4(x)
        if self.cc4_enabled:
            x = torch.cat([x, cc4], 1)
        x = nn.functional.relu(self.conv4_3_D_bn(self.conv4_3_D(x)))
        x = nn.functional.relu(self.conv4_2_D_bn(self.conv4_2_D(x)))
        x = nn.functional.relu(self.conv4_1_D_bn(self.conv4_1_D(x)))
        x = self.decdrop4(x)
        x = self.upconv3(x)
        if self.cc3_enabled:
            x = torch.cat([x, cc3], 1)
        x = nn.functional.relu(self.conv3_3_D_bn(self.conv3_3_D(x)))
        x = nn.functional.relu(self.conv3_2_D_bn(self.conv3_2_D(x)))
        x = nn.functional.relu(self.conv3_1_D_bn(self.conv3_1_D(x)))
        x = self.decdrop3(x)
        x = self.upconv2(x)
        if self.cc2_enabled:
            x = torch.cat([x, cc2], 1)
        x = nn.functional.relu(self.conv2_2_D_bn(self.conv2_2_D(x)))
        x = nn.functional.relu(self.conv2_1_D_bn(self.conv2_1_D(x)))
        x = self.upconv1(x)
        if self.cc1_enabled:
            x = torch.cat([x, cc1], 1)
        x = nn.functional.relu(self.conv1_2_D_bn(self.conv1_2_D(x)))
        x = nn.functional.relu(self.conv1_1_D_bn(self.conv1_1_D(x)))
        return x
