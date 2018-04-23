#! /usr/bin/python3

import torch
from torch.autograd import Variable
from torch import optim

class BaseTrainer:
    def __init__(self, model, optimizer, training_loss, deterministic, scheduler=None, supervised=True):
        self.deterministic = deterministic
        if deterministic:
            self.__set_deterministic()
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = optimizer
        self.training_loss = training_loss
        self.scheduler = scheduler
        self.supervised = supervised

    def create_optimizer(self, net, optimizer_params):
        if optimizer_params["algorithm"] == 'sgd':
           return optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), 
                lr=optimizer_params["learning_rate"] if "learning_rate" in optimizer_params else 0.001, 
                momentum=optimizer_params["momentum"] if "momentum" in optimizer_params else 0.9, 
                weight_decay=optimizer_params["weight_decay"] if "weight_decay" in optimizer_params else 0.0005)
        elif optimizer_params["algorithm"] == 'adam':
            return optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()), 
                lr=optimizer_params["learning_rate"] if "learning_rate" in optimizer_params else 0.001, 
                weight_decay=optimizer_params["weight_decay"] if "weight_decay" in optimizer_params else 0.0005)
        else:
            return optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()), 
                lr=0.001, 
                momentum=0.9, 
                weight_decay=0.0005)
        

    def __set_deterministic(self):
        import random
        import numpy as np
        #Set RNG seeds
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        random.seed(42)
        np.random.seed(42)
        #Make results deterministic by disabling undeterministic functions in cuDNN
        torch.backends.cudnn.deterministic = True

    def set_supervised(self, supervised):
        self.supervised = supervised

    def set_training_loss(self, training_loss):
        self.training_loss = training_loss

    def train(self, dataloader, epochs, print_frequency=50, max_gradient=None, checkpoint_file=None, checkpoint_frequency=50):
        #Train model
        self.model.train()
        #Create list to keep track of losses foreach epoch
        epoch_losses = []
        #Run trainign loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                #Zero the parameter gradients
                self.optimizer.zero_grad()

                #Get the inputs
                inputs = data['input']
                #Copy inputs to GPU
                if torch.cuda.is_available():
                    if isinstance(inputs, list):
                        inputs = [element.cuda() for element in inputs]
                    else:
                        inputs = inputs.cuda()
                #Wrap inputs in Variable
                if isinstance(inputs, list):
                    inputs = [Variable(element) for element in inputs]
                else:
                    inputs = Variable(inputs)

                #Forward
                if isinstance(inputs, list):
                    output_approx = self.model(*inputs)
                else:
                    output_approx = self.model(inputs)

                if self.supervised:
                    #Get the inputs
                    outputs = data['output']
                    #Copy outputs to GPU
                    if torch.cuda.is_available():
                        if isinstance(outputs, list):
                            outputs = [element.cuda() for element in outputs]
                        else:
                            outputs = outputs.cuda()
                    #Wrap outputs in Variable
                    if isinstance(outputs, list):
                        outputs = [Variable(element) for element in outputs]
                        #Calculate loss
                        loss = self.training_loss(*output_approx, *outputs)
                    else:
                        outputs = Variable(outputs)
                        #Calculate loss
                        loss = self.training_loss(output_approx, outputs)
                else:
                    #Calculate loss
                    if isinstance(inputs, list):
                        loss = self.training_loss(*output_approx, *inputs)
                    else:
                        loss = self.training_loss(output_approx, inputs)

                #Backward
                loss.backward()
                
                #Clip gradients
                if max_gradient is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), max_gradient, norm_type=2)

                #Optimize
                self.optimizer.step()

                #Store epoch loss
                epoch_loss += loss.data[0]

                #Print statistics
                running_loss += loss.data[0]
                if i % print_frequency == print_frequency-1:    # print every print_frequency mini-batches   
                    print('[%d, %5d] loss: ' %
                      (epoch + 1, i + 1) + str(running_loss / print_frequency))
                    running_loss = 0.0

                #Save checkpoint
                if checkpoint_file is not None and i % checkpoint_frequency == checkpoint_frequency-1:  
                    self.save_checkpoint(checkpoint_file, epoch=(epoch+1), save_optimizer=True)

            #Save loss of epoch
            epoch_losses += [epoch_loss/len(dataloader)]

            #Update learning rate based on defined schedule
            if self.scheduler is not None:
                self.scheduler.step()
        #Save final checkpoint
        if checkpoint_file is not None:  
            self.save_checkpoint(checkpoint_file, epoch=epochs, save_optimizer=True)
        print("Training finished")
        return epoch_losses

    def evaluate(self, inputs):
        #Set model to eval mode in order to disable dropout
        self.model.eval()
        inputs = Variable(inputs, volatile=True)
        return self.model(inputs)


    def save_checkpoint(self, filename, epoch=None, save_optimizer=True):
        state = {'state_dict': self.model.state_dict()}
        if save_optimizer:
            state['optimizer'] = self.optimizer.state_dict()
        if epoch is not None:
            state['epoch'] = epoch
        torch.save(state, filename)

    def load_checkpoint(self, filename, load_optimizer=True, load_scheduler=True):
        #Load model to cpu
        checkpoint = torch.load(filename, map_location=lambda storage, location: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        #Upload model to GPU
        if torch.cuda.is_available():
            self.model.cuda()
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if load_scheduler and self.scheduler is not None and 'epoch' in checkpoint:
            self.scheduler.last_epoch = checkpoint['epoch']
        if 'epoch' in checkpoint:
            return checkpoint['epoch']
