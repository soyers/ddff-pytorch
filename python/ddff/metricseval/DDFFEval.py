import numpy as np
import ddff.dataproviders.datareaders.FocalStackDDFFH5Reader as FocalStackDDFFH5Reader
import ddff.trainers.DDFFTrainer as DDFFTrainer
from ddff.metricseval.BaseDDFFEval import BaseDDFFEval
import torchvision
from torch.utils.data import DataLoader

class DDFFEval(BaseDDFFEval):
    def __init__(self, checkpoint, focal_stack_size=10):
        self.trainer = DDFFTrainer.DDFFTrainer.from_checkpoint(checkpoint, focal_stack_size)
        super(DDFFEval, self).__init__(self.trainer)

    def evaluate(self, filename_testset, stack_key="stack_test", disp_key="disp_test", image_size=(383,552)):
        #Calculate pat size for images
        test_pad_size = (np.ceil((image_size[0] / 32)) * 32, np.ceil((image_size[1] / 32)) * 32) #32=2**numPoolings(=5)
        #Create test set transforms
        transform_test = [FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.ToTensor(), 
                            FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.ClipGroundTruth(0.0202, 0.2825), 
                            FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.PadSamples(test_pad_size), 
                            FocalStackDDFFH5Reader.FocalStackDDFFH5Reader.Normalize(mean_input=[0.485, 0.456, 0.406], std_input=[0.229, 0.224, 0.225])]
        transform_test = torchvision.transforms.Compose(transform_test)
        #Create dataloader
        datareader = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(filename_testset, transform=transform_test, stack_key=stack_key, disp_key=disp_key)
        dataloader = DataLoader(datareader, batch_size=1, shuffle=True, num_workers=0)
        return super(DDFFEval, self).evaluate(dataloader)
