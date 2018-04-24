import ddff.dataproviders.datareaders.FocalStackDDFFH5Reader as FocalStackDDFFH5Reader
import ddff.metricseval.DDFFTFLearnEval as DDFFTFLearnEval

if __name__ == "__main__":
    #Set parameters
    image_size = (383,552)
    filename_testset = "ddff-dataset-trainval.h5"
    checkpoint_file = "ddffnet-cc3-snapshot-121256.npz"
    stack_key = "stack_val"
    disp_key="disp_val"

    #Create validation reader
    tmp_datareader = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(filename_testset, transform=None, stack_key=stack_key, disp_key=disp_key)

    #Create PSPDDFF evaluator
    evaluator = DDFFTFLearnEval.DDFFTFLearnEval(checkpoint_file, focal_stack_size=tmp_datareader.get_stack_size(), norm_mean=None, norm_std=None)
    #Evaluate
    metrics = evaluator.evaluate(filename_testset, stack_key=stack_key, disp_key=disp_key, image_size=image_size)
    print(metrics)
