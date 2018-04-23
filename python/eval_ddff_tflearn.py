import ddff.dataproviders.datareaders.FocalStackDDFFH5Reader as FocalStackDDFFH5Reader
import ddff.metricseval.DDFFTFLearnEval as DDFFTFLearnEval

if __name__ == "__main__":
    #Set parameters
    image_size = (383,552)
    filename_testset = "/usr/data/soyers/Original_Dataset/PCH(224)_STK[0.28#0.02#10]_test.h5"
    checkpoint_file = "/usr/data/cvpr_shared/hazirbas/ddff-tf/snapshot-121256.npz"

    #Create validation reader
    tmp_datareader = FocalStackDDFFH5Reader.FocalStackDDFFH5Reader(filename_testset, transform=None, stack_key="stack_test", disp_key="disp_test")

    #Create PSPDDFF evaluator
    evaluator = DDFFTFLearnEval.DDFFTFLearnEval(checkpoint_file, focal_stack_size=tmp_datareader.get_stack_size(), norm_mean=None, norm_std=None)
    #Evaluate
    metrics = evaluator.evaluate(filename_testset, image_size=image_size)
    print(metrics)
