# Deep Depth From Focus
[Deep Depth From Focus](http://hazirbas.com/projects/ddff/) implementation in PyTorch. Please check the [ddff-toolbox](https://github.com/hazirbas/ddff-toolbox) for refocusing and camera parameters.

## Usage
### Installation
To run the project a Python 3.7.0 environment and a number of packages are required. The easiest way to fetch all dependencies is to install via pip.
```
pip install -r requirements.txt
```

### Training and Testing
This implementation contains the [Deep Depth from Focus model](python/ddff/models/DDFFNet.py) and a class to run the [training and prediction](python/ddff/trainers/DDFFTrainer.py) on a provided dataset. Furthermore a [datareader](python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py) class is provided to read hdf5 files containing focal stacks and their corresponding disparity maps.

In order to evaluate the model, an [evaluation class](python/ddff/metricseval/DDFFEval.py) is provided. It takes a model checkpoint and a path to the test data (h5 file) and features a method to calculate the errors described in the Deep Depth From Focus paper.

ince the original implementation of Deep Depth From Focus was created in TensorFlow and TFLearn the class [DDFFTFLearnEval](python/ddff/metricseval/DDFFTFLearnEval.py) loads the checkpoint exported from the original model in order to perform the error evlauation. [eval_ddff_tflearn.py](python/eval_ddff_tflearn.py) shows an example of how to use the class.

The pretrained weights exported from the TensorFlow/TFLearn model and converted to a PyTorch compatible dict is available [here](https://vision.in.tum.de/webarchive/hazirbas/ddff12scene/ddffnet-cc3-snapshot-121256.npz)(159.3MB).

The training process can be started by running [run_ddff.py](python/run_ddff.py) which can be provided with a training dataset passing the parameter ```--dataset```. To evaulate the results the generated checkpoint file can be loaded as shown in [eval_ddff.py](python/eval_ddff.py) which calculates the error metrics on a test dataset.

#### Initiazation
To train the network on the dataset introduced in the Deep Depth From Focus paper [run_ddff.py](python/run_ddff.py) has to be run with respective arguments specifying where the dataset is located and other hyper parameters that can be inspected by passing the argument ```-h```.
The [datareader](python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py) class requires the provided h5 file to contain a key for the focal stacks (default: "stack_train") and a key for the corresponding disparity maps (default: "disp_train") that can be passed during initialization of the reader.

#### Data preparation
The focal stacks in the hdf5 file have to be of shape [stacksize, height, width, channels] containing values in the range [0,255].

The disparity maps have to be of shape [1, height, width] containing the disparity in pixels. The dataset introduced in the Deep Depth From Focus paper contains disparities in the range [0.0202, 0.2825]

Please download the [trainval](https://vision.in.tum.de/webarchive/hazirbas/ddff12scene/ddff-dataset-trainval.h5) (12.6GB) and [test](https://vision.in.tum.de/webarchive/hazirbas/ddff12scene/ddff-dataset-test.h5) (761.1MB) hdf5 datasets. Focal stacks can be read as:
~~~~
import h5py

dataset = h5py.File("ddff-dataset-trainval.h5", "r")
focal_stacks = dataset["stacks_train"]
disparities = dataset["disp_train"]
~~~~

Please submit your results to the [Competition](https://competitions.codalab.org/competitions/17807) to evaluate on the test set.

**Note that** test scores are a slightly worse by a margin of 0.0001 (MSE) than the results presented on the paper due to the framework switch.

## Citation
If you use this code or the publicly shared model, please cite the following paper.

Caner Hazirbas, Sebastian Georg Soyer, Maximilian Christian Staab, Laura Leal-Taixé and Daniel Cremers, _"Deep Depth From Focus"_, ACCV, 2018. ([arXiv](https://arxiv.org/abs/1704.01085))

    @InProceedings{hazirbas18ddff,
     author    = {C. Hazirbas and S. G. Soyer and M. C. Staab and L. Leal-Taixé and D. Cremers},
     title     = {Deep Depth From Focus},
     booktitle = {Asian Conference on Computer Vision (ACCV)},
     year      = {2018},
     month     = {December},
     eprint    = {1704.01085},
     url       = {https://hazirbas.com/projects/ddff/},
    }

## License
The code is released under [GNU General Public License Version 3 (GPLv3)](http://www.gnu.org/licenses/gpl.html).
