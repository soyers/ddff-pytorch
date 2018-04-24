# Deep Depth From Focus
[Deep Depth from Focus](http://hazirbas.com/projects/ddff/) implementation in PyTorch. Please check the [ddff-toolbox](https://github.com/hazirbas/ddff-toolbox) for refocusing and camera parameters.

## Usage
### Installation
To run the project a Python 3.5 environment and a number of packages are required. The easiest way to fetch all dependencies is to install [Anaconda](https://conda.io/) and load the [environment](condaenv.yml) provided in this repository:
```
conda env create -f condaenv.ym
```

### Training and Testing
This implementation contains the [Deep Depth from Focus model](python/ddff/models/DDFFNet.py) and a class to run the [training and prediction](python/ddff/trainers/DDFFTrainer.py) on a provided dataset. Furthermore a [datareader](python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py) class is provided to read hdf5 files containing focal stacks and their corresponding disparity maps.

In order to evaluate the model, an [evaluation class](python/ddff/metricseval/DDFFEval.py) is provided. It takes a model checkpoint and a path to the test data (h5 file) and features a method to calculate the errors described in the Deep Depth From Focus paper.

Since the original implementation of Deep Depth From Focus was created in TensorFlow with TFLearn, the class [DDFFTFLearnEval](python/ddff/metricseval/DDFFTFLearnEval.py) loads the checkpoint exported from the original model in order to perform the error evlauation. [python/eval_ddff_tflearn.py](eval_ddff_tflearn.py) shows an example of how to use the class.

#### Initiazation
To train the network on the dataset introduced in the Deep Depth From Focus paper [run_ddff.py](python/run_ddff.py) has to be run with respective arguments specifying where the dataset is located and other hyper parameters that can be inspected by passing the argument ```-h```.
The [datareader](python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py) class requires the provided h5 file to contain a key for the focal stacks (default: "stack_test") and a key for the corresponding disparity maps (default: "disp_test") that can be passed during initialization of the reader.

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

**Note that** test scores are a bit worse than the results presented on the paper due to the framework switch.

## Citation
If you use this code or the publicly shared model, please cite the following paper.

Caner Hazirbas, Laura Leal-Taixé  and Daniel Cremers, _"Deep Depth From Focus"_, ArXiv, 2017. ([arXiv](https://arxiv.org/abs/1704.01085))

    @inproceedings{ddff17arxiv,
     author    = {C. Hazirbas and L. Leal-Taixé and D. Cremers},
     title     = {Deep Depth From Focus},
     booktitle = {ArXiv},
     year      = {2017},
     month     = {April},
     eprint = {1704.01085},
     url = {https://github.com/hazirbas/ddff-toolbox},
    }

## License
The code is released under [GNU General Public License Version 3 (GPLv3)](http://www.gnu.org/licenses/gpl.html).
