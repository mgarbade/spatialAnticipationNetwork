# spatialAnticipationNetwork

## Requirements
Training:
- [Tensorflow](https://www.tensorflow.org/versions/r0.12/get_started/os_setup) (tested with version 0.12)  

Evaluation:
- Matlab
- [MatConvNet](http://www.vlfeat.org/matconvnet/)

## Quick Start
- Prepare Cityscapes dataset: Convert background label `255` to `19`
- Download [models](https://drive.google.com/open?id=0BxsTYGkWsxcJWlR1V0pkRW1YVWM) and place them inside the 'spatialAnticipationNetwork'-root folder.
- Adapt the paths in `train.py`
- Train the model using `python train.py`
- Adapt the paths in `eval.py`
- Predict labels on the validation set `python eval.py`
- Compute IoU and F1-scores by using `./matlab/evaluateAllResults.m` after adapting paths


## Acknowledgement
Most of the python code in this repository was written by modifying a duplicate of DrSleep's-[deeplab-tensorflow project](https://github.com/DrSleep/tensorflow-deeplab-resnet).
The Matlab evaluation scripts were written by modifying Liang-Chieh Chen's [deeplab-public-ver2](https://bitbucket.org/aquariusjay/deeplab-public-ver2)
