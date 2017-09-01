# spatialAnticipationNetwork

Most of the code in this repository was written by modifying a duplicate of DrSleep's-[deeplab-tensorflow project](https://github.com/DrSleep/tensorflow-deeplab-resnet)


## Requirements
Training:
- tensorflow (tested with version 0.12)  

Evaluation:
- Matlab
- [MatConvNet](http://www.vlfeat.org/matconvnet/)

## Quick Start
- Download [models] and place them inside the 'spatialAnticipationNetwork'-root folder.
- Adapt the paths in `train.py`
- Train the model using `python train.py`
- Adapt the paths in `eval.py`
- Compute the evaluation results using `python eval.py`


