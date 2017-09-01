% Setup matconvnet
run matlab/vl_setupnn.m
% Change to result folder
cd ~/models_tf/05_Cityscapes/CodeRelease/
resDir = 'val_prob/';
gtDir = '~/datasets/cityscapes/labels_ic19/';
kernel = 10;
stride = 10;
mode = 'val_prob';
use_ic = 0;
use_L1_loss = 0;



[acc_all] = city_evalSeg_F1(resDir,...
                              gtDir,...
                              kernel,...
                              stride,...
                              mode, ...
                              'ExpName','CodeRelease');
                          
%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
gtDir = '~/datasets/cityscapes/labels/';
resDir = 'val_ind/';
[pixel_acc,class_acc,IoU,conf] = city_evalSeg_IoU(resDir,gtDir);