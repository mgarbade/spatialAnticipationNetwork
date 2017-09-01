"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import scipy.io as sio # Save probabilities as mat files

import tensorflow as tf
import numpy as np

# from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label, decode_labels, decode_labels_old
from deeplab_resnet import DeepLabResNetModel, ImageReader, ImageReaderEval, prepare_label, decode_labels, decode_labels_old

OUTPUT_IMGS = True

### Cityscapes (19 classes + BG)
PRINT_PROPABILITIES = True
n_classes=19
ignore_label=19 # Everything less than "ignore_label - 1" will be ignored
ignore_labels_above = 18
DATA_DIRECTORY='/home/garbade/datasets/cityscapes/'

# Validation fully visible
DATA_LIST_PATH_ID='./dataset/city/val_id.txt'

# sz 100
#mode = 'val_sz100_rgb_input'
#DATA_LIST_PATH='./dataset/city/val_nc20_new.txt'
#MASK_FILE = './dataset/city/masks/01_center_visible/mask_sz100_1024x2048.png'
#MASK_FILE = './dataset/city/masks/01_center_visible/no_mask.png'

# sz50
#DATA_LIST_PATH='./dataset/city/small_50/val_nc20.txt'
#MASK_FILE = './dataset/city/masks_50/01_center_visible/mask_sz50_320x640.png'

INPUT_SIZE = '321,642'
# Validation on Training set
#DATA_LIST_PATH='./dataset/city/small_50/train_aug_nc20.txt'
#DATA_LIST_PATH_ID='./dataset/city/small_50/train_id.txt'

## Phase1
#mode = 'val_predNetCol_4fold_pad4'
#DATA_LIST_PATH='./dataset/city/val_4fold_pad4.txt'
#MASK_FILE = './dataset/city/masks/mask_4fold_pad4.png'

# Phase 2
#mode = 'val_sz100_phase2_predNet_641x1281_colors_23'
#DATA_LIST_PATH='./dataset/city/val_2fold_pad1.txt'
#MASK_FILE = './dataset/city/masks/01_center_visible/mask_sz100_1024x2048.png'


#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/18_predNet_nc20/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/19_2_predNet_nc20_ganLoss/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/20_nc20_ic19/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/30_retryingGanLoss_nc20_ic19_restore_17_1_full/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/30_2_noGanLoss_nc20_ic19_restore_17_1_full/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/30_3_GanLoss_nc20_ic19_restore_17_1_full/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/31_predNet_adversarial/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/31_3_predNet_adversarial/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/32_1_predNet_nc19/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/33_2_noDelocLoss_nc19_newRandMasking/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/34_3_DelocLoss_kernel3_stride2_sqerr/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/31_4_GanLoss_w1e5_nc19_new_random_masking/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/31_8_NOGL_DL_nc19_NM3/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/31_7_NOGanLoss_w1e5_nc19_NM3/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/36_1_Msk4_nc19_sc1_032/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/36_2_Msk1_nc19_sc1_032/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/38_1_fixMskAndDelocLoss/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/38_2_Msk1_DelocLoss/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/38_4_Msk1_DL10/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/38_5_Msk1_DL100/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/38_3_Msk1_Adam/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/39_1_M1_DL_noLossInside/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/39_2_M1_noLossInside/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/39_3_M1_DL10_noLossInside/'
#EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/39_4_M1_DL100_noLossInside/'
EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/'





#SAVE_DIR = EXP_ROOT + '/images_val_pred_masked_sz100_snap14900/'
#SAVE_DIR_IND = EXP_ROOT + '/images_val_pred_ind_masked_sz100_snap14900/'
# sz50
#SAVE_DIR = EXP_ROOT + '/images_val_pred_masked_sz50/'
#SAVE_DIR_IND = EXP_ROOT + '/images_val_pred_ind_masked_sz50/'


imgList = []
with open(DATA_LIST_PATH_ID, "rb") as fp:
    for i in fp.readlines():
        tmp = i[:-1]
        try:
            imgList.append(tmp)
        except:pass

if imgList == []:
    print('Error: Filelist is empty')
else:
    print('Filelist loaded successfully')
NUM_STEPS = len(imgList)
print(NUM_STEPS)
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--expFolder", type=str, help="Specify expFolder")
    parser.add_argument("--mode", type=str, help="mode")
    parser.add_argument("--data_list_path", type=str, help="data_list_path")
    parser.add_argument("--mask_file", type=str, help="mask_file")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    #parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        #help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
#    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
#                        help="Where restore model parameters from.")
#    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
#                        help="Where to save predicted masks.")
#    parser.add_argument("--save_dir_ind", type=str, default=SAVE_DIR_IND,
#                        help="Where to save predicted masks index.")
    parser.add_argument("--n_classes", type=int, default=n_classes,
                        help="How many classes to predict (default = n_classes).")
    parser.add_argument("--ignore_label", type=int, default=ignore_label,
			help="All labels >= ignore_label are beeing ignored")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")   
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    start_time = time.time()

    mode = args.mode
    DATA_LIST_PATH = args.data_list_path
    MASK_FILE = args.mask_file
    
    EXP_DIR = EXP_ROOT + args.expFolder
    
    SAVE_DIR = EXP_DIR + '/' +  mode + '/'
    SAVE_DIR_IND = EXP_DIR + '/' + mode + '_ind/'
    SAVE_DIR_PROB = EXP_DIR + '/' + mode + '_prob/'    
    RESTORE_FROM = EXP_DIR + '/snapshots_finetune/model.ckpt-20000'
    print(RESTORE_FROM)




    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    
    # Load mask if given
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)      
    if MASK_FILE is not None:
        mask = tf.image.decode_png(tf.read_file(MASK_FILE),channels=1)
        mask = tf.cast(mask, dtype=tf.float32) 
        # Downsample to input image size -> needs same size for evaluation of IoU
#        mask_int = tf.cast(mask, dtype=tf.int32)         
#        mask_with_weights = tf.concat(2,[mask,mask])

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReaderEval(args.data_dir, 
                                 DATA_LIST_PATH, 
                                 coord, 
                                 mask = mask)         
        image = reader.image
    image_batch = tf.expand_dims(image, dim=0) # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, args.n_classes, is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output_small = net.layers['fc1_voc12']
    raw_output_big = tf.image.resize_bilinear(raw_output_small, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output_big, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # mIoU
    pred_lin = tf.reshape(pred, [-1,])
#    gt = tf.reshape(label_batch, [-1,])
    
#    locs = gt < 255    
#    locs(mask == 0) = false;    
    
#    weights = tf.cast(tf.less_equal(gt, args.n_classes - 1), tf.int32) # TODO: Include n_classes -1 ->Ignore void label '255'.
#    weights = tf.cast(tf.less_equal(gt, ignore_labels_above), tf.int32) # TODO: Include n_classes -1 ->Ignore void label '255'.


#    mask_inside = tf.reshape(mask_int, [-1])
#    weights_inside = tf.select(mask_inside == 0, weights, tf.zeros_like(weights))
#    mask_outside = (mask_inside - 1)  * -1
#    weights_outside = tf.select(mask_outside == 0, weights, tf.zeros_like(weights))

#    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt, num_classes = args.n_classes, weights = weights)
#    mIoU_inside, update_op_inside = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt, num_classes = args.n_classes, weights = mask_inside)
#    mIoU_outside, update_op_outside = tf.contrib.metrics.streaming_mean_iou(pred_lin, gt, num_classes = args.n_classes, weights = mask_outside)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, RESTORE_FROM)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)    # Iterate over training steps.
    if not os.path.exists(SAVE_DIR_IND):
        os.makedirs(SAVE_DIR_IND)    # Iterate over training steps.        
    if not os.path.exists(SAVE_DIR_PROB):
        os.makedirs(SAVE_DIR_PROB)    # Iterate over training steps.                
    for step in range(args.num_steps):
    #for step in range(1):    
        if PRINT_PROPABILITIES:
            preds, preds_lin, probs_small, probs = sess.run([pred, pred_lin, raw_output_small, raw_output_big])
        else:
            preds, preds_lin = sess.run([pred, pred_lin])
        if step % 1 == 0:
            print('step {:d}'.format(step))
        if OUTPUT_IMGS:
            # print(np.array(preds).shape)
            msk = decode_labels_old(np.array(preds)[0, :, :, 0], args.n_classes)
            im = Image.fromarray(msk)
            im.save(SAVE_DIR + imgList[step] + '.png')

            mask_ind = np.array(preds)[0, :, :, 0]
            cv2.imwrite(SAVE_DIR_IND + imgList[step] + '.png', mask_ind)
            
        # Store probabilities
        if PRINT_PROPABILITIES:
            sio.savemat(SAVE_DIR_PROB + imgList[step],{'data':np.array(probs_small)[0, :, :, :]})
            
            # print('File saved to {}'.format(args.save_dir + imgList[step] + '.png'))
#    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
#    print('Mean IoU_inside: {:.3f}'.format(mIoU_inside.eval(session=sess)))
#    print('Mean IoU_outside: {:.3f}'.format(mIoU_outside.eval(session=sess)))

    coord.request_stop()
    coord.join(threads)
    duration = time.time() - start_time
    print('Time for inference: {:.3f} sec'.format(duration)) 
    
if __name__ == '__main__':
    main()





