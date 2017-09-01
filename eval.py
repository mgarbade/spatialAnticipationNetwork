from __future__ import print_function

import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
from PIL import Image
import cv2
import scipy.io as sio # Save probabilities as mat files
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReaderEval, decode_labels

OUTPUT_IMGS = True

### Cityscapes (19 classes + BG)
PRINT_PROPABILITIES = True
n_classes = 19
ignore_label = 19 # Everything less than "ignore_label - 1" will be ignored
ignore_labels_above = 18
DATA_DIRECTORY = '/home/garbade/datasets/cityscapes/'
EXP_FOLDER = '/home/garbade/models_tf/05_Cityscapes/CodeRelease/'
OUTPUT_FOLDER = 'val'

# Validation fully visible
DATA_LIST_PATH = './cityscapes/filelist/val.txt'
DATA_LIST_PATH_ID = './cityscapes/filelist/val_id.txt'
EXP_ROOT = '/home/garbade/models_tf/05_Cityscapes/'
MASK_FILE = './mask/mask_642x1282.png'

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
    parser.add_argument("--expFolder", type=str, default=EXP_FOLDER, help="Specify expFolder")
    parser.add_argument("--mode", type=str, default=OUTPUT_FOLDER, help="mode")
    parser.add_argument("--data_list_path", type=str, default=DATA_LIST_PATH, help="data_list_path")
    parser.add_argument("--mask_file", type=str, default=MASK_FILE, help="mask_file")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--n_classes", type=int, default=n_classes,
                        help="How many classes to predict (default = n_classes).")
    parser.add_argument("--ignore_label", type=int, default=ignore_label,
			help="All labels >= ignore_label are beeing ignored")
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
    EXP_DIR = args.expFolder
    SAVE_DIR = EXP_DIR + '/' +  mode + '/'
    SAVE_DIR_IND = EXP_DIR + '/' + mode + '_ind/'
    SAVE_DIR_PROB = EXP_DIR + '/' + mode + '_prob/'    
    RESTORE_FROM = EXP_DIR + '/snapshots_finetune/model.ckpt-200'
    print(RESTORE_FROM)




    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    
    # Load mask if given
    if MASK_FILE is not None:
        mask = tf.image.decode_png(tf.read_file(MASK_FILE),channels=1)
        mask = tf.cast(mask, dtype=tf.float32) 

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
            msk = decode_labels(np.array(preds)[0, :, :, 0], args.n_classes)
            im = Image.fromarray(msk)
            im.save(SAVE_DIR + imgList[step] + '.png')

            mask_ind = np.array(preds)[0, :, :, 0]
            cv2.imwrite(SAVE_DIR_IND + imgList[step] + '.png', mask_ind)
            
        # Store probabilities
        if PRINT_PROPABILITIES:
            sio.savemat(SAVE_DIR_PROB + imgList[step],{'data':np.array(probs_small)[0, :, :, :]})
            
    coord.request_stop()
    coord.join(threads)
    duration = time.time() - start_time
    print('Time for inference: {:.3f} sec'.format(duration)) 
    
if __name__ == '__main__':
    main()





