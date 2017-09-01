from __future__ import print_function

import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label, get_delocalized_loss

SOLVER_MODE = 1

#### Cityscapes (19 classes + BG)
loss_type = 'sigmoid_cross_entropy'
#loss_type = 'softmax_cross_entropy'
kernel_size = 10
stride = 10

EXP_FOLDER = '/home/garbade/models_tf/05_Cityscapes/CodeRelease/' # Output folder for model, log-files

restoring_mode = 'restore_all'
DATASET = 'CITY'
n_classes = 19
ignore_label = 19
ignore_labels_above = 18 # Class indices are 0-based
DATA_DIRECTORY = '/home/garbade/datasets/cityscapes/'
DATA_LIST_PATH = './cityscapes/filelist/train.txt'
RESTORE_FROM = './models/init_labelSasNet_35_1/model.ckpt-20000'
MASK_FILE = './mask/mask_642x1282.png' # 1 on inside, 0 on outside

BATCH_SIZE = 10
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
NUM_STEPS = 20001

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100

print('restoring_mode: ' + restoring_mode)
print('Dataset: ' + DATASET + '\n' + 
          'Restore from: ' + RESTORE_FROM)

## OPTIMISATION PARAMS ##
WEIGHT_DECAY = 0.0005
BASE_LR = LEARNING_RATE
POWER = 0.9
MOMENTUM = 0.9


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--expFolder", type=str, default=EXP_FOLDER, help="Specify expFolder")
    parser.add_argument("--kernel_size", type=int, default=kernel_size, help="Specify kernel_size")
    parser.add_argument("--stride", type=int, default=stride, help="Specify stride")    
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,help="Learning rate for training.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,help="Number of training steps.")
    parser.add_argument("--random-scale", action="store_true",help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,help="Save summaries and checkpoint every often.")
    parser.add_argument("--n_classes", type=int, default=n_classes,help="Number of classes.")
    parser.add_argument("--ignore_label", type=int, default=ignore_label,help="Ignore label class number.")   
    parser.add_argument("--restoring_mode", type=str, default=restoring_mode,help="restoring_modes: restore_all, restore_all_but_last.") 
    parser.add_argument("--mask_file", type=str, default=MASK_FILE,help="MASK_FILE.")                           
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

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
    """Create the model and start the training."""
    args = get_arguments()
    
    # Output dirs
    OUTPUT_ROOT = args.expFolder
    SAVE_DIR = OUTPUT_ROOT + '/images_finetune/'
    SNAPSHOT_DIR = OUTPUT_ROOT + '/snapshots_finetune/'
    LOG_DIR = OUTPUT_ROOT + '/logs/' 
    print('OUTPUT_ROOT: ' + OUTPUT_ROOT)
    # Kernel params
    kernel_size = args.kernel_size
    stride = args.stride
    print('kernel_size: ' + str(kernel_size))
    print('stride: ' + str(stride))
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load mask if given
    if args.mask_file is not None:
        print('Load mask from: ' + args.mask_file)
        print('Mask before cropping')
        mask = tf.image.decode_png(tf.read_file(args.mask_file),channels=1)
        mask = tf.cast(mask, dtype=tf.float32)   
        mask_with_weights = tf.concat(2,[mask,mask])
        
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            'train',    # phase is either 'train', 'val' or 'test'
            coord,
            args.ignore_label,
            mask = mask_with_weights,
            scale = True,
            mirror = True,
            crop = True)
        image_batch, label_mask_batch = reader.dequeue(args.batch_size)
    
   
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)    
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, args.n_classes, is_training=args.is_training)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = tf.global_variables()
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    
    vars_restore_gist = [v for v in tf.global_variables() if not 'fc' in v.name] # Restore everything but last layer
    
    # labels and masks are still concatenated until here
    label_batch, mask_batch, lossWeight_batch = tf.split(split_dim=3,num_split=3,value=label_mask_batch)
    label_batch = tf.cast(label_batch, dtype=tf.uint8)
    label_proc = prepare_label(label_batch, tf.pack(raw_output.get_shape()[1:3]),args.n_classes, one_hot=False) # [batch_size,41,41]
    mask_proc = prepare_label(mask_batch, tf.pack(raw_output.get_shape()[1:3]),args.n_classes, one_hot=False) # [batch_size,41,41]

    # Get masks highlighting everything but the "void" class ("19")
    indices_ic = tf.less_equal(label_proc, ignore_labels_above) # [batch_size,41,41] -->ignore_labels_above should be 18
    indices_ic = tf.expand_dims(indices_ic, 3) # [batch_size,41,41,1]
    
    # Get mask for outside labels
    mask_proc_1c = tf.expand_dims(mask_proc,axis=3) # mask (float32): 1 inside, 0 outside
    label_proc_one_hot = tf.one_hot(label_proc, depth=args.n_classes) # shape=(10, 41, 41, 19)    
    
    
                                
    # Mask which includes everything (inside and outside labels)        
    mask_proc_full_image = tf.zeros_like(mask_proc_1c) # 0 values will be selected by "get_delocalized_loss"
    loss = get_delocalized_loss(raw_output,
               			label_proc_one_hot,
               			mask_proc_full_image,
               			indices_ic,
               			args.n_classes,
               			args.batch_size,
               			loss_type=loss_type, # sigmoid_cross_entropy, square_error, softmax_cross_entropy
               			kernel_size=kernel_size,
               			stride=stride)

        
    reduced_loss = loss 
    
    # Processed predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    ## OPTIMISER ##
    base_lr = tf.constant(BASE_LR)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / 20000), POWER))
    tf.summary.scalar('learning_rate', learning_rate)

    opt_conv = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, MOMENTUM)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, MOMENTUM)    

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
    
    # Set up tf session and initialize variables. 
    init = tf.global_variables_initializer()

    # Log variables
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph) 
    tf.summary.scalar("reduced_loss", reduced_loss) 
    for v in conv_trainable + fc_w_trainable + fc_b_trainable: # Add histogram to all variables
        tf.summary.histogram(v.name.replace(":","_"),v)
    merged_summary_op = tf.summary.merge_all() 
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)

    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        if args.restoring_mode ==  'restore_all_but_last':
            print('Restore everything but last layer')
            loader = tf.train.Saver(var_list=vars_restore_gist)
        elif args.restoring_mode ==  'restore_all':
            print('Restore all layers')            
            loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)
    
    # Create save_dir
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)        
        
    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step}

        if step % args.save_pred_every == 0:
            loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, 
                                                                      image_batch, 
                                                                      label_batch, 
                                                                      pred, 
                                                                      merged_summary_op,
                                                                      train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            ### Print intermediary images
            fig, axes = plt.subplots(args.save_num_images, 3, figsize = (16, 12))
            for i in xrange(args.save_num_images):
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels(labels[i, :, :, 0], args.n_classes))

                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels(preds[i, :, :, 0], args.n_classes))
            plt.savefig(SAVE_DIR + str(start_time) + ".png")
            plt.close(fig)
            ###
            if args.save_pred_every is not 2:
                save(saver, sess, SNAPSHOT_DIR, step)
        else:
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))            
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
