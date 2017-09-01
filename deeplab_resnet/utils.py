from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

# colour map cityscapes
label_colours = [(128,64,128)
                # 0='road'
                ,(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153)
                # 1='sidewalk', 2='building', 3='wall', 4='fence', 5='pole'
                ,(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180)
                # 6='traffic light', 7='traffic sign', 8='vegetation', 9='terrain', 10='sky'
                ,(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100)
                # 11='person', 12='rider', 13='car', 14='truck', 15='bus'
                ,(0,80,100),(0,0,230),(119,11,32),(0,0,0)]
                # 16='train', 17='motorcycle', 18='bicycle', 19='background'                
                
# image mean
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    
def decode_labels(mask, n_classes):
    """Decode batch of segmentation masks.
    
    Args:
      label_batch: result of inference after taking argmax.
    
    Returns:
      An batch of RGB images of the same size
    """
    img = Image.new('RGB', (len(mask[0]), len(mask)))
    pixels = img.load()
    for j_, j in enumerate(mask):
        for k_, k in enumerate(j):
            if k < n_classes:
                pixels[k_,j_] = label_colours[k]
    return np.array(img)



def prepare_label(input_batch, new_size, n_classes, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
          input_batch = tf.one_hot(input_batch, depth=n_classes)
    return input_batch

def inv_preprocess(imgs, num_images):
  """Inverse preprocessing of the batch of images.
     Add the mean vector and convert from BGR to RGB.
   
  Args:
    imgs: batch of input images.
    num_images: number of images to apply the inverse transformations on.
  
  Returns:
    The batch of the size num_images with the same spatial dimensions as the input.
  """
  n, h, w, c = imgs.shape
  assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
  outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
  for i in range(num_images):
    outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
  return outputs


def get_delocalized_loss(images_one_hot,
			labels_one_hot,
			mask_proc_1c, # [batch_size,41,41,1]
			indices_ignore_class, # [batch_size,41,41,1]
			n_classes,
			batch_size,
			loss_type='square_error',
			kernel_size=3,
			stride=1):
	"""Computes custom loss.

	Args:
	  images_one_hot: input tensor of shape [batch_size H W 1].
	  labels: input tensor of shape [batch_size H W 1].
	  mask: shape [batch_size H W 1].
  
	Returns:
	  Outputs a tensor of shape [1]
	"""
	images_deloc = tf.nn.max_pool(images_one_hot,[1, kernel_size, kernel_size, 1],[1,stride,stride,1],'SAME')
	images_deloc_lin = tf.reshape(images_deloc, [-1, n_classes]) ###

	labels_deloc = tf.nn.max_pool(labels_one_hot,[1, kernel_size, kernel_size, 1],[1,stride,stride,1],'SAME')
	labels_deloc_lin = tf.reshape(labels_deloc, [-1, n_classes]) ###
 
	# Mask needs to be downsampled according to size of pred and gt after max_pooling
	h_new = tf.shape(labels_deloc)[1]
	w_new = tf.shape(labels_deloc)[2]
	new_shape = tf.pack([h_new, w_new])
	mask_small = tf.image.resize_nearest_neighbor(mask_proc_1c, new_shape) # mask_proc_1c: 1 inside, 0 outside
	indices_ignore_class_small = tf.image.resize_nearest_neighbor(tf.cast(indices_ignore_class,tf.int32), new_shape)

	### This is the cross_entropy loss
	indices_outside = tf.less(mask_small, 0.5)
	mask_combined = tf.logical_and(tf.cast(indices_outside,tf.bool),tf.cast(indices_ignore_class_small,tf.bool))
	mask_combined_lin = tf.reshape(mask_combined,[-1,])
	indices = tf.squeeze(tf.where(mask_combined_lin),1)# 10x41x41x20
 
	images_deloc_lin_sel = tf.gather(images_deloc_lin, indices)
	labels_deloc_lin_sel = tf.gather(labels_deloc_lin, indices)

	if loss_type == 'square_error':
		print('square_error')
		images_deloc_lin_sel = tf.nn.sigmoid(images_deloc_lin_sel)
		loss_square_pixel = tf.squared_difference(images_deloc_lin_sel,labels_deloc_lin_sel)
		loss = tf.reduce_mean(loss_square_pixel)
	elif loss_type == 'sigmoid_cross_entropy':
		print('Sigmoid cross entropy loss')     
		loss_sel_pixel = tf.nn.sigmoid_cross_entropy_with_logits(logits=images_deloc_lin_sel, targets=labels_deloc_lin_sel)
		loss = tf.reduce_mean(loss_sel_pixel)
	elif loss_type == 'softmax_cross_entropy':
		print('Softmax cross entropy loss')          
		loss_sel_pixel = tf.nn.softmax_cross_entropy_with_logits(logits=images_deloc_lin_sel, labels=labels_deloc_lin_sel)
		loss = tf.reduce_mean(loss_sel_pixel)     
	else:
		print('Error: Loss type must be either cross_entropy or square_error' )
	return loss
 
 
