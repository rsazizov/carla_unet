import math
import tensorflow as tf

from tensorflow import keras
from pathlib import Path

def unet_conv_block(input_tensor, filters, last='pool'):
  conv2d_1 = keras.layers.Conv2D(filters, 3,
                                 activation='relu',
                                 padding='same')(input_tensor)
  bn1 = keras.layers.BatchNormalization()(conv2d_1)

  conv2d_2 = keras.layers.Conv2D(filters, 3,
                                 activation='relu',
                                 padding='same')(bn1)

  bn2 = keras.layers.BatchNormalization()(conv2d_2)

  if last == 'pool':
    return bn2, keras.layers.MaxPool2D(2, 2)(bn2)
  elif last == 'upsample':
    conv2d_t = keras.layers.Conv2DTranspose(filters//2, 2, 2, padding='same')(bn2)
    bn3 = keras.layers.BatchNormalization()(conv2d_t)
    return bn3
  else:
    return bn2

def get_unet(input_tensor, classes):
  bn = keras.layers.BatchNormalization()(input_tensor)
  block1, block1_p = unet_conv_block(bn, 64, last='pool')
  block2, block2_p = unet_conv_block(block1_p, 128, last='pool')
  block3, block3_p = unet_conv_block(block2_p, 256, last='pool')
  block4, block4_p = unet_conv_block(block3_p, 512, last='pool')

  block5 = unet_conv_block(block4_p, 1024, last='upsample')

  def crop_size(shape1, shape2):
    top_crop = math.ceil((shape1.shape[1] - shape2.shape[1]) / 2)
    bottom_crop = math.floor((shape1.shape[1] - shape2.shape[1]) / 2)

    left_crop = math.ceil((shape1.shape[2] - shape2.shape[2]) / 2)
    right_crop = math.floor((shape1.shape[2] - shape2.shape[2]) / 2)

    return ((top_crop, bottom_crop), (left_crop, right_crop))

  block4 = keras.layers.Cropping2D(crop_size(block4, block5))(block4)
  block6_in = keras.layers.Concatenate()([block4, block5])
  block6 = unet_conv_block(block6_in, 512, last='upsample')

  block3 = keras.layers.Cropping2D(crop_size(block3, block6))(block3)
  block7_in = keras.layers.Concatenate()([block3, block6])
  block7 = unet_conv_block(block7_in, 256, last='upsample')

  block2 = keras.layers.Cropping2D(crop_size(block2, block7))(block2)
  block8_in = keras.layers.Concatenate()([block2, block7])
  block8 = unet_conv_block(block8_in, 128, last='upsample')

  block1 = keras.layers.Cropping2D(crop_size(block1, block8))(block1)
  block9_in = keras.layers.Concatenate()([block8, block1])
  block9 = unet_conv_block(block9_in, 64, last='none')

  block10 = keras.layers.Conv2D(classes, 3, padding='same', activation='softmax')(block9)

  return keras.Model(inputs=[input_tensor], outputs=[block10])

class IoU(keras.metrics.MeanIoU):
  def __init__(self, num_classes, name=None):
    super().__init__(num_classes, name)

  def update_state(self, y_true, y_pred, w=None):
    return super().update_state(tf.argmax(y_true, axis=-1),
                                tf.argmax(y_pred, axis=-1), w)

