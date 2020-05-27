import tensorflow as tf
import matplotlib as mpl

from functools import partial
from tensorflow.data.experimental import AUTOTUNE
from tensorflow import data

def process_image(path):
  image = tf.io.read_file(path)
  image = tf.image.decode_png(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, tf.constant([512, 512]))
  image = tf.cast(image * 255, tf.uint8)

  return image

def process_path(img_path, set_path):
  splits = tf.strings.split(img_path, '/')
  img_name = splits[-1]

  rgb_image_path = set_path + '/CameraRGB/' + img_name
  seg_image_path = set_path + '/CameraSeg/' + img_name

  seg_img = process_image(seg_image_path)
  seg_shape = seg_img.shape
  seg_img = seg_img[:, :, 0]

  return (process_image(rgb_image_path), seg_img)

def get_carla_data(set_path):
  dataset = data.Dataset.list_files(set_path + '/CameraRGB/*.png')
  dataset = dataset.map(partial(process_path, set_path=set_path))
  dataset = dataset.prefetch(AUTOTUNE)
  return dataset.shuffle(64)

def get_carla_cmap():
  # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
  # 0	Unlabeled	( 0, 0, 0)
  # 1	Building	( 70, 70, 70)
  # 2	Fence	(190, 153, 153)
  # 3	Other	(250, 170, 160)
  # 4	Pedestrian	(220, 20, 60)
  # 5	Pole	(153, 153, 153)
  # 6	Road line	(157, 234, 50)
  # 7	Road	(128, 64, 128)
  # 8	Sidewalk	(244, 35, 232)
  # 9	Vegetation	(107, 142, 35)
  # 10	Car	( 0, 0, 142)
  # 11	Wall	(102, 102, 156)
  # 12	Traffic sign	(220, 220, 0)
  carla_color_dict = {
    0: [0, 0, 0],
    1: [70, 70, 70],
    2: [190, 153, 153],
    3: [250, 170, 160],
    4: [220, 20, 60],
    5: [153, 153, 153],
    6: [157, 234, 50],
    7: [128, 64, 128],
    8: [244, 35, 232],
    9: [107, 142, 35],
    10: [0, 0, 142],
    11: [102, 102, 156],
    12: [220, 220, 0]
  }

  carla_color_dict = {k: [x / 255 for x in v] for k, v in color_dict.items()}

  carla_cmap = mpl.colors.ListedColormap(carla_color_dict.values())
  carla_bounds = tuple(carla_color_dict.keys()) + (13,)
  carla_norm = mpl.colors.BoundaryNorm(carla_bounds, carla_bounds.N)

  return carla_cmap, carla_norm
