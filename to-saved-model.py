#!/usr/bin/env python3
from argparse import ArgumentParser
from nets import NET_CHOICES
from heads import HEAD_CHOICES
import common
import os
parser = ArgumentParser(description='Train a ReID network.')

# Required.
parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--saved_model', default=None,
    help='Name of the saved model ')

parser.add_argument(
    '--model_name', default='resnet_v1_50', choices=NET_CHOICES,
    help='Name of the model to use.')

parser.add_argument(
    '--head_name', default='fc1024', choices=HEAD_CHOICES,
    help='Name of the head to use.')
    
parser.add_argument(
    '--experiment_root', required=True, type=common.writeable_directory,
    help='Location used to store checkpoints and dumped data.')

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
from importlib import import_module

def main():
  args = parser.parse_args()

  # filename2 = "/media/pierre/KlapStorage/equidia/jersey-identification_tmp/2018-06-09_R1C1/boxed/2330.0-0.2-00020-0.png"
  # image = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(filename2), channels=3), (256, 128))
  #input = tf.keras.backend.cast(tf.keras.preprocessing.image.load_img(filename2, target_size=[256, 128]), 'float')
  # images = tf.expand_dims(image, 0)
  model_name=args.model_name
  head_name=args.head_name

  # tf.debugging.set_log_device_placement(True)
  
  if args.checkpoint is None:
      checkpoint = tf.train.latest_checkpoint(args.experiment_root)
  else:
      checkpoint = os.path.join(args.experiment_root, args.checkpoint)
  
  
  if args.saved_model is None:
      saved_model = os.path.join(args.experiment_root, 'saved_model.pb')
  else:
      saved_model = os.path.join(args.experiment_root, args.saved_model)
  
  # "/media/pierre/KlapStorage/equidia/reid-checkpoints/horserider_202000804-test/saved_model-9"
  # saved_model_path2 = "/media/pierre/KlapStorage/equidia/reid-checkpoints/horserider_202000804-test/saved_model-9-test"
  # checkpoint=args.checkpoint
  # '/media/pierre/KlapStorage/equidia/reid-checkpoints/horserider_202000804-test/checkpoint-9'

  model = import_module('nets.' + model_name)
  head = import_module('heads.' + head_name)
  
  placeholder_img = tf.compat.v1.placeholder(tf.float32, shape=(None,256,128, 3))
  endpoints, body_prefix = model.endpoints(placeholder_img, is_training=False)

  with tf.name_scope('head'):
      endpoints = head.head(endpoints, 128, is_training=False)

  with tf.compat.v1.Session() as sess:

      saver = tf.compat.v1.train.Saver()
      sess.run(tf.compat.v1.global_variables_initializer())
      saver.restore(sess, checkpoint)
      tf.compat.v1.saved_model.simple_save(
          sess, 
          saved_model,
          {
            "images": placeholder_img
          },
          {
            "emb": tf.identity(endpoints['emb'], name="emb")
          }
      )
  
  print('Model saved from '+checkpoint+' to '+saved_model)

if __name__ == '__main__':
    main()
    

# Expected
# import h5py
# filename = '/media/pierre/KlapStorage/equidia/reid-checkpoints/horserider_202000804-test/horserider_202000804-test_query_embeddings-one-line.h5'
# 
# with h5py.File(filename, "r") as f:
#     data = list(f['emb'])

# Actual
## From checkpoints

# model = import_module('nets.' + model_name)
# head = import_module('heads.' + head_name)
# placeholder_img = tf.compat.v1.placeholder(tf.float32, shape=(1,256,128, 3))
# endpoints, body_prefix = model.endpoints(placeholder_img, is_training=False)
# 
# with tf.name_scope('head'):
#     endpoints = head.head(endpoints, 128, is_training=False)
# 
# list1 = tf.train.list_variables(checkpoint)
# 
# 
# with tf.compat.v1.Session() as sess:
# 
#     saver = tf.compat.v1.train.Saver()
#     saver.restore(sess, checkpoint) 
#     img_values = sess.run(images)
#     emb = sess.run(endpoints['emb'], feed_dict={placeholder_img: img_values})
#     tf.compat.v1.saved_model.simple_save(
#         sess, 
#         saved_model_path2,
#         {
#           "placeholder_img": placeholder_img
#         },
#         {
#           "emb": tf.identity(endpoints['emb'], name="emb")
#         }
#     )
# print('norm chckpoints', np.linalg.norm(data[0]-emb))


## From Saved model

# model2 = tf.compat.v1.saved_model.load_v2(saved_model_path2)
# # model2.restore(checkpoint)
# # # pruned = model2.prune('images:0', 'head/out_emb:0')
# # # endpoints3 = pruned(inputs)
# prune = model2.signatures['serving_default']
# 
# 
# endpoints3 = prune(images)
# output3 = endpoints3['emb'].numpy()
# print('norm saved model v2', np.linalg.norm(data[0]-output3[0]))

# print(data[0])
# print(output3[0])


# Other
# 
# with tf.compat.v1.Session() as sess:
#     init = tf.compat.v1.global_variables_initializer()
#     sess.run(init)
#     tf.compat.v1.saved_model.loader.load(sess, ["serve"], saved_model_path)
#     out4 = sess.run('head/out_emb:0',feed_dict={'images:0': inputs.eval(session=sess)})
# 
# print('norm saved model v1', np.linalg.norm(data[0]-out4[0]))
# print('norm saved model v1 vs v2', np.linalg.norm(output3[0]-out4[0]))

# 