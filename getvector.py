import numpy as np
import os
import tensorflow as tf
import urllib.request
import matplotlib.pyplot as plt
import os
import dataset_utils
import inception_preprocessing
import inception_v3 as v3


def getvector(imagedir):
	slim = tf.contrib.slim

	batch_size = 3
	image_size = v3.inception_v3.default_image_size

	url = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
	checkpoints_dir = os.getcwd()

	if not tf.gfile.Exists(checkpoints_dir+'/inception_v3.ckpt'):
		dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

	with tf.Graph().as_default():

		#imagedir = '/home/jiexun/Desktop/Siraj/ImageChallenge/Necessary/train/cat.0.jpg'
		image_string = tf.read_file(imagedir)
		image = tf.image.decode_jpeg(image_string, channels=3)

		processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
		processed_images  = tf.expand_dims(processed_image, 0)

		# Create the model, use the default arg scope to configure the batch norm parameters.
		with slim.arg_scope(v3.inception_v3_arg_scope()):
		    vector, _ = v3.inception_v3(processed_images, num_classes=1001, is_training=False)

		init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v3.ckpt'), slim.get_model_variables('InceptionV3'))
		with tf.Session() as sess:
		    init_fn(sess)
		    np_image, vector = sess.run([image, vector])

		a = np.asarray([x for xs in vector for xss in xs for xsss in xss for x in xsss])
		np.reshape(a, (1,2048))

	return a