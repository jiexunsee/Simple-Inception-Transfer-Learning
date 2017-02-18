import tensorflow as tf
import numpy as np

data_inputs = np.loadtxt('data_inputs.txt')

with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('testsave.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	all_vars = tf.get_collection('vars')
	for v in all_vars:
		v_ = sess.run(v)
		print(v_)
	prediction = tf.get_collection('pred')[0]

	prediction = sess.run(prediction, feed_dict={inputs: data_inputs[0]})
