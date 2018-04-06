import tensorflow as tf
import pandas as pd
import numpy as np

from modules import iemocap
from modules import lstm

NUM_CLASSES = 1
NUM_UNITS = 2
seq_len = 3

high_val = 3.5 
low_val = 2.5

tf.set_random_seed(1234)

def fn(t):
	t = tf.cond(tf.less_equal(t, low_val), 
				true_fn=lambda: 0., 
				false_fn=lambda: tf.cond(tf.greater_equal(t, high_val), 
										   true_fn=lambda: 2., 
										   false_fn=lambda: 1.))

	return t

features = pd.read_pickle('features')
labels = pd.read_pickle('labels')

data, dimensions = iemocap.prepare_data(features, labels, 'activation')
X_train, X_test, y_train, y_test = data
f_dim, l_dim = dimensions

# # Generate placeholders for the data and labels
x = tf.placeholder(tf.float32, [None, seq_len, f_dim], name='input_placeholder')
y = tf.placeholder(tf.float32, [None, 1], name='label_placeholder')

weights = tf.Variable(tf.truncated_normal([NUM_UNITS, NUM_CLASSES]), name='weights')
biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='biases')

# weights = tf.get_variable('fc_weights', dtype=tf.float32, initializer=tf.constant(1.0, shape=[NUM_UNITS, NUM_CLASSES]))
# bias = tf.get_variable('fc_bias', dtype=tf.float32, initializer=tf.constant(0.1, shape=[NUM_CLASSES]))

logits_op = lstm.inference(x, NUM_UNITS)
flatten = tf.reshape(logits_op, shape=[-1])
predictions = tf.map_fn(fn, flatten, name='map_to_predictions')
labels = tf.map_fn(fn, tf.reshape(y, shape=[-1]), name='map_to_labels')

# Add an op to calculate the number of correctly predicted samples 
correct = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32), name='correct')

# Add an op to calculate confusion matrix
confusion_matrix = tf.confusion_matrix(tf.squeeze(predictions), tf.squeeze(labels), name='confusion_matrix')

init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

# # # Add a saver for writting training checkpoints
saver = tf.train.Saver()

with tf.Session() as sess:
	
# 	# Initialize the variables
	sess.run(init_g)
	sess.run(init_l)

	num_correct = 0
	total = 0
	for batch_f, batch_l in iemocap.batches(np.array(X_train), np.array(y_train), batch_size=128, seq_len=seq_len):
		_ = sess.run(correct, feed_dict={x: batch_f, y: batch_l})
		num_correct += _ 
		total += batch_f.shape[0]

	print(num_correct)
	print(total)
	saver.save(sess, 'logs\\checkpoints\\test\\test-model')