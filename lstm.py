import math
import tensorflow as tf
import numpy as np

# The Ruslana dataset has 6 classes, representing the 6 emotional states.
NUM_CLASSES = 1

def define_inference(features, num_units):
	"""
	Build the model up to where it may be used for inference.
 	
 	Args:
    	features: features placeholder, from inputs().
		hidden1_units: Size of the first hidden layer.
		hidden2_units: Size of the second hidden layer.
	Returns:
		softmax_linear: Output tensor with the computed logits.
	"""
	# Define an LSTM cell
	lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)

	# Define 2 LSTM cells
	# lstm_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]
	# lstm_cell = tf.nn.rnn_cell.Multicell(lstm_layers)

	# Create an RNN with LSTM cell 
	# outputs is a tensor of shape [batch_size, max_time, cell_state_size] - a result of linear activation of last layer of RNN
	# state is a tensor of shape [batch_size, cell_state_size]
	outputs, state = tf.nn.dynamic_rnn(lstm_cell, features, dtype=tf.float32)	

	# Reshape output to have the shape [max_time, batch_size, cell_state_size]
	outputs = tf.transpose(outputs, [1, 0, 2])

	# Get the output from the last time step  
	last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name='last_output')

	weights = tf.Variable(tf.truncated_normal([num_units, NUM_CLASSES]), name='fc_weights')
	biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='fc_biases')

	logits = tf.add(tf.matmul(last, weights), biases, name='logits')
	
	return logits, weights, biases

def define_loss(logits, labels, weights, beta=0.01): 
	"""Calculates the average loss from the logits and the labels.
	Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size].
	Returns:
		loss: Loss tensor of type float.
	"""
	mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels, logits))
	l2_loss = beta * tf.nn.l2_loss(weights)

	loss = tf.add(mse_loss, l2_loss, name='loss')
	
	# return tf.reduce_mean(cross_entropy, name='xentropy_mean')
	# return tf.reduce_mean(RMSE)
	return loss

def define_training(loss, learning_rate):
	"""Sets up the training Ops.
	Creates a summarizer to track the loss over time in TensorBoard.
	Creates an optimizer and applies the gradients to all trainable variables.
	The Op returned by this function is what must be passed to the
	`sess.run()` call to cause the model to train.
	Args:
		loss: Loss tensor, from loss().
		learning_rate: The learning rate to use for gradient descent.
	Returns:
		train_op: The Op for training.
	"""

	# Add a scalar summary for the snapshot loss
	tf.summary.scalar('loss', loss)

	# Create a gradient descent optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate, name='adam_optimizer')

	# Create a variable to track the global step
	global_step = tf.Variable(0, name='global_step', trainable=False)

	# Define a single train step, which applies the gradients to minimize the loss
	# and also increments the global step counter
	train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')
	return train_op

def build_model(sess, dimensions, parameters, save_path, show_progress=True):

	# Unwrap parameters
	seq_len = parameters['seq_len']
	num_units = parameters['num_units']
	learning_rate = parameters['learning_rate']
	batch_size = parameters['batch_size']
	f_dim, l_dim = dimensions
	high_val = parameters['high_thresh'] 
	low_val = parameters['low_thresh']

	model = {}

	def map_fn(t):
		t = tf.cond(tf.less_equal(t, low_val), 
					true_fn=lambda: 0., 
					false_fn=lambda: tf.cond(tf.greater_equal(t, high_val), 
											   true_fn=lambda: 2., 
											   false_fn=lambda: 1.))
		return t	

	# Generate placeholders for the data and labels
	x = tf.placeholder(tf.float32, [None, seq_len, f_dim], name='input_placeholder')
	y = tf.placeholder(tf.float32, [None, l_dim], name='label_placeholder')

	# Add to the Graph the Ops for calculating logits
	logits, weights, biases = define_inference(x, num_units)

	# Add to the Graph the Ops for loss calculation
	loss = define_loss(logits, y, weights)

	# Add to the Graph the Ops that calculate and apply gradients
	train_op = define_training(loss, learning_rate)

	# Add to the Graph the Ops that evaluate predictions 
	rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, logits))), name='rmse')
	# rmse = tf.metrics.root_mean_squared_error(y, logits, name='rmse')

	# Add an op that maps labels
	mapped_labels = tf.map_fn(map_fn, tf.squeeze(y), name='labels')

	# Add an op to get the predictions
	mapped_logits = tf.map_fn(map_fn, tf.squeeze(logits), name='predictions')

	# Add an op to calculate the number of correctly predicted samples 
	correct = tf.reduce_sum(tf.cast(tf.equal(mapped_logits, mapped_labels), tf.float32), name='correct')

	# Add an op to calculate confusion matrix
	confusion_matrix = tf.confusion_matrix(tf.squeeze(mapped_labels), tf.squeeze(mapped_logits), name='confusion_matrix')

	# # Instantiate s SummaryWriter to output summaries on the Graph
	# summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

	# # Build the summary tensor
	# summary = tf.summary.merge_all()

	init_g = tf.global_variables_initializer()
	init_l = tf.local_variables_initializer()

	# Add a saver for writting training checkpoints
	saver = tf.train.Saver()

	# Initialize the variables
	sess.run(init_g)
	sess.run(init_l)

	model_path = saver.save(sess, save_path + '\\lstm-model')

	model['x'] = x
	model['y'] = y
	model['logits'] = logits
	model['weights'] = weights
	model['biases'] = biases
	model['loss'] = loss
	model['predictions'] = mapped_logits
	model['labels'] = mapped_labels
	model['train_op'] = train_op 
	model['confusion_matrix'] = confusion_matrix
	model['correct'] = correct
	model['rmse'] = rmse
	model['saver'] = saver
	model['batch_size'] = batch_size
	model['model_path'] = model_path

	print(save_path)
	print(model_path)

	return model

def get_logits(sess, data, labels, model):

	x = model['x']
	y = model['y']

	logits = []
	for batch_f, batch_l in iemocap.batches(np.array(data), np.array(labels)):
		logits.append(sess.run(model['logits'], feed_dict={x: batch_f, y: batch_l}))
	logits = np.vstack(logits)
	return logits
