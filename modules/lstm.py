import math
import tensorflow as tf
import numpy as np

# The Ruslana dataset has 6 classes, representing the 6 emotional states.
NUM_CLASSES = 1

def inference(features, num_units):
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
	last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1, name='inference_gather')

	weights = tf.Variable(tf.truncated_normal([num_units, NUM_CLASSES]), name='weights')
	biases = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name='biases')

	logits = tf.matmul(last, weights) + biases
	return logits

def loss(logits, labels): 
	"""Calculates the average loss from the logits and the labels.
	Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size].
	Returns:
		loss: Loss tensor of type float.
	"""
	RMSE = tf.nn.l2_loss(labels - logits) 
	loss = tf.losses.mean_squared_error(labels, logits)
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
		labels=labels, logits=logits, name='xentropy')
	
	# return tf.reduce_mean(cross_entropy, name='xentropy_mean')
	# return tf.reduce_mean(RMSE)
	return tf.reduce_mean(loss)

def training(loss, learning_rate):
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

def evaluation(logits, labels):
	"""Evaluate the quality of the logits at predicting the label.
	Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size], with values in the
				range [0, NUM_CLASSES).
	Returns:
		A scalar int32 tensor with the number of examples (out of batch_size)
		that were predicted correctly.
	"""
	# Bool tensor that is true for examples where the label is in the top k 
	# (here k=1) of all logits for that example
	correct = tf.nn.in_top_k(logits, labels, 1)
	# correct = tf.nn.l2_loss(labels - logits) 
	return tf.reduce_mean(tf.cast(correct, tf.float32), name='eval_op')
