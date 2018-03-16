import os
import argparse
import sys

import tensorflow as tf
import numpy as np 
import pandas as pd

from random import shuffle
from six.moves import xrange 
from itertools import islice
from argparse import RawTextHelpFormatter 
from collections import Counter 

import progressbar as pb
import matplotlib.pyplot as plt

from modules import lstm
from modules import iemocap

widgets = [pb.Percentage(), ' ', pb.Bar(pb.ETA())]

def batches(features, labels, batch_size=128):
	'''
	Form minibatches 

	Arguments:
		features - np.array of shape 
		labels - np.array of shape 

	Returns:
		pair of batch_f, batch_l
	'''
	# j is the pointer to the start of the minibatch
	for j in range(0, len(features), batch_size): # от начала до конца с шагом batch_size
		batch_f = []
		batch_l = []

		# for each frame in the minibatch form a (sequence, label) pair 
		for frame in range(j, j + batch_size):
			# break if all frames are processed 
			if frame >= len(features): 
				break
			
			f = [] # this will hold a sequence of features
			cur = frame # for current frame 
			
			# add label to the label list
			batch_l.append(labels[frame]) 

			# form a sequence
			for i in range(FLAGS.seq_len):
				f.append(features[cur])

				if cur > 0: 
					cur -= 1
				# else repeat first frame of the sequence

			# add the sequence to the list
			batch_f.append(np.array(f)[::-1]) # we've assembled sequence in reverse

		batch_f = np.stack(batch_f) # of shape [batch_size, seq_len, num_features]
		batch_l = np.stack(batch_l) # of shape [batch_size]

		# if batch_l.shape[0] != 1:
		yield batch_f, batch_l

def train_test_split(data, labels):

	id_list = []
	for i in data.index.get_level_values(0).unique():
		id_list.append(i)
	shuffle(id_list)

	thresh = round(0.9 * len(id_list)) 

	train_ids = id_list[0:thresh]
	test_ids = id_list[thresh:]

	train_index = data.index.drop(test_ids)
	test_index = data.index.drop(train_ids)

	X_train = pd.DataFrame(data, index=train_index, dtype=np.float32)
	X_test = pd.DataFrame(data, index=test_index, dtype=np.float32)

	y_train = pd.DataFrame(labels, index=X_train.index, dtype=np.float32)
	y_test = pd.DataFrame(labels, index=X_test.index, dtype=np.float32)

	return X_train, X_test, y_train, y_test
	
def prepare_data(data, labels):
	
	# Normalize data
	data = (data - data.mean()) / (data.max() - data.min())

	# Process labels
	if FLAGS.mode == 'categorical':
		mapping = {'LA': 0, 'HP': 1, 'HN': 2, 'xxx': 3}
		labels = labels[labels['emotion'] != 'xxx'].replace(mapping)
		data = pd.DataFrame(data, index=labels.index, dtype=np.float32)
	
	else: 
		labels = labels[FLAGS.mode]

	# Split train/test 
	X_train, X_test, y_train, y_test = train_test_split(data, labels)
	
	f_dim = X_train.iloc[0].shape[0]
	l_dim = 1

	return (X_train, X_test, y_train, y_test), (f_dim, l_dim)

def one_step_training(X_train, y_train, model, sess):

	_, train_op, loss, logits, x, y = model

	total_loss = 0 

	# Group train data by turn utterances  (1 turn - 1 label)
	grp = X_train.groupby(level=[0])
	timer = pb.ProgressBar(widgets=widgets, maxval=len(grp)).start()

	# Process train data turn by turn 
	for i, (dialog, data) in zip(range(len(grp)), grp): 
		dialog_labels = np.array(pd.DataFrame(y_train, index=data.index, dtype=np.float32))
		dialog_features = np.array(data)

		# Form a minibatch and make predictions for train data 
		for batch_f, batch_l in batches(dialog_features, dialog_labels):

			# Run one step of training and get the loss 
			sess.run(train_op, feed_dict={x: batch_f, y: batch_l})
			batch_loss = sess.run(loss, feed_dict={x: batch_f, y: batch_l}) 
			total_loss += batch_loss 
		timer.update(i+1)
	timer.finish()

	return total_loss

def plot_learning_curve(data):
	'''
	Arguments:
		data - a dictionary containing following keys:
			train_accuracy
			train_loss
			train_rmse
			train_uar
			test_accuracy
			test_rmse
			test_uar
	'''
	plt.figure(1)
	plt.plot(np.arange(FLAGS.max_steps), data['train_uar'], 'b', label='Train') 
	plt.plot(np.arange(FLAGS.max_steps), data['test_uar'], 'g', label='Test')
	plt.xlabel('Number of epochs')
	plt.ylabel('units of UAR')
	plt.title('Train/test UAR')
	plt.grid(True)
	plt.legend()
	
	plt.figure(2)
	plt.plot(np.arange(FLAGS.max_steps), data['train_accuracy'], 'b', label='Train') 
	plt.plot(np.arange(FLAGS.max_steps), data['test_accuracy'], 'g', label='Test')
	plt.xlabel('Number of epochs')
	plt.ylabel('Accuracy')
	plt.title('Train/test accurac')
	plt.grid(True)
	plt.legend()

	plt.figure(3)
	plt.plot(np.arange(FLAGS.max_steps), data['train_rmse'], 'b', label='Train') 
	plt.plot(np.arange(FLAGS.max_steps), data['test_rmse'], 'g', label='Test')
	plt.xlabel('Number of epochs')
	plt.ylabel('No units')
	plt.title('Train/test RMSE')
	plt.grid(True)
	plt.legend()

	plt.figure(4)
	plt.plot(np.arange(FLAGS.max_steps), data['train_loss'], 'b', label='Train') 
	plt.xlabel('Number of epochs')
	plt.ylabel('loss units')
	plt.title('Train loss')
	plt.grid(True)
	plt.legend()

	plt.show()

def run_evaluation(results, X_train, y_train, X_test, y_test, model, sess):

	# results = {'train_loss': [], 'train_rmse': [], 'test_rmse': [], 
	# 		   'train_accuracy': [], 'test_accuracy': [], 
	# 		   'train_uar': [], 'test_uar': []}

	rmse, accuracy, uar = evaluate(X_train, y_train, model, sess)

	# Save the results 
	results['train']['accuracy'].append(accuracy)
	results['train']['rmse'].append(rmse)
	results['train']['uar'].append(uar)

	rmse, accuracy, uar = evaluate(X_test, y_test, model, sess)

	# Save the results 
	results['test']['accuracy'].append(accuracy)
	results['test']['rmse'].append(rmse)
	results['test']['uar'].append(uar)

	return results 

def evaluate(features, labels, model, sess):
	
	# Unwrap the model
	eval_ops, train_op, loss, logits, x, y = model
	rmse, correct, confusion_matrix = eval_ops
	rmse_op, rmse_update = rmse

	# Initialize variables 
	rmse = 0
	num_correct = 0
	c_matrix = np.zeros((3, 3), dtype=int)
	num_batches = np.int(np.ceil(len(features) / FLAGS.batch_size))

	# Process in minibatches  
	timer = pb.ProgressBar(widgets=widgets, maxval=num_batches).start()
	for i, (batch_f, batch_l) in zip(range(num_batches), batches(np.array(features), np.array(labels))):
		
		rmse_update.eval(session=sess, feed_dict={y: batch_l, x: batch_f})	
		rmse += rmse_op.eval()
		num_correct += sess.run(correct, feed_dict={x: batch_f, y: batch_l})
		c_matrix += sess.run(confusion_matrix, feed_dict={y: batch_l, x: batch_f})		
		timer.update(i+1)
	timer.finish()
	
	rmse = rmse / num_batches	
	accuracy = num_correct / len(features)
	recall = np.diagonal(c_matrix) / np.sum(c_matrix, axis=1)
	uar = np.mean(recall)
	
	return rmse, accuracy, uar

def write_log_header(log_file, learning_rate, seq_len, mode):

	log_file.write('------------------------------------------------------------------------\r')
	log_file.write('Learning rate: {} \r'.format(learning_rate))
	log_file.write('Sequence length: {} \r'.format(seq_len))
	log_file.write('Training for: {} \r\n'.format(mode))

def update_log_file(log_file, step, results):

	log_file.write('Epoch {} \r'.format(step))
	log_file.write('RMSE Train: {} \r'.format(results['train']['rmse']))
	log_file.write('RMSE Test: {} \r'.format(results['test']['rmse']))
	log_file.write('Train accuracy: {} \r'.format(results['train']['accuracy']))
	log_file.write('Test accuracy: {} \r\n'.format(results['test']['accuracy']))
	log_file.write('Train uar: {} \r'.format(results['train']['uar']))
	log_file.write('Test uar: {} \r\n'.format(results['test']['uar']))

def run_training(sess, X_train, y_train, X_test, y_test, model):

	# Add a saver for writting training checkpoints
	saver = tf.train.Saver()

	# Create a dictionary that will store the evaluation results  
	results = {'train': {}, 'test': {}}
	for subset in ['train', 'test']:
		results[subset]['accuracy'] = []
		results[subset]['uar'] = []
		results[subset]['rmse'] = []

	# Open the log file
	with open(FLAGS.log_dir + 'results\\train_log.txt', 'a') as log_file:
		
		# Write a header to the log file 
		write_log_header(log_file, FLAGS.learning_rate, FLAGS.seq_len, FLAGS.mode)

		# Training loop 
		for step in xrange(FLAGS.max_steps):

			print()
			print('Epoch ', step)

			# Run one step of training and get the loss
			loss = one_step_training(X_train, y_train, model, sess)
			
			# Evaluate the model on the train data
			results = run_evaluation(results, X_train, y_train, X_test, y_test, model, sess)	

			# Print the results on the screen 
			print('Train uar: ', results['train']['uar'][-1])
			print('Test uar: ', results['test']['uar'][-1])

			# Write the results to the log file 
			update_log_file(log_file, step, results)

			# Save the model every 5th epoch 
			if step % 5 == 0:
				# Append the step number to the checkpoint name:
				saver.save(sess, 'logs\checkpoints\lstm-model', global_step=step)

		log_file.write('\r\n')

	return results

def build_model(sess, f_dim, l_dim, show_progress=True):

	print()
	print('Building model...')
	print()

	def fn(t):
		t = tf.cond(tf.less_equal(t, low_val), 
					true_fn=lambda: 0., 
					false_fn=lambda: tf.cond(tf.greater_equal(t, high_val), 
											   true_fn=lambda: 2., 
											   false_fn=lambda: 1.))
		return t	

	high_val = 3.5 
	low_val = 2.5

	# Instantiate s SummaryWriter to output summaries on the Graph
	summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

	# Generate placeholders for the data and labels
	x = tf.placeholder(tf.float32, [None, FLAGS.seq_len, f_dim], name='input_placeholder')
	y = tf.placeholder(tf.float32, [None, l_dim], name='label_placeholder')

	# Build the Graph that computes predictions from the inference model
	logits = lstm.inference(x, FLAGS.num_hidden)

	# Add an op that maps labels
	labels = tf.map_fn(fn, tf.squeeze(y))

	# Add an op to get the predictions
	predictions = tf.map_fn(fn, tf.squeeze(logits))

	# Add to the Graph the Ops for loss calculation
	loss = lstm.loss(logits, y)

	# Add to the Graph the Ops that calculate and apply gradients
	train_op = lstm.training(loss, FLAGS.learning_rate)

	# Add an op to calculate the number of correctly predicted samples 
	correct = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32))

	# Add an op to calculate confusion matrix
	confusion_matrix = tf.confusion_matrix(tf.squeeze(labels), tf.squeeze(predictions))

	# Add to the Graph the Ops that evaluate predictions 
	rmse = tf.metrics.root_mean_squared_error(y, logits, name='rmse')

	# Build the summary tensor
	summary = tf.summary.merge_all()

	init_g = tf.global_variables_initializer()
	init_l = tf.local_variables_initializer()

	# Initialize the variables
	sess.run(init_g)
	sess.run(init_l)   

	return (rmse, correct, confusion_matrix), train_op, loss, logits, x, y

def main(_):

	# Read in labels
	labels = iemocap.read_labels(FLAGS.label_dir, include_self_evaluation=False, apply_mapping=True)

	# Read in data
	features = iemocap.read_features(FLAGS.data_dir)

	# Preprocess 
	data, dimensions = prepare_data(features, labels)
	X_train, X_test, y_train, y_test = data 
	f_dim, l_dim = dimensions

	# Open a new tensorflow session
	with tf.Session() as sess:

		# Build an lstm model
		lstm_model = build_model(sess, f_dim, l_dim)
		
		# Start training 
		results = run_training(sess, X_train, y_train, X_test, y_test, lstm_model)

		# Plot the learning curve
		plot_learning_curve(results)		

		# Well done!
		print('Program finished ^^')

if __name__ == '__main__': 
	
	parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument(
		'-m', '--mode', 
		type=str,
		default='valence',
		help='The target values')

	parser.add_argument(
		'-e', '--emotions', 
		type=str,
		default='basic3',
		help='Possible emotions: \n  Neutral\n  Happy\n  Angry\n  Sad\n  Scared\n  Excited\n  Frustrated\n  Surprised\n  Disgusted')

	parser.add_argument(
		'-l', '--learning_rate',
		type=float,
		default=0.00001,
		help='Initial learning rate')

	parser.add_argument(
		'--max_steps',
		type=int,
		default=50,
		help='Number of steps to run the trainer')

	parser.add_argument(
		'--num_hidden', 
		type=int,
		default=128,
		help='Number of units in hidden layer 1')

	parser.add_argument(
		'-s', '--seq_len', 
		type=int,
		default=10,
		help='The sequence length')

	parser.add_argument(
		'--batch_size',
		type=int,
		default=128,
		help='Batch size. Must divide evenly into the dataset sizes')

	parser.add_argument(
		'--data_dir',
		type=str,
		default='C:\\Users\\SMIL\\dev\\tf\\workspace\\features\\iemocap\\',
		help='Directory to put the input data')

	parser.add_argument(
		'--label_dir',
		type=str,
		default='D:\\IEMOCAP_full_release\\',
		help='Directory to put the input data')

	parser.add_argument(
		'--log_dir',
		type=str,
		default='C:\\Users\\SMIL\\dev\\tf\\workspace\\lstm\\logs\\',    
		help='Directory to put the log data')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)    
