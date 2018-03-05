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

import progressbar as pb

from modules import lstm
from modules import iemocap

widgets = [pb.Percentage(), ' ', pb.Bar(pb.ETA())]

def batches(features, labels):
	'''
	Form minibatches 

	Arguments:
		features - np.array of shape 
		labels - np.array of shape 

	Returns:
		pair of batch_f, batch_l
	'''

	# j is the pointer to the start of the minibatch
	for j in range(0, len(features), FLAGS.batch_size): # от начала до конца с шагом batch_size
		batch_f = []
		batch_l = []

		# for each frame in the minibatch form a (sequence, label) pair 
		for frame in range(j, j + FLAGS.batch_size):
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

def print_parameters(learning_rate, seq_len, num_hidden, target, stats):
	os.system('cls')

	print('Train sessions: ', stats['train_sess'])
	print('Test sessions: ', stats['test_sess'])
	print('Number of train samples: ', stats['X_train'])
	print('Number of test samples: ', stats['X_test'])   
	print('Features dimension: ', stats['f_dim'])
	print('Labels dimension: ', stats['l_dim'])
	print()

	print('Start training:')
	print()
	print('Learning rate: ', learning_rate)
	print('Sequence length: ', seq_len)
	print('Model size: ', num_hidden)
	print('Training for: ', target)
	print()

def build_model(sess, f_dim, l_dim, show_progress=True):

	print()
	print('Building model...')

	# Instantiate s SummaryWriter to output summaries on the Graph
	summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

	# Generate placeholders for the data and labels
	x = tf.placeholder(tf.float32, [None, FLAGS.seq_len, f_dim])
	y = tf.placeholder(tf.float32, [None, l_dim])

	# Build the Graph that computes predictions from the inference model
	logits = lstm.inference(x, FLAGS.num_hidden)

	# Add to the Graph the Ops for loss calculation
	loss = lstm.loss(logits, y)

	# Add to the Graph the Ops that calculate and apply gradients
	train_op = lstm.training(loss, FLAGS.learning_rate)

	# Add to the Graph the Ops that evaluate predictions 
	eval_op = tf.metrics.root_mean_squared_error(y, logits)

	# Build the summary tensor
	summary = tf.summary.merge_all()

	# Add a saver for writting training checkpoints
	saver = tf.train.Saver()

	init_g = tf.global_variables_initializer()
	init_l = tf.local_variables_initializer()

	# Initialize the variables
	sess.run(init_g)
	sess.run(init_l)   

	return eval_op, train_op, loss, logits, x, y

def run_training(sess, X_train, y_train, X_test, y_test, lstm_model):

	# Unwrap the model 
	eval_op, train_op, loss, logits, x, y = lstm_model
	rmse_op, update = eval_op

	# Open the log file
	with open(FLAGS.log_dir + 'results\\train_log.txt', 'a') as log_file:
		
		log_file.write('------------------------------------------------------------------------\r'.format(FLAGS.learning_rate))
		log_file.write('Learning rate: {} \r'.format(FLAGS.learning_rate))
		log_file.write('Sequence length: {} \r'.format(FLAGS.seq_len))
		log_file.write('Training for: {} \r\n'.format(y_train.name))

		# Training loop 
		for step in xrange(FLAGS.max_steps):
			print()
			print('Epoch ', step)
			log_file.write('Epoch {} \r'.format(step))

			# Group train data by turn utterances  (1 turn - 1 label)
			grp = X_train.groupby(level=[0])
			total_loss = 0 
			predictions = []
			timer = pb.ProgressBar(widgets=widgets, maxval=len(grp)).start()

			# Process train data turn by turn 
			for i, (dialog, data) in zip(range(len(grp)), grp): 
				dialog_labels = np.array(pd.DataFrame(y_train, index=data.index, dtype=np.float32))
				dialog_features = np.array(data)

				# Form a minibatch and make predictions for train data 
				for batch_f, batch_l in batches(dialog_features, dialog_labels):
					batch_predictions = sess.run(logits, feed_dict={x: batch_f, y: batch_l})
					predictions.append(batch_predictions)

					# Run one step of training and get the loss 
					sess.run(train_op, feed_dict={x: batch_f, y: batch_l})
					batch_loss = sess.run(loss, feed_dict={x: batch_f, y: batch_l}) 
					total_loss += batch_loss 
				timer.update(i+1)
			timer.finish()
			
			# Calculate train RMSE error 
			predictions = np.vstack(predictions)
			rmse_train = rmse_op.eval(session=sess, feed_dict={y: np.expand_dims(y_train, axis=1), logits: predictions})
			update.eval(session=sess, feed_dict={y: np.expand_dims(y_train, axis=1), logits: predictions})
			rmse_train = rmse_op.eval(session=sess, feed_dict={y: np.expand_dims(y_train, axis=1), logits: predictions})

			# Group the test data by turn utterances
			grp = X_test.groupby(level=[0])
			predictions = []

			# Process test data turn by turn 
			for dialog, data in grp: 
				dialog_labels = np.array(pd.DataFrame(y_test, index=data.index, dtype=np.float32))
				dialog_features = np.array(data)

				# Form a minibatch and make predictions for test data 
				for batch_f, batch_l in batches(dialog_features, dialog_labels):
					batch_predictions = sess.run(logits, feed_dict={x: batch_f, y: batch_l})
					predictions.append(batch_predictions)

			# Calculate test RMSE error
			predictions = np.vstack(predictions)
			rmse_test = rmse_op.eval(session=sess, feed_dict={y: np.expand_dims(y_test, axis=1), logits: predictions})
			update.eval(session=sess, feed_dict={y: np.expand_dims(y_test, axis=1), logits: predictions})
			rmse_test = rmse_op.eval(session=sess, feed_dict={y: np.expand_dims(y_test, axis=1), logits: predictions})

			print('RMSE Train: ', rmse_train)
			print('RMSE Test: ', rmse_test)

			log_file.write('RMSE Train: {} \r'.format(rmse_train))
			log_file.write('RMSE Test: {} \r\n'.format(rmse_test))        

		log_file.write('\r\n')

def run_evaluation(sess, X_test, y_test, lstm_model):

	print()
	print('Start evaluation')

	f_dim = len(X_test.columns)
	_, train_op, loss, logits, x, y = lstm_model

	# Group test data by dialog ids  (1 turn - 1 label)
	grp = X_test.groupby(level=[0])
	total_loss = 0 
	predictions = []
	timer = pb.ProgressBar(widgets=widgets, maxval=len(grp)).start()

	# Process dialog by dialog 
	for i, (dialog, data) in zip(range(len(grp)), grp): 
		dialog_labels = np.array(pd.DataFrame(y_test, index=data.index, dtype=np.float32))
		dialog_features = np.array(data)

		# Form minibatches and make predictions for test data
		for batch_f, batch_l in batches(dialog_features, dialog_labels):
			batch_predictions = sess.run(logits, feed_dict={x: batch_f, y: batch_l})
			predictions.append(batch_predictions)
		timer.update(i+1)
	timer.finish()
	
	# Map predictions from regression values to 3 clusters
	predictions = np.vstack(predictions)
	predictions = [(lambda val: 0 if val <= 2 else (2 if val >= 3 else 1))(val) for val in predictions]
	predictions = np.vstack(predictions)
	predictions = np.float32(predictions)

	# Map the labels to same 3 clusters 
	labels = [(lambda val: 0 if val <= 2 else (2 if val >= 3 else 1))(val) for val in y_test]
	labels = np.vstack(labels)

	# Calculate the number of correct predictions and accuracy 
	correct = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32)).eval()
	accuracy = correct / len(predictions)
	
	print('Test accuracy: ', accuracy)
	print()
	
	with open(FLAGS.log_dir + 'results\\train_log.txt', 'a') as log_file:
		log_file.write('Test accuracy: {} \r\n'.format(accuracy)) 

def main(_):

	# Define sessions for training and testing 
	train_sessions = [1, 2, 3, 4]
	test_sessions = [5]

	# Read in labels
	y_train = iemocap.read_labels(train_sessions, FLAGS.label_dir, include_self_evaluation=False, apply_mapping=True)
	y_test = iemocap.read_labels(test_sessions, FLAGS.label_dir, include_self_evaluation=False, apply_mapping=True)

	# Read in data 
	X_train = iemocap.read_data(train_sessions, FLAGS.data_dir)
	X_test = iemocap.read_data(test_sessions, FLAGS.data_dir)

	# Process labels
	if FLAGS.mode == 'categorical':
		mapping = {'LA': 0, 'HP': 1, 'HN': 2, 'xxx': 3}
		y_train = y_train[y_train['emotion'] != 'xxx'].replace(mapping)
		y_test = y_test[y_test['emotion'] != 'xxx'].replace(mapping)
		X_train = pd.DataFrame(X_train, index=y_train.index, dtype=float32)
		X_test = pd.DataFrame(X_test, index=y_test.index, dtype=float32)
	
	elif FLAGS.mode == 'activation': 
		y_train = y_train['activation']
		y_test = y_test['activation']
	
	elif FLAGS.mode == 'valence':
		y_train = y_train['valence']
		y_test = y_test['valence']

	# Collect some statistics
	stats ={}
	stats['train_sess'] = train_sessions
	stats['test_sess'] = test_sessions
	stats['X_train'] = len(X_train)
	stats['y_train'] = len(y_train)
	stats['X_test'] = len(X_test)
	stats['y_test'] = len(y_test)
	stats['f_dim'] = X_train.iloc[0].shape[0]
	stats['l_dim'] = 1

	# Print the parameters of the training
	print_parameters(FLAGS.learning_rate, FLAGS.seq_len, FLAGS.num_hidden, y_train.name, stats)

	# Open a new tensorflow session
	with tf.Session() as sess:

		# Build an lstm model
		lstm_model = build_model(sess, stats['f_dim'], stats['l_dim'])
		
		# Start training on the train data 
		run_training(sess, X_train, y_train, X_test, y_test, lstm_model)
		
		# Start evaluation on the test data 
		run_evaluation(sess, X_test, y_test, lstm_model)
		
		# Well done!
		print('Program finished ^^')

if __name__ == '__main__': 
	
	parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument(
		'-m', '--mode', 
		type=str,
		default='activation',
		help='The target values')

	parser.add_argument(
		'-e', '--emotions', 
		type=str,
		default='basic3',
		help='Possible emotions: \n  Neutral\n  Happy\n  Angry\n  Sad\n  Scared\n  Excited\n  Frustrated\n  Surprised\n  Disgusted')

	parser.add_argument(
		'-d', '--dimensions', 
		type=str,
		default='va',
		help='Dimensional axes')

	parser.add_argument(
		'-l', '--learning_rate',
		type=float,
		default=0.01,
		help='Initial learning rate')

	parser.add_argument(
		'--max_steps',
		type=int,
		default=100,
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
