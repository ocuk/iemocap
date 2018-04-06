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
import matplotlib.pyplot as plt
import pickle

from modules import lstm
from modules import iemocap
import evaluate

widgets = [pb.Percentage(), ' ', pb.Bar(pb.ETA())]

def initialize_results():

	results = {'train': {}, 'test': {}}
	results['train']['loss'] = []

	for subset in ['train', 'test']:
		results[subset]['accuracy'] = []
		results[subset]['uar'] = []
		results[subset]['rmse'] = []

	return results

def save_results(save_path, results):

	with open(save_path + 'train_results.txt', 'w') as f:
		f.write('Train rmse: ' + str(results['train']['rmse'][-1]) + '\r')
		f.write('Test rmse: ' + str(results['test']['rmse'][-1])+ '\r')
		f.write('Train accuracy: ' + str(results['train']['accuracy'][-1])+ '\r')
		f.write('Test accuracy: ' + str(results['test']['accuracy'][-1])+ '\r')
		f.write('Train uar: ' + str(results['train']['uar'][-1])+ '\r')
		f.write('Test uar: ' + str(results['test']['uar'][-1]))

def create_model_name(parameters):

	model_name = 'lstm'
	model_name += '_' + str(parameters['mode'])
	model_name += '_' + str(parameters['learning_rate'])
	model_name += '_' + str(parameters['seq_len'])
	model_name += '_' + str(parameters['num_units'])
	model_name += '_' + str(parameters['max_steps'])

	return model_name

def create_param_dict(seq_len, num_units, learning_rate, max_steps, batch_size, model_prefix, mode, high_val, low_val):

	parameters = {}

	parameters['seq_len'] = seq_len
	parameters['num_units'] = num_units
	parameters['learning_rate'] = learning_rate
	parameters['mode'] = mode
	parameters['max_steps'] = max_steps
	parameters['batch_size'] = batch_size
	parameters['high_thresh'] = high_val 
	parameters['low_thresh'] = low_val

	return parameters

def plot_learning_curve(data, max_steps):

	plt.figure(1)
	plt.plot(np.arange(max_steps), data['train']['uar'], 'b', label='Train') 
	plt.plot(np.arange(max_steps), data['test']['uar'], 'g', label='Test')
	plt.xlabel('Number of epochs')
	plt.ylabel('units of UAR')
	plt.title('Train/test UAR')
	plt.grid(True)
	plt.legend()
	
	plt.figure(2)
	plt.plot(np.arange(max_steps), data['train']['accuracy'], 'b', label='Train') 
	plt.plot(np.arange(max_steps), data['test']['accuracy'], 'g', label='Test')
	plt.xlabel('Number of epochs')
	plt.ylabel('Accuracy')
	plt.title('Train/test accurac')
	plt.grid(True)
	plt.legend()

	plt.figure(3)
	plt.plot(np.arange(max_steps), data['train']['rmse'], 'b', label='Train') 
	plt.plot(np.arange(max_steps), data['test']['rmse'], 'g', label='Test')
	plt.xlabel('Number of epochs')
	plt.ylabel('No units')
	plt.title('Train/test RMSE')
	plt.grid(True)
	plt.legend()

	plt.figure(4)
	plt.plot(np.arange(max_steps), data['train']['loss'], 'b', label='Train') 
	plt.xlabel('Number of epochs')
	plt.ylabel('loss units')
	plt.title('Train loss')
	plt.grid(True)
	plt.legend()

	plt.show()

def write_log_header(log_file, learning_rate, seq_len, mode):

	log_file.write('------------------------------------------------------------------------\r')
	log_file.write('Learning rate: {} \r'.format(learning_rate))
	log_file.write('Sequence length: {} \r'.format(seq_len))
	log_file.write('Training for: {} \r\n'.format(mode))

def update_log_file(log_file, step, results):

	log_file.write('Epoch {} \r'.format(step))
	log_file.write('RMSE Train: {:10.4f} \r'.format(results['train']['rmse'][-1]))
	log_file.write('RMSE Test: {:10.4f} \r'.format(results['test']['rmse'][-1]))
	log_file.write('Train accuracy: {:10.2f} \r'.format(results['train']['accuracy'][-1]))
	log_file.write('Test accuracy: {:10.2f} \r'.format(results['test']['accuracy'][-1]))
	log_file.write('Train uar: {:10.2f} \r'.format(results['train']['uar'][-1]))
	log_file.write('Test uar: {:10.2f} \r\n'.format(results['test']['uar'][-1]))

def one_step_training(sess, X_train, y_train, model, parameters):

	train_op, loss = model['train_op'], model['loss']
	seq_len = parameters['seq_len']
	x, y = model['x'], model['y']

	total_loss = 0 

	# Group train data by turn utterances  (1 turn - 1 label)
	grp = X_train.groupby(level=[0])
	timer = pb.ProgressBar(widgets=widgets, maxval=len(grp)).start()

	# Process train data turn by turn 
	for i, (dialog, data) in zip(range(len(grp)), grp): 
		dialog_labels = np.array(pd.DataFrame(y_train, index=data.index, dtype=np.float32))
		dialog_features = np.array(data)

		# Form a minibatch and make predictions for train data 
		for batch_f, batch_l in iemocap.batches(dialog_features, dialog_labels, seq_len=seq_len):

			# Run one step of training and get the loss 
			sess.run(train_op, feed_dict={x: batch_f, y: batch_l})
			batch_loss = sess.run(loss, feed_dict={x: batch_f, y: batch_l}) 
			total_loss += batch_loss 
		timer.update(i+1)
	timer.finish()

	return total_loss

def run_training(sess, data, parameters, model):   	

	# Get the required parameters
	max_steps = parameters['max_steps']

	# Unpack the data
	X_train, X_test, y_train, y_test = data

	# Get the saver op
	saver = model['saver']
	save_path = model['model_path']
	
	# Create a dictionary that will store the evaluation results  
	results = initialize_results()
	
	# Open the log file
	with open(FLAGS.log_dir + 'results\\train_log.txt', 'a') as log_file:
		
		# Write a header to the log file 
		write_log_header(log_file, FLAGS.learning_rate, FLAGS.seq_len, FLAGS.mode)

		# Training loop 
		for step in xrange(max_steps):

			print()
			print('Epoch ', step)

			# Run one step of training and get the loss
			loss = one_step_training(sess, X_train, y_train, model, parameters)
			results['train']['loss'].append(loss)
			
			# Evaluate the model on the train data
			evaluate.run_evaluation(sess, results, X_train, y_train, X_test, y_test, model, parameters)	

			# Print the results on the screen 
			print('Train rmse: ', results['train']['rmse'][-1])
			print('Test rmse: ', results['test']['rmse'][-1])

			# Write the results to the log file 
			update_log_file(log_file, step, results)

			# Save the model every 5th epoch 
			if step % 5 == 0:
				# Append the step number to the checkpoint name:
				saver.save(sess, save_path, global_step=step, write_meta_graph=False)

		log_file.write('\r\n')

	return results

def main(_):

	# Define the parameters of the model
	seq_len = FLAGS.seq_len
	num_units = FLAGS.num_units
	learning_rate = FLAGS.learning_rate
	max_steps = FLAGS.max_steps
	batch_size = FLAGS.batch_size
	model_prefix = FLAGS.model_prefix
	mode = FLAGS.mode
	low_val = 2.3
	high_val = 3.5 
	
	# Pack the parameters in the dictionary
	parameters = create_param_dict(seq_len, num_units, learning_rate, max_steps, 
				  batch_size, model_prefix, mode, high_val, low_val)

	# Create a unique model name and path
	model_name = create_model_name(parameters)
	model_path = FLAGS.model_prefix + model_name

	# Create a folder for storing the model
	if not os.path.exists(model_path):
		os.makedirs(model_path)
		
	# Read in data 	
	features = pd.read_pickle('features')
	labels = pd.read_pickle('labels')
	
	# Preprocess 
	data, dimensions = iemocap.prepare_data(features, labels, mode) 

	# Open a new tensorflow session
	with tf.Session() as sess:

		# Build an lstm model
		lstm_model = lstm.build_model(sess, dimensions, parameters, model_path)
		
		# Start training 
		results = run_training(sess, data, parameters, lstm_model)

		# Plot the learning curve
		plot_learning_curve(results, parameters['max_steps'])	

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
		'--num_units', 
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

	parser.add_argument(
		'--model_prefix',
		type=str,
		default='C:\\Users\\SMIL\\dev\\tf\\workspace\\lstm\\logs\\checkpoints\\',    
		help='Directory to put the log data')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)    
