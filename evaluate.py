import numpy as np
import pandas as pd
import tensorflow as tf

import argparse
import sys

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file 
from argparse import RawTextHelpFormatter 

from modules import lstm
from modules import iemocap

def run_evaluation(sess, results, X_train, y_train, X_test, y_test, model, parameters):

	# results = {'train_loss': [], 'train_rmse': [], 'test_rmse': [], 
	# 		   'train_accuracy': [], 'test_accuracy': [], 
	# 		   'train_uar': [], 'test_uar': []}
	X = {}
	y = {}
	X['train'] = X_train
	X['test'] = X_test
	y['train'] = y_train
	y['test'] = y_test

	for subset in ['train', 'test']:

		rmse, accuracy, uar = get_metrics(sess, X[subset], y[subset], model, parameters)

		# Save the results 
		results[subset]['accuracy'].append(accuracy)
		results[subset]['rmse'].append(rmse)
		results[subset]['uar'].append(uar)

def get_metrics(sess, features, labels, model, parameters):
	
	# Unwrap the model
	x, y = model['x'], model['y']
	rmse_op, correct, confusion_matrix = model['rmse'], model['correct'], model['confusion_matrix']
	# rmse_op, rmse_update = rmse

	# Initialize variables
	rmse = 0
	num_correct = 0
	c_matrix = np.zeros((3, 3), dtype=int)
	num_batches = np.int(np.ceil(len(features) / model['batch_size']))

	# Process in minibatches
	for batch_f, batch_l in iemocap.batches(np.array(features), np.array(labels), seq_len=parameters['seq_len']):
		
		# rmse_update.eval(session=sess, feed_dict={y: batch_l, x: batch_f})	
		rmse += sess.run(rmse_op, feed_dict={x: batch_f, y: batch_l})
		num_correct += sess.run(correct, feed_dict={x: batch_f, y: batch_l})
		c_matrix += sess.run(confusion_matrix, feed_dict={y: batch_l, x: batch_f})		
	
	rmse = rmse / num_batches	
	accuracy = num_correct / len(features)
	recall = np.diagonal(c_matrix) / np.sum(c_matrix, axis=1)
	uar = np.mean(recall)
	
	return rmse, accuracy, uar

def restore_model(sess, model_dir, model_name, batch_size=128):

	model = {}

	saver = tf.train.import_meta_graph(model_dir + model_name + '.meta')
	graph = tf.get_default_graph()

	x = graph.get_tensor_by_name('input_placeholder:0')
	y = graph.get_tensor_by_name('label_placeholder:0')

	logits = graph.get_tensor_by_name('logits:0')
	predictions = graph.get_tensor_by_name('predictions/TensorArrayStack/TensorArrayGatherV3:0')
	labels = graph.get_tensor_by_name('labels/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0')

	rmse_update = graph.get_tensor_by_name('rmse:0')
	correct = graph.get_tensor_by_name('correct:0')
	confusion_matrix = graph.get_tensor_by_name('confusion_matrix/SparseTensorDenseAdd:0')

	init_g = graph.get_operation_by_name('init')
	init_l = graph.get_operation_by_name('init_1')

	weights = graph.get_tensor_by_name('fc_weights:0')
	biases = graph.get_tensor_by_name('fc_biases:0')

	# sess.run(init_g)
	sess.run(init_l)

	model['x'] = x
	model['y'] = y
	model['weights'] = weights
	model['biases'] = biases
	model['logits'] = logits
	model['labels'] = labels
	model['predictions'] = predictions
	model['confusion_matrix'] = confusion_matrix
	model['correct'] = correct
	model['rmse'] = rmse_update
	model['saver'] = saver
	model['batch_size'] = batch_size

	return model

def initialize_results():

	results = {'train': {}, 'test': {}}
	for subset in ['train', 'test']:
		results[subset]['accuracy'] = []
		results[subset]['uar'] = []
		results[subset]['rmse'] = []

	return results

def get_logits(sess, data, labels, model, parameters):

	x = model['x']
	y = model['y']

	logits = []
	for batch_f, batch_l in iemocap.batches(np.array(data), np.array(labels), seq_len=parameters['seq_len']):
		logits.append(sess.run(model['logits'], feed_dict={x: batch_f, y: batch_l}))
	logits = np.vstack(logits)
	return logits

def save_results(save_path, results):

	with open(save_path + 'eval_results.txt', 'w') as f:
		f.write('Train rmse: ' + str(results['train']['rmse']) + '\r')
		f.write('Test rmse: ' + str(results['test']['rmse'])+ '\r')
		f.write('Train accuracy: ' + str(results['train']['accuracy'])+ '\r')
		f.write('Test accuracy: ' + str(results['test']['accuracy'])+ '\r')
		f.write('Train uar: ' + str(results['train']['uar'])+ '\r')
		f.write('Test uar: ' + str(results['test']['uar']))

def main(_):

	# Read in data 	
	features = pd.read_pickle('features')
	labels = pd.read_pickle('labels')

	data, dimensions = iemocap.prepare_data(features, labels, 'activation')
	X_train, X_test, y_train, y_test = data
	f_dim, l_dim = dimensions

	# Initialize a dictionary that will store the results 
	results = initialize_results()
	
	with tf.Session() as sess:

		lstm_model = restore_model(sess, FLAGS.chkpt_dir, FLAGS.model_name)
		lstm_model['saver'].restore(sess, tf.train.latest_checkpoint('logs\\checkpoints\\'))

		run_evaluation(sess, results, X_train, y_train, X_test, y_test, lstm_model)
		print(results['train']['rmse'])

		# print('train rmse: ', results['train']['rmse'][-1])
		# print('test rmse: ', results['test']['rmse'][-1])

		logits = lstm_model['logits']
		labels = lstm_model['labels']
		y = lstm_model['y']

		lgts = get_logits(sess, X_train, y_train, lstm_model)
		prds = sess.run(lstm_model['predictions'], feed_dict={logits: lgts})
		lbls = sess.run(lstm_model['labels'], feed_dict={y: np.array(y_train)})

		rmse = sess.run(lstm_model['rmse'], feed_dict={logits: lgts, y: np.array(y_train)})

		eval_weights = 'C:\\Users\\SMIL\\dev\\tf\\workspace\\lstm\\logs\\results\\eval_weights.txt'
		eval_bias = 'C:\\Users\\SMIL\\dev\\tf\\workspace\\lstm\\logs\\results\\eval_bias.txt'
		eval_logits = 'C:\\Users\\SMIL\\dev\\tf\\workspace\\lstm\\logs\\results\\eval_logits.txt'
		eval_predictions = 'C:\\Users\\SMIL\\dev\\tf\\workspace\\lstm\\logs\\results\\eval_predictions.txt'
		
		np.savetxt(eval_weights, lstm_model['weights'].eval())
		np.savetxt(eval_bias, lstm_model['biases'].eval())
		np.savetxt(eval_logits, lgts)
		np.savetxt(eval_predictions, prds)

		save_results('C:\\Users\\SMIL\\dev\\tf\\workspace\\lstm\\logs\\results\\', results)

if __name__ == '__main__': 
	
	parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
	parser.add_argument(
		'--chkpt_dir', 
		type=str,
		default='logs\\checkpoints\\',
		help='The checkpoint directory')

	parser.add_argument(
		'--model_name', 
		type=str,
		default='lstm-model',
		help='The name of the model')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)    