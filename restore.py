import numpy as np
import pandas as pd
import tensorflow as tf
from modules import iemocap
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file 

from modules import lstm

tf.set_random_seed(1234)

# Read in data 	
features = pd.read_pickle('features')
labels = pd.read_pickle('labels')

data, dimensions = iemocap.prepare_data(features, labels, 'activation')
X_train, X_test, y_train, y_test = data
f_dim, l_dim = dimensions

saver = tf.train.import_meta_graph('logs\\checkpoints\\lstm-model-45.meta')
graph = tf.get_default_graph()

x = graph.get_tensor_by_name('input_placeholder:0')
y = graph.get_tensor_by_name('label_placeholder:0')
# logits_op = graph.get_tensor_by_name('get_logits_op:0')
# # predictions_op = graph.get_tensor_by_name('map_to_predictions/TensorArray:0')

correct = graph.get_tensor_by_name('correct:0')
# predictions = graph.get_tensor_by_name('map_to_predictions/TensorArrayStack/TensorArrayGatherV3:0')
# labels_op = graph.get_tensor_by_name('map_to_labels/TensorArrayStack/TensorArrayGatherV3:0')
confusion_matrix = graph.get_tensor_by_name('confusion_matrix/SparseTensorDenseAdd:0')

with tf.Session() as sess:

	saver.restore(sess, tf.train.latest_checkpoint('logs\\checkpoints\\'))
	# sess.run(init_g)
	# sess.run(init_l)
	print()
	print('Model restored')
	print()

	# print(predictions_op.eval())
	
	num_correct = 0
	total = 0
	c_matrix = np.zeros((3,3), dtype=np.float32)
	for batch_f, batch_l in iemocap.batches(np.array(X_train), np.array(y_train)):

		num_correct += sess.run(correct, feed_dict={x: batch_f, y: batch_l})
		total += batch_f.shape[0]
		c_matrix += sess.run(confusion_matrix, feed_dict={x: batch_f, y: batch_l})

	print(c_matrix)



	# for batch_f, batch_l in batches(np.array(X_train), np.array(y_train)):
	# 	pred = sess.run(logits, feed_dict={x:batch_f, y: batch_l})

	# print_tensors_in_checkpoint_file(file_name=tf.train.latest_checkpoint('logs/checkpoints/activation/unnorm_80/'), tensor_name='', all_tensors=True)
	# tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]

	# for t in tensors:
	# 	print(t)
	
	
	
