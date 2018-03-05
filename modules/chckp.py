import tensorflow as tf
# from tensorflow.python.tools import inspect_checkpoint as chkp

log_dir = 'C:\\Users\\overk\\workspace\\tensorflow\\mnist\\logs\\fully_connected_feed\\model.ckpt'

saver = tf.train.Saver()

with tf.Session() as sess:
	
	saver.restore(sess, log_dir)

# chkp.print_tensors_in_checkpoint_file(log_dir, tensor_name='', all_tensors=True)

