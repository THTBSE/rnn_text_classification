#-*- coding:utf-8 -*-

import tensorflow as tf
from model import BiRNN
from utils import InputHelper
import time
import os

# Parameters
# =================================================

tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 15, 'Sequence length (default : 32)')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 30, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.002, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.97, 'decay rate for rmsprop')
tf.flags.DEFINE_string('train_file', 'train.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'data directory')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def train():
	data_loader = InputHelper()
	data_loader.create_dictionary(FLAGS.data_dir+'/'+FLAGS.train_file, FLAGS.data_dir+'/')
	data_loader.create_batches(FLAGS.data_dir+'/'+FLAGS.train_file, FLAGS.batch_size, FLAGS.sequence_length)
	FLAGS.vocab_size = data_loader.vocab_size
	FLAGS.n_classes = data_loader.n_classes

	test_data_loader = InputHelper()
	test_data_loader.load_dictionary(FLAGS.data_dir+'/dictionary')
	test_data_loader.create_batches(FLAGS.data_dir+'/'+FLAGS.test_file, 1000, FLAGS.sequence_length)


	model = BiRNN(FLAGS.rnn_size, FLAGS.layer_size, FLAGS.vocab_size, 
		FLAGS.batch_size, FLAGS.sequence_length, FLAGS.n_classes, FLAGS.grad_clip)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())

		for e in xrange(FLAGS.num_epochs):
			data_loader.reset_batch()
			sess.run(tf.assign(model.lr, FLAGS.learning_rate * (FLAGS.decay_rate ** e)))
			for b in xrange(data_loader.num_batches):
				start = time.time()
				x, y = data_loader.next_batch()
				feed = {model.input_data:x, model.targets:y, model.output_keep_prob:FLAGS.dropout_keep_prob}
				train_loss, _ = sess.run([model.cost, model.train_op], feed_dict=feed)
				end = time.time()
				print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            FLAGS.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))

			test_data_loader.reset_batch()
			for i in xrange(test_data_loader.num_batches):
				test_x, test_y = test_data_loader.next_batch()
				feed = {model.input_data:test_x, model.targets:test_y, model.output_keep_prob:1.0}
				accuracy = sess.run(model.accuracy, feed_dict=feed)
				print 'accuracy:{0}'.format(accuracy)

			checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches)
			print 'model saved to {}'.format(checkpoint_path)


if __name__ == '__main__':
	train()