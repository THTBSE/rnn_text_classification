#-*- coding:utf-8 -*-

import tensorflow as tf
from model import BiRNN
from utils import InputHelper
import time
import os
import numpy as np

# Parameters
# =================================================
tf.flags.DEFINE_integer('embedding_size', 100, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 15, 'Sequence length (default : 32)')
tf.flags.DEFINE_integer('attn_size', 200, 'attention layer size')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 30, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_string('train_file', 'train.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model saved directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log info directiory')
tf.flags.DEFINE_string('pre_trained_vec', None, 'using pre trained word embeddings, npy file format')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')
tf.flags.DEFINE_integer('save_steps', 1000, 'num of train steps for saving model')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print '\nParameters:'
for attr, value in sorted(FLAGS.__flags.items()):
    print '{0}={1}'.format(attr.upper(), value)

def train():
	data_loader = InputHelper()
	data_loader.create_dictionary(FLAGS.data_dir+'/'+FLAGS.train_file, FLAGS.data_dir+'/')
	data_loader.create_batches(FLAGS.data_dir+'/'+FLAGS.train_file, FLAGS.batch_size, FLAGS.sequence_length)
	FLAGS.vocab_size = data_loader.vocab_size
	FLAGS.n_classes = data_loader.n_classes
	FLAGS.num_batches = data_loader.num_batches

	test_data_loader = InputHelper()
	test_data_loader.load_dictionary(FLAGS.data_dir+'/dictionary')
	test_data_loader.create_batches(FLAGS.data_dir+'/'+FLAGS.test_file, 100, FLAGS.sequence_length)

	if FLAGS.pre_trained_vec:
		embeddings = np.load(FLAGS.pre_trained_vec)
		print embeddings.shape
		FLAGS.vocab_size = embeddings.shape[0]
		FLAGS.embedding_size = embeddings.shape[1]

	if FLAGS.init_from is not None:
		assert os.path.isdir(FLAGS.init_from), '{} must be a directory'.format(FLAGS.init_from)
		ckpt = tf.train.get_checkpoint_state(FLAGS.init_from)
		assert ckpt,'No checkpoint found'
		assert ckpt.model_checkpoint_path,'No model path found in checkpoint'

	# Define specified Model
	model = BiRNN(embedding_size=FLAGS.embedding_size, rnn_size=FLAGS.rnn_size, layer_size=FLAGS.layer_size,	
		vocab_size=FLAGS.vocab_size, attn_size=FLAGS.attn_size, sequence_length=FLAGS.sequence_length,
		n_classes=FLAGS.n_classes, grad_clip=FLAGS.grad_clip, learning_rate=FLAGS.learning_rate)

	# define value for tensorboard
	tf.summary.scalar('train_loss', model.cost)
	tf.summary.scalar('accuracy', model.accuracy)
	merged = tf.summary.merge_all()

	# 调整GPU内存分配方案
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True

	with tf.Session(config=tf_config) as sess:
		train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())

		# using pre trained embeddings
		if FLAGS.pre_trained_vec:
			sess.run(model.embedding.assign(embeddings))
			del embeddings

		# restore model
		if FLAGS.init_from is not None:
			saver.restore(sess, ckpt.model_checkpoint_path)

		total_steps = FLAGS.num_epochs * FLAGS.num_batches
		for e in xrange(FLAGS.num_epochs):
			data_loader.reset_batch()
			for b in xrange(FLAGS.num_batches):
				start = time.time()
				x, y = data_loader.next_batch()
				feed = {model.input_data:x, model.targets:y, model.output_keep_prob:FLAGS.dropout_keep_prob}
				train_loss, summary,  _ = sess.run([model.cost, merged, model.train_op], feed_dict=feed)
				end = time.time()

				global_step = e * FLAGS.num_batches + b

				print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(global_step,
                            total_steps,
                            e, train_loss, end - start))

				if global_step % 20 == 0:
					train_writer.add_summary(summary, e * FLAGS.num_batches + b)

				if global_step % FLAGS.save_steps == 0:
					checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')		
					saver.save(sess, checkpoint_path, global_step=global_step)
					print 'model saved to {}'.format(checkpoint_path)

			test_data_loader.reset_batch()
			test_accuracy = []
			for i in xrange(test_data_loader.num_batches):
				test_x, test_y = test_data_loader.next_batch()
				feed = {model.input_data:test_x, model.targets:test_y, model.output_keep_prob:1.0}
				accuracy = sess.run(model.accuracy, feed_dict=feed)
				test_accuracy.append(accuracy)
			print 'test accuracy:{0}'.format(np.average(test_accuracy))

if __name__ == '__main__':
	train()
