#-*- coding:utf-8 -*-


import tensorflow as tf
from utils import InputHelper
from model import BiRNN

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


def main():
	data_loader = InputHelper()
	data_loader.create_dictionary(FLAGS.data_dir+'/'+FLAGS.train_file, FLAGS.data_dir+'/')
	FLAGS.vocab_size = data_loader.vocab_size
	FLAGS.n_classes = data_loader.n_classes

	model = BiRNN(FLAGS.rnn_size, FLAGS.layer_size, FLAGS.vocab_size, 
		FLAGS.batch_size, FLAGS.sequence_length, FLAGS.n_classes, FLAGS.grad_clip)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		while True:
			x = raw_input('请输入一个地址:\n')
			x = [data_loader.transform_raw(x, FLAGS.sequence_length)]

			labels = model.inference(sess, data_loader.labels, x)
			print labels

if __name__ == '__main__':
	main()