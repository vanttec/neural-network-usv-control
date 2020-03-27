from boat import Boat
import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf2
import matplotlib.pyplot as plt

tf.disable_eager_execution()

def weight_variable(shape):
	'''Create a weight variable for a neural network
	Params
	======
		shape: int list, shape for the weight variable
	Returns
	======
		rank-len(shape) tensor, weight variable
	'''
	initial = tf.truncated_normal(shape, stddev=1e-3, dtype=tf.float32)
	return tf.Variable(initial)

class TfToNp():

	def __init__(self, num_hidden_units=[64, 64], model_name=None):

		#self.boat = boat

		self.action_network_num_inputs = 5
		self.action_network_num_outputs = 1
		self.num_hidden_units = num_hidden_units

		tf.reset_default_graph()

		self.W1 = weight_variable([self.action_network_num_inputs, self.num_hidden_units[0]])
		self.b1 = weight_variable([1, self.num_hidden_units[0]])
		self.W2 = weight_variable([self.action_network_num_inputs +
								   self.num_hidden_units[0], self.num_hidden_units[1]])
		self.b2 = weight_variable([1, self.num_hidden_units[1]])
		self.W3 = weight_variable([self.action_network_num_inputs + self.num_hidden_units[0] +
								   self.num_hidden_units[1], self.action_network_num_outputs])
		self.b3 = weight_variable([1, self.action_network_num_outputs])

		self.sess = tf.Session()
		self.saver = tf.train.Saver()

		if model_name is not None:
		    self.restore_model(model_name)
		else:
		    self.sess.run(tf.global_variables_initializer())
		self.to_numpy_model(model_name)

	def restore_model(self, model_name):
		'''Restore weight variables
		Params
		======
		    model_name: string, name of the model to restore
		'''
		self.saver.restore(self.sess, './models/' + model_name)

	def to_numpy_model(self, model_name):
		'''Convert tensorflow weight variables into numpy weight variables'''
		self.np_W1 = self.sess.run(self.W1)
		self.np_b1, = self.sess.run(self.b1)
		self.np_W2 = self.sess.run(self.W2)
		self.np_b2, = self.sess.run(self.b2)
		self.np_W3 = self.sess.run(self.W3)
		self.np_b3, = self.sess.run(self.b3)

		np.savez(model_name, w1=self.np_W1, b1=self.np_b1, w2=self.np_W2, b2=self.np_b2, w3=self.np_W3, b3=self.np_b3)

# Eta limits
xlim = [-2.5, 2.5]
ylim = [-2.5, 2.5]
yawlim = [-np.pi, np.pi]
# Random upsilon limits
xvel_lim = [-0.0, 0.0]
yvel_lim = [-0.0, 0.0]
yaw_ang_vel_lim = [-0.0, 0.0]
# Lists of parameters
eta_limits = [xlim, ylim, yawlim]
upsilon_limits = [xvel_lim, yvel_lim, yaw_ang_vel_lim]
boat = Boat(random_eta_limits=eta_limits, random_upsilon_limits=upsilon_limits)
