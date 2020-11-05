import re
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


class LARSVariant(tf.train.Optimizer):

	def __init__(self,
				 learning_rate,
				 momentum = 0,
				 weight_decay = 0,
				 trust_coef = 0.001,
				 epsilon = 1e-5,
				 next_ord = '1',	# '1' or 'inf'
				 current_epoch = 0,
				 lp_epoch = None,	# he epoch to turn to use lp norm
				 norm_beta = 0,	# the moving average coefficient for weight norm
				 adadecay_alpha = 4,
				 exclude=['batch_normalization', 'LayerNorm', 'layer_norm', 'bias'], # use SGD
				 name = "LARSVariant"):

		super(LARSVariant, self).__init__(use_locking=False, name=name)
		if not learning_rate >= 0.0:
			raise ValueError("Invalid learning rate: {}".format(learning_rate))
		if not momentum >= 0.0:
			raise ValueError("Invalid momentum: {}".format(momentum))
		if not weight_decay >= 0.0:
			raise ValueError("Invalid weight decay: {}".format(weight_decay))
		if not trust_coef > 0:
			raise ValueError("Invalid trust coefficient: {}".format(trust_coef))
		if not epsilon > 0:
			raise ValueError("Invalid epsilon: {}".format(epsilon))
		if next_ord is not '1' and next_ord is not 'inf':
			raise ValueError("Invalid norm order: {}".format(next_ord))
		if not norm_beta >= 0 and not norm_beta <= 1:
			raise ValueError("Invalid norm moving average coefficient: {}".format(norm_beta))
        
		self.lr = learning_rate
		self.momemtum = momentum
		self.weight_decay = weight_decay
		self.trust_coef = trust_coef
		self.epsilon = epsilon
		if next_ord is '1':
			self.next_ord = 1,
		elif next_ord is 'inf':
			self.next_ord = np.inf
		self.current_ord = 2
		self.lp_epoch = lp_epoch
		self.current_epoch = current_epoch
		self.norm_beta = norm_beta
		self.adadecay_alpha = adadecay_alpha
		self.exclude = exclude


	def apply_gradients(self, grads_and_vars, global_step=None, name=None):

		assignments = []
		for (grad, param) in grads_and_vars:
			if grad is None or param is None:
				continue

			param_name = self._get_variable_name(param.name)

			if not self.whether_to_exclude(param_name):					
				# Use LARSVariant to optimize

				w_norm = linalg_ops.norm(param, ord=self.current_ord)
				g_norm = linalg_ops.norm(grad, ord=self.current_ord)

				# Get the moving average for weight/grad norm
				w_m = tf.get_variable(
					name=param_name + "/weight_m",
					shape=param.shape.as_list(),
					dtype=tf.float32,
					trainable=False,
					initializer=tf.zeros_initializer()
				)

				g_m = tf.get_variable(
					name=param_name + "/grad_m",
					shape=param.shape.as_list(),
					dtype=tf.float32,
					trainable=False,
					initializer=tf.zeros_initializer()
				)

				next_w_m = (tf.multiply(self.norm_beta, w_m)
							+ tf.multiply(1 - self.norm_beta, w_norm))
				next_g_m = (tf.multiply(self.norm_beta, g_m)
							+ tf.multiply(1 - self.norm_beta, g_norm))

				# Calculate the trust ratio
				trust_ratio = array_ops.where(math_ops.greater(w_norm, 0),
					array_ops.where(math_ops.greater(g_norm, 0),
					(self.trust_coef * next_w_m
					/ (next_g_m + self.weight_decay * next_w_m + self.epsilon)),
					1.0),
				1.0)

				if self.weight_decay != 0:
					grad = grad + self.weight_decay * param

				grad = grad * trust_ratio

				if self.momemtum != 0:
					# Get momentum or create it with zero initializer
					m = tf.get_variable(
						name=param_name + "/momentum",
						shape=param.shape.as_list(),
						dtype=tf.float32,
						trainable=False,
						initializer=tf.zeros_initializer()
					)

					next_m = (tf.multiply(self.momemtum, m)
							+ tf.multiply(1-self.momemtum, grad))

					assignments.extend([m.assign(next_m)])

					update = self.lr * next_m
				
				else:
					update = self.lr * grad
				
				next_param = param - update

				assignments.extend(
					[param.assign(next_param)]
				)
			
			else:
				# Use SGD (with momentum and weight decay optionally)

				if self.weight_decay != 0:

					grad = grad + self.weight_decay * param # grad.assign()?

				if self.momemtum != 0:
					# Get momentum or create it with zero initializer
					m = tf.get_variable(
						name=param_name + "/momentum",
						shape=param.shape.as_list(),
						dtype=tf.float32,
						trainable=False,
						initializer=tf.zeros_initializer()
					)

					next_m = (tf.multiply(self.momemtum, m)
							+ tf.multiply(1-self.momemtum, grad))

					assignments.extend([m.assign(next_m)])
				
					update = self.lr * next_m

				else:
					update = self.lr * grad

				next_param = param - update

				assignments.extend([param.assign(next_param)])

		return tf.group(*assignments, name=name)
	
	def whether_to_exclude(self, param_name):
		"""Whether to use SGD instead of LARSVariant"""
		if self.exclude:
			for r in self.exclude:
				if re.search(r, param_name) is not None:
					return False
		return True

	def _get_variable_name(self, param_name):
		"""Get the variable name from the tensor name"""
		m = re.match("^(.*):\\d+$", param_name)
		if m is not None:
			param_name = m.group(1)
		return param_name

	def step_epoch_count(self):
		if self.lp_epoch is not None:
			self.current_epoch += 1
			if self.current_epoch >= self.lp_epoch:
				self.current_ord = self.next_ord