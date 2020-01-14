import tensorflow as tf
from tensorflow import keras

from . import utils


class PatchNormConv2D(keras.layers.Layer):
  """Notes:

  Whereas batch norm computes the mean and variance over the entire dataset, we
  compute these values over the inputs for each image patch, for a single image input.

  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               axis=3,
               **kwargs):
    """Patch norm + convolution.

    :param filters: 
    :param kernel_size: 
    :param strides: 
    :param padding: 
    :param activation: 
    :param use_bias: 
    :param kernel_initializer: 
    :param bias_initializer: 
    :param kernel_regularizer: 
    :param bias_regularizer: 
    :param activity_regularizer: 
    :param kernel_constraint: 
    :param bias_constraint: 
    :param axis: 
    :returns: 
    :rtype: 

    """
    super().__init__(**kwargs)

    self.filters = filters
    self.kernel_size = utils.tuplify(kernel_size, 2)
    self.strides = utils.tuplify(strides, 2)
    self.padding = padding
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.axis = axis

    assert self.padding == 'same' or self.kernel_size == (1, 1), 'todo: padding != same'
    assert self.axis == 3, 'todo: axis != 3'

  def build(self, input_shape):
    self.beta = self.add_weight('beta',
                                dtype=tf.float32,
                                trainable=True,
                                initializer=tf.constant_initializer(0))
    self.gamma = self.add_weight('gamma',
                                 dtype=tf.float32,
                                 trainable=True,
                                 initializer=tf.constant_initializer(1))
    self.epsilon = tf.constant(1e-5, tf.float32)

    self.conv = keras.layers.Conv2D(
      filters=self.filters,
      kernel_size=self.kernel_size,
      strides=self.kernel_size,
      padding='valid',
      activation=self.activation,
      use_bias=self.use_bias,
      kernel_initializer=self.kernel_initializer,
      bias_initializer=self.bias_initializer,
      kernel_regularizer=self.kernel_regularizer,
      bias_regularizer=self.bias_regularizer,
      activity_regularizer=self.activity_regularizer,
      kernel_constraint=self.kernel_constraint,
      bias_constraint=self.bias_constraint)

  def call(self, x):
    patches = tf.image.extract_patches(
      images=x,
      sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
      strides=[1, self.strides[0], self.strides[1], 1],
      rates=[1, 1, 1, 1],
      padding=self.padding.upper())
    # approximately [N, H, W, h * w * C] (if stride is 1 and padding is 'same')

    mus = tf.math.reduce_mean(patches, axis=3, keepdims=True)
    sigs = tf.math.reduce_std(patches, axis=3, keepdims=True)

    centered = (patches - mus) / tf.sqrt(tf.square(sigs) + self.epsilon)
    shifted = self.gamma * centered + self.beta
    shifted = tf.reshape(shifted, [-1, shifted.shape[1], shifted.shape[2],
                                   self.kernel_size[0], self.kernel_size[1], x.shape[3]])

    shifted_t = tf.transpose(shifted, perm=[0, 1, 3, 2, 4, 5])
    shifted_flat = tf.reshape(shifted_t, [-1, shifted.shape[1] * self.kernel_size[0],
                                          shifted.shape[2] * self.kernel_size[1], x.shape[3]])
    return self.conv(shifted_flat)

  def get_config(self):
    config = super().get_config()
    config.update({'filters': self.filters,
                   'kernel_size': self.kernel_size,
                   'strides': self.strides,
                   'padding': self.padding,
                   'activation': self.activation,
                   'use_bias': self.use_bias,
                   'kernel_initializer': self.kernel_initializer,
                   'bias_initializer': self.bias_initializer,
                   'kernel_regularizer': self.kernel_regularizer,
                   'bias_regularizer': self.bias_regularizer,
                   'activity_regularizer': self.activity_regularizer,
                   'kernel_constraint': self.kernel_constraint,
                   'bias_constraint': self.bias_constraint,
                   'axis': self.axis})
    return config

  def set_conv_weights(self, weights):
    """Set the weights for the internal conv layer.

    :param weights:
    :returns:
    :rtype:

    """
    self.conv.set_weights(weights)


class EfficientPatchNormConv2D(PatchNormConv2D):
  def build(self, input_shape):
      # TODO: Both pairs of beta and gamma setting work fine. (seems like filters one converge a bit faster?)
    self.beta = self.add_weight(
      'beta',
      shape=(1, 1, 1, self.filters),
      dtype=tf.float32,
      trainable=True,
      initializer=tf.constant_initializer(0))
    self.gamma = self.add_weight(
      'gamma',
      shape=(1, 1, 1, self.filters),
      dtype=tf.float32,
      trainable=True,
      initializer=tf.constant_initializer(1))
    self.epsilon = tf.constant(1e-5, tf.float32)

    self.conv = keras.layers.Conv2D(
      filters=self.filters,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      activation=None,
      use_bias=False,
      kernel_initializer=self.kernel_initializer,
      bias_initializer=self.bias_initializer,
      kernel_regularizer=self.kernel_regularizer,
      bias_regularizer=self.bias_regularizer,
      activity_regularizer=self.activity_regularizer,
      kernel_constraint=self.kernel_constraint,
      bias_constraint=self.bias_constraint)

    if self.use_bias:
      raise NotImplementedError('use_bias must be False')
      self.bias = self.add_weight(
        'bias',
        shape=(self.filters,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=tf.float32)

    if self.activation is not None:
      self.act = keras.layers.Activation(self.activation)

    factor = 1.0 / (input_shape[3] * self.kernel_size[0] * self.kernel_size[1])
    self.window = factor * tf.ones((self.kernel_size[0], self.kernel_size[1], input_shape[3], 1), dtype=tf.float32)
    self.var_bias_correction = (input_shape[3] * self.kernel_size[0] * self.kernel_size[1]) / \
                               (input_shape[3] * self.kernel_size[0] * self.kernel_size[1] - 1.0)

  def call(self, x):
    """Implement the patch norm operation using Xingtong's more efficient method.

    """
    # (N, H', W', 1)
    mean_map = tf.nn.conv2d(x, self.window, strides=self.strides, padding=self.padding.upper())
    mean_sq_map = tf.nn.conv2d(tf.math.square(x), self.window, strides=self.strides, padding=self.padding.upper())
    std_map = tf.math.sqrt(self.var_bias_correction * (mean_sq_map - tf.math.square(mean_map)) + self.epsilon)

    # (N, H', W', filters)
    x = self.conv(x)

    # (1, 1, 1, filters)
    kernel_summation = tf.reshape(tf.reduce_sum(self.conv.kernel, axis=(0, 1, 2)), (1, 1, 1, -1))  # .read_value()

    x = (x - (mean_map * kernel_summation)) / std_map
    x = self.gamma * x + self.beta * kernel_summation

    if self.use_bias:
      x += self.bias

    if self.activation is not None:
      x = self.act(x)

    return x


def load_traditional_weights(model, weights_path):
  """Load the weights from a traditional model file into a model using PatchNorm.

  Assumes that the conv layers in the weights file have a corresponding layer
  in the model with the same name. Ignore BatchNorm layers if they are preceded
  by a conv layer with a matching PatchNormConv2D or EfficientPatchNormConv2D
  layer.

  :param model: model to load the weights into
  :param weights_path: path to the .h5 file containing the weights

  """
  source_model = keras.models.load_model(weights_path)
  loaded_patchnorm_weights = False
  
  for i, source_layer in enumerate(source_model.layers):
    if source_layer.count_params() == 0:
      continue

    if isinstance(source_layer, keras.layers.BatchNormalization) and loaded_patchnorm_weights:
      continue

    target_layer = model.get_layer(source_layer.name)
    if issubclass(type(target_layer), PatchNormConv2D):
      target_layer.set_conv_weights(source_layer.get_weights())
      loaded_patchnorm_weights = True
    else:
      target_layer.set_weights(source_layer.get_weights())
      loaded_patchnorm_weights = False

  del source_model
  return model
