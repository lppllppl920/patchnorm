from .patchnorm import PatchNormConv2D
from .patchnorm import EfficientPatchNormConv2D
from tensorflow import keras


def tuplify(val, length):
  if type(val) in [tuple, list] and len(val) == length:
    return tuple(val)
  else:
    return tuple(val for _ in range(length))


def listify(val, length):
  if type(val) in [tuple, list] and len(val) == length:
    return list(val)
  else:
    return [val for _ in range(length)]


def listwrap(val):
  if type(val) in [tuple, list]:
    return list(val)
  else:
    return [val]


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
