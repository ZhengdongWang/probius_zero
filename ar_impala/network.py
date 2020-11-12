'''A very simple AlphaStar architecture.

The inputs are,
[Number of probes, number of pylons]
Concatenated with the map data.

The scalar inputs are embedded by a MLP and the map a CNN.
The embeddings and state are inputs to an LSTM core.
The LSTM core output is the input to ActionTypeHead, then LocationValueHead.
'''

import sonnet as snt
import tensorflow as tf
from absl import flags

from acme.tf.networks import base

from heads import ActionTypeHead, LocationValueHead

FLAGS = flags.FLAGS


class AtariTorso(base.Module):
  """Simple convolutional stack commonly used for Atari."""
  def __init__(self):
    super().__init__(name='atari_torso')
    self._network = snt.Sequential([
      snt.Conv2D(32, [8, 8], [4, 4]),
      tf.nn.relu,
      snt.Conv2D(64, [4, 4], [2, 2]),
      tf.nn.relu,
      snt.Conv2D(64, [3, 3], [1, 1]),
      tf.nn.relu,
      snt.Flatten(),
    ])

  def __call__(self, inputs) -> tf.Tensor:
    return self._network(inputs)


class ProbiusNetwork(snt.RNNCore):
  """A recurrent network for use with IMPALA."""
  def __init__(self, num_actions: int):
    super().__init__(name='probius_network')
    self._scalar_embed = snt.Sequential(
      [snt.Flatten(), snt.nets.MLP([50, 50])])
    self._map_embed = AtariTorso()
    self._core = snt.LSTM(256)
    self._heads = snt.Sequential([
      ActionTypeHead(num_actions),
      LocationValueHead(),
    ])
    self._num_actions = num_actions

  def __call__(self, inputs, state):

    scalars, map = tf.split(
      inputs, [FLAGS.state_size, FLAGS.resolution * FLAGS.resolution], 1)
    # B, H, W, C
    map = tf.reshape(map, (-1, FLAGS.resolution, FLAGS.resolution, 1))

    scalar_embeddings = self._scalar_embed(scalars)
    map_embeddings = self._map_embed(map)
    embeddings = scalar_embeddings

    embeddings, new_state = self._core(embeddings, state)

    action_types, locations, action_logits, location_logits, value = self._heads(
      embeddings)
    return (action_types, locations, action_logits, location_logits,
            value), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> snt.LSTMState:
    return self._core.initial_state(batch_size)
