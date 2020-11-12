'''Based off Acme policy-value head.
https://github.com/deepmind/acme/blob/master/acme/tf/networks/policy_value.py

Returns action_type, location, logits, and value.
The first head, ActionTypeHead, passes the sampled action_type to the second and
last head, LocationValueHead. The sampled action_type is used as input to the
location network.
'''

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags

from acme.tf.networks import base

FLAGS = flags.FLAGS
tfd = tfp.distributions


class ActionTypeHead(snt.Module):
  def __init__(self, num_actions: int):
    super().__init__(name='action_type_network')
    self._num_actions = num_actions
    self._action_policy_layer = snt.Sequential([
      snt.nets.MLP([50, 50, self._num_actions]),
      tf.nn.relu,
    ])
    self._core_policy_layer = snt.Linear(50)

  def __call__(self, inputs: tf.Tensor):
    action_logits = self._action_policy_layer(inputs)
    core_logits = self._core_policy_layer(inputs)

    action_type = tfd.Categorical(action_logits).sample()
    action_one_hot = tf.one_hot(action_type, self._num_actions)

    embedding = tf.concat([action_one_hot, core_logits], axis=1)
    return action_logits, action_type, embedding


class LocationValueHead(snt.Module):
  def __init__(self):
    super().__init__(name='location_value_network')
    self._value_layer = snt.Linear(1)
    self._deconv_net = snt.Sequential([
      snt.Linear(50),
      snt.Linear(FLAGS.resolution // 4 * FLAGS.resolution // 4),
      # [B, H, W, C, D], batch size omitted.
      snt.Reshape((FLAGS.resolution // 4, FLAGS.resolution // 4, 1)),
      snt.Conv2DTranspose(1, 1,
                          (FLAGS.resolution // 4, FLAGS.resolution // 4)),
      tf.keras.layers.UpSampling2D(2),
      snt.Conv2DTranspose(1, 1,
                          (FLAGS.resolution // 2, FLAGS.resolution // 2)),
      tf.keras.layers.UpSampling2D(2),
      snt.Conv2DTranspose(1, 1, (FLAGS.resolution, FLAGS.resolution)),
    ])

  def __call__(self, inputs):
    action_logits, action_type, embedding = inputs
    location_logits = self._deconv_net(embedding)

    location_logits = tf.reshape(location_logits,
                                 (-1, FLAGS.resolution * FLAGS.resolution))
    location = tfd.Categorical(location_logits).sample()

    value = tf.squeeze(self._value_layer(embedding), axis=-1)
    return action_type, location, action_logits, location_logits, value
