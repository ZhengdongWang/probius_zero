'''Based off
https://github.com/deepmind/acme/blob/master/acme/agents/tf/impala/acting.py

But each action returns a location as well as an action_type.
Location is returned whether or not the action requires one (eg no_op does not).
'''

import dm_env
import sonnet as snt
import tensorflow as tf
from absl import flags

from acme import adders
from acme import core
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

FLAGS = flags.FLAGS


class ArImpalaActor(core.Actor):
  """A recurrent actor."""
  def __init__(
    self,
    network: snt.RNNCore,
    adder: adders.Adder = None,
    variable_client: tf2_variable_utils.VariableClient = None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._network = network

    # TODO(b/152382420): Ideally we would call tf.function(network) instead but
    # this results in an error when using acme RNN snapshots.
    self._policy = tf.function(network.__call__)

    self._state = None
    self._prev_state = None
    self._prev_action_logits = None
    self._prev_location = None
    self._prev_location_logits = None

  def select_action(self, observation: types.NestedArray):
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    if self._state is None:
      self._state = self._network.initial_state(1)

    # Forward.
    (action_type, location, action_logits, location_logits,
     _), new_state = self._policy(batched_obs, self._state)

    self._prev_action_logits = action_logits
    self._prev_location_logits = location_logits

    self._prev_state = self._state
    self._state = new_state

    action_type = tf2_utils.to_numpy_squeeze(action_type)
    location = tf2_utils.to_numpy_squeeze(location)

    # Adds batch dimension.
    self._prev_location = tf.convert_to_tensor([location], dtype=tf.int32)

    return {
      'action_type': action_type,
      'location': location,
    }

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None

  def observe(
    self,
    action_dict,
    next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return

    # Decompose to just the action type.
    action_type = action_dict['action_type']
    location = action_dict['location']

    # Store other action parameters to learn.
    extras = {
      'core_state': self._prev_state,
      'action_logits': self._prev_action_logits,
      'locations': self._prev_location,
      'location_logits': self._prev_location_logits,
    }

    extras = tf2_utils.to_numpy_squeeze(extras)
    self._adder.add(action_type, next_timestep, extras)

  def update(self):
    if self._variable_client:
      self._variable_client.update()
