'''Based off
https://github.com/deepmind/acme/blob/master/acme/agents/tf/impala/learning.py

Sums loss for both action_type logits and location logits.
'''

import time
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional

import acme
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl
from absl import flags

from acme import specs
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers

FLAGS = flags.FLAGS

tfd = tfp.distributions


class ArImpalaLearner(acme.Learner, tf2_savers.TFSaveable):
  """Learner for an importanced-weighted advantage actor-critic."""
  def __init__(
    self,
    environment_spec: specs.EnvironmentSpec,
    network: snt.RNNCore,
    dataset: tf.data.Dataset,
    learning_rate: float,
    discount: float = 0.99,
    entropy_cost: float = 0.,
    baseline_cost: float = 1.,
    max_abs_reward: Optional[float] = None,
    max_gradient_norm: Optional[float] = None,
    counter: counting.Counter = None,
    logger: loggers.Logger = None,
  ):

    # Internalise, optimizer, and dataset.
    self._env_spec = environment_spec
    self._optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
    self._network = network
    self._variables = network.variables
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

    # Hyperparameters.
    self._discount = discount
    self._entropy_cost = entropy_cost
    self._baseline_cost = baseline_cost

    # Set up reward/gradient clipping.
    if max_abs_reward is None:
      max_abs_reward = np.inf
    if max_gradient_norm is None:
      max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
    self._max_abs_reward = tf.convert_to_tensor(max_abs_reward)
    self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    self._snapshotter = tf2_savers.Snapshotter(
      objects_to_save={'network': network}, time_delta_minutes=60.)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @property
  def state(self) -> Mapping[str, tf2_savers.Checkpointable]:
    """Returns the stateful objects for checkpointing."""
    return {
      'network': self._network,
      'optimizer': self._optimizer,
    }

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Does an SGD step on a batch of sequences."""
    def calculate_losses(behaviour_logits, labels, label_logits, rewards,
                         discounts, values):
      # Compute importance sampling weights: current policy / behavior policy.
      pi_behaviour = tfd.Categorical(logits=behaviour_logits[:-1])
      pi_target = tfd.Categorical(logits=label_logits[:-1])
      log_rhos = pi_target.log_prob(labels) - pi_behaviour.log_prob(labels)

      # Critic loss.
      vtrace_returns = trfl.vtrace_from_importance_weights(
        log_rhos=tf.cast(log_rhos, tf.float32),
        discounts=tf.cast(self._discount * discounts, tf.float32),
        rewards=tf.cast(rewards, tf.float32),
        values=tf.cast(values[:-1], tf.float32),
        bootstrap_value=values[-1],
      )
      critic_loss = tf.square(vtrace_returns.vs - values[:-1])

      policy_gradient_loss = trfl.policy_gradient(
        policies=pi_target,
        actions=actions,
        action_values=vtrace_returns.pg_advantages,
      )

      # Entropy regulariser.
      entropy_loss = trfl.policy_entropy_loss(pi_target).loss

      loss = tf.reduce_mean(policy_gradient_loss + self._baseline_cost *
                            (critic_loss) + self._entropy_cost *
                            (entropy_loss))

      loss_dict = {
        'loss': loss,
        'critic_loss': critic_loss,
        'policy_gradient_loss': policy_gradient_loss,
        'entropy_loss': entropy_loss,
      }
      return loss_dict

    # Retrieve a batch of data from replay.
    inputs = next(self._iterator)
    data = tf2_utils.batch_to_sequence(inputs.data)
    observations, actions, rewards, discounts, extra = (data.observation,
                                                        data.action,
                                                        data.reward,
                                                        data.discount,
                                                        data.extras)
    core_state = tree.map_structure(lambda s: s[0], extra['core_state'])

    actions = actions[:-1]  # [T-1]
    rewards = rewards[:-1]  # [T-1]
    discounts = discounts[:-1]  # [T-1]

    locations = extra['locations']
    locations = locations[:-1]

    with tf.GradientTape() as tape:
      # Unroll current policy over observations.
      (unused_action_types, unused_locations, action_logits, location_logits,
       values), _ = snt.static_unroll(self._network, observations, core_state)

      # Optionally clip rewards.
      rewards = tf.clip_by_value(rewards,
                                 tf.cast(-self._max_abs_reward, rewards.dtype),
                                 tf.cast(self._max_abs_reward, rewards.dtype))

      action_loss_dict = calculate_losses(extra['action_logits'], actions,
                                          action_logits, rewards, discounts,
                                          values)
      location_loss_dict = calculate_losses(extra['location_logits'],
                                            locations, location_logits,
                                            rewards, discounts, values)

      # TODO: Better way of recording different weights for losses.
      all_loss_dict = action_loss_dict
      for loss_dict in [location_loss_dict]:
        all_loss_dict['loss'] += loss_dict['loss'] * .5
        all_loss_dict['critic_loss'] += loss_dict['loss'] * .5
        all_loss_dict[
          'policy_gradient_loss'] += loss_dict['policy_gradient_loss'] * .5
        all_loss_dict['entropy_loss'] += loss_dict['entropy_loss'] * .5

    # Compute gradients and optionally apply clipping.
    gradients = tape.gradient(all_loss_dict['loss'],
                              self._network.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    metrics = {
      'loss':
      all_loss_dict['loss'],
      'critic_loss':
      tf.reduce_mean(all_loss_dict['critic_loss']),
      'policy_gradient_loss':
      tf.reduce_mean(all_loss_dict['policy_gradient_loss']),
      'entropy_loss':
      tf.reduce_mean(all_loss_dict['entropy_loss']),
    }

    return metrics

  def step(self):
    """Does a step of SGD and logs the results."""

    # Do a batch of SGD.
    results = self._step()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    results.update(counts)

    # Snapshot and attempt to write logs.
    self._snapshotter.save()
    self._logger.write(results)

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return [tf2_utils.to_numpy(self._variables)]
