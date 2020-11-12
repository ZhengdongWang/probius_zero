'''Based off
https://github.com/deepmind/acme/blob/master/acme/agents/tf/impala/agent.py

Basically exactly the same, except initializing custom actor and learner.
'''

from typing import Optional

import dm_env
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
from absl import flags

from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as reverb_adders
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme.agents.tf import impala

from actor import ArImpalaActor
from learner import ArImpalaLearner

FLAGS = flags.FLAGS


class ArImpala(impala.agent.IMPALA):
  """IMPALA Agent."""
  def __init__(
    self,
    environment_spec: specs.EnvironmentSpec,
    network: snt.RNNCore,
    sequence_length: int,
    sequence_period: int,
    counter: counting.Counter = None,
    logger: loggers.Logger = None,
    discount: float = 0.99,
    max_queue_size: int = 100000,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    entropy_cost: float = 0.01,
    baseline_cost: float = 0.5,
    max_abs_reward: Optional[float] = None,
    max_gradient_norm: Optional[float] = None,
  ):
    # Refers to action_type only.
    num_actions = environment_spec.actions.num_values

    self._logger = logger or loggers.TerminalLogger('agent')
    queue = reverb.Table.queue(name=reverb_adders.DEFAULT_PRIORITY_TABLE,
                               max_size=max_queue_size)
    self._server = reverb.Server([queue], port=None)
    self._can_sample = lambda: queue.can_sample(batch_size)
    address = f'localhost:{self._server.port}'

    # Component to add things into replay.
    adder = reverb_adders.SequenceAdder(
      client=reverb.Client(address),
      period=sequence_period,
      sequence_length=sequence_length,
    )

    # The dataset object to learn from.
    extra_spec = {
      'core_state':
      network.initial_state(1),
      'action_logits':
      tf.ones(shape=(1, num_actions), dtype=tf.float32),
      'locations':
      tf.ones(shape=(1), dtype=tf.int32),
      'location_logits':
      tf.ones(shape=(1, FLAGS.resolution * FLAGS.resolution),
              dtype=tf.float32),
    }
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
    dataset = datasets.make_reverb_dataset(client=reverb.TFClient(address),
                                           environment_spec=environment_spec,
                                           batch_size=batch_size,
                                           extra_spec=extra_spec,
                                           sequence_length=sequence_length)

    tf2_utils.create_variables(network, [environment_spec.observations])

    self._actor = ArImpalaActor(network, adder)
    self._learner = ArImpalaLearner(
      environment_spec=environment_spec,
      network=network,
      dataset=dataset,
      counter=counter,
      logger=logger,
      discount=discount,
      learning_rate=learning_rate,
      entropy_cost=entropy_cost,
      baseline_cost=baseline_cost,
      max_gradient_norm=max_gradient_norm,
      max_abs_reward=max_abs_reward,
    )

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(
    self,
    action: types.NestedArray,
    next_timestep: dm_env.TimeStep,
  ):
    self._actor.observe(action, next_timestep)

  def update(self):
    # Run a number of learner steps (usually gradient steps).
    while self._can_sample():
      self._learner.step()

  def select_action(self, observation: np.ndarray):
    return self._actor.select_action(observation)
