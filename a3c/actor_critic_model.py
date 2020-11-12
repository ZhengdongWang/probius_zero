"""A3C model.

"""

import logging

import numpy as np
import numpy.random as npr
import tensorflow as tf
from absl import flags
from tensorflow import keras
from tensorflow.keras import layers

FLAGS = flags.FLAGS


class ActorCriticModel(keras.Model):  # pylint: disable=too-many-ancestors
  """A3C model."""
  def __init__(self, state_dim, action_dim):
    super().__init__()

    self.state_dim = state_dim
    self.action_dim = action_dim

    self.policy_0 = layers.Dense(100, activation='relu')
    self.policy_1 = layers.Dense(action_dim)

    self.value_0 = layers.Dense(100, activation='relu')
    self.value_1 = layers.Dense(1)

  def call(self, inputs):  # pylint: disable=arguments-differ
    """Forward pass."""

    inter_logits = self.policy_0(inputs)
    logits = self.policy_1(inter_logits)

    inter_values = self.value_0(inputs)
    values = self.value_1(inter_values)

    return logits, values

  def choose_action(self, state, obs):
    """Choose action."""

    logits, _ = self.call(
      tf.convert_to_tensor(state[None, :], dtype=tf.float32))
    probs = tf.nn.softmax(logits)

    # Exclude bad actions if not CartPole test.
    if not FLAGS.cartpole:
      logging.info(obs.observation.player.minerals)

    action = npr.choice(self.action_dim, p=probs.numpy()[0])

    return action

  def compute_loss(self, done, state, memory):
    """Compute loss."""

    if done:
      # Terminal.
      reward_sum = 0.
    else:
      reward_sum = self.call(
        tf.convert_to_tensor(state[None, :], dtype=tf.float32))[-1].numpy()[0]

    # Reverse reward buffer.
    discounted_rewards = []
    for reward in memory.rewards[::-1]:
      reward_sum = reward + FLAGS.gamma * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    # Advantage.
    logits, values = self.call(
      tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                     dtype=tf.float32) - values

    # Value loss.
    value_loss = advantage**2

    # Policy loss.
    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy,
                                                      logits=logits)

    # Total loss.
    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=memory.actions, logits=logits)
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

    return total_loss
