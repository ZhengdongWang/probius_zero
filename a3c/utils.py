"""Helpers.

"""

import logging

import matplotlib.pyplot as plt
import tensorflow as tf


class Memory:
  """Training buffers."""
  def __init__(self):
    self.clear()

  def store(self, state, action, reward):
    """Add to buffers."""
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    """Clear buffers."""
    self.states = []
    self.actions = []
    self.rewards = []


def push_pull(global_model, local_model, optimizer, done, next_state, memory):  # pylint: disable=too-many-arguments
  """Communicate between global and local models."""

  # Calculate gradient with respect to local model.
  with tf.GradientTape() as tape:
    total_loss = local_model.compute_loss(done, next_state, memory)
  gradients = tape.gradient(total_loss, local_model.trainable_weights)

  # Push local gradients to global model.
  optimizer.apply_gradients(zip(gradients, global_model.trainable_weights))
  # Update local model with new weights.
  local_model.set_weights(global_model.get_weights())


def record(
  worker_i,
  global_episode,
  global_reward,
  episode_reward,
  result_queue,
):
  """Recording helper."""
  if global_reward == 0:
    global_reward = episode_reward
  else:
    global_reward = global_reward * 0.99 + episode_reward * 0.01
  logging.info(
    'Worker: %d | '
    'Episode: %d | '
    'Moving Average Reward: %.2f | '
    'Episode Reward: %.2f',
    worker_i,
    global_episode,
    global_reward,
    episode_reward,
  )
  result_queue.put(global_reward)
  return global_reward


def save_final(rewards):
  """Save training metrics."""
  plt.plot(rewards)
  plt.ylabel('Moving average ep reward')
  plt.xlabel('Step')
  plt.savefig('Moving Average.png')
  plt.show()
