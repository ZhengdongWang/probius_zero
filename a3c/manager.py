"""A3C manager.

"""

import multiprocessing
from queue import Queue

import gym
import numpy.random as npr
import tensorflow as tf
from absl import flags

from actor_critic_model import ActorCriticModel
from sc2_env_wrapper import SC2EnvWrapper
from utils import save_final
from worker import Worker

FLAGS = flags.FLAGS


class Manager():
  """A3C manager."""
  def __init__(self):
    if FLAGS.cartpole:
      self.env = gym.make('CartPole-v0')
      state_dim = self.env.observation_space.shape[0]
      action_dim = self.env.action_space.n
    else:
      self.env = SC2EnvWrapper()
      state_dim = self.env.state_dim
      action_dim = self.env.action_dim

    self.optimizer = tf.optimizers.Adam(FLAGS.learning_rate)
    self.global_model = ActorCriticModel(state_dim, action_dim)
    # Initialize global model.
    self.global_model(
      tf.convert_to_tensor(npr.random((1, state_dim)), dtype=tf.float32))  # pylint: disable=no-member

  def train(self):
    """Train A3C."""
    result_queue = Queue()

    if FLAGS.num_workers == 0:
      num_workers = multiprocessing.cpu_count()
    else:
      num_workers = min(FLAGS.num_workers, multiprocessing.cpu_count())

    workers = [
      Worker(worker_i, self.global_model, self.optimizer, result_queue)
      for worker_i in range(num_workers)
    ]

    [worker.start() for worker in workers]  # pylint: disable=expression-not-assigned

    rewards = []
    while True:
      reward = result_queue.get()
      if reward is not None:
        rewards.append(reward)
      else:
        break

    [worker.join() for worker in workers]  # pylint: disable=expression-not-assigned

    save_final(rewards)

  @staticmethod
  def play():
    """Play against agent."""
    return
