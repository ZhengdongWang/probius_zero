"""A3C worker.

"""

import logging
import threading

import gym
from absl import flags

from actor_critic_model import ActorCriticModel
from sc2_env_wrapper import SC2EnvWrapper
from utils import Memory
from utils import push_pull
from utils import record

FLAGS = flags.FLAGS


class Worker(threading.Thread):
  """A3C worker."""

  global_episode = 0
  global_reward = 0.

  def __init__(self, worker_i, global_model, optimizer, result_queue):
    super().__init__()

    self.worker_i = worker_i
    self.global_model = global_model
    self.optimizer = optimizer
    self.result_queue = result_queue

    if FLAGS.cartpole:
      self.env = gym.make('CartPole-v0')
      state_dim = self.env.observation_space.shape[0]
      action_dim = self.env.action_space.n
    else:
      self.env = SC2EnvWrapper()
      state_dim = self.env.state_dim
      action_dim = self.env.action_dim

    self.local_model = ActorCriticModel(state_dim, action_dim)

    logging.info('Worker %d initialized.', self.worker_i)

  def run(self):
    """Run worker."""

    total_step = 1
    memory = Memory()

    while Worker.global_episode < FLAGS.max_episodes:

      if FLAGS.cartpole:
        state = self.env.reset()
      else:
        state, obs = self.env.reset()

      memory.clear()
      episode_reward = 0.

      while True:

        if FLAGS.cartpole:
          # Choose the next action.
          action = self.local_model.choose_action(state, None)
          next_state, reward, done, _ = self.env.step(action)
        else:
          action = self.local_model.choose_action(state, obs)
          next_state, reward, done = self.env.step(action, obs)

        episode_reward += reward
        memory.store(state, action, reward)

        if total_step % FLAGS.update_global_iter == 0 or done:
          # Sync.
          push_pull(self.global_model, self.local_model, self.optimizer, done,
                    next_state, memory)
          memory.clear()

          if done:
            Worker.global_episode += 1
            Worker.global_reward = record(
              self.worker_i,
              Worker.global_episode,
              Worker.global_reward,
              episode_reward,
              self.result_queue,
            )
            break

        state = next_state
        total_step += 1

    self.env.close()
    self.result_queue.put(None)
