"""Run A3C training.

Run default training:
python run.py

Run a CartPole test:
python run.py \
--cartpole=True --max_episodes=1000 --update_global_iter=100 --num_workers=0

"""

import logging
import os

from absl import app
from absl import flags

from manager import Manager

# See info level logging.
logging.getLogger().setLevel(logging.INFO)
# Each Python process remains single-threaded, faster.
os.environ['OMP_NUM_THREADS'] = '1'

# Training flags.
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for Adam.')
flags.DEFINE_integer('max_episodes', 2,
                     'Number of episodes to run the training for.')
flags.DEFINE_integer('update_global_iter', 1,
                     'How often to update the global network.')
flags.DEFINE_float(
  'gamma', .9,
  'Discount factor. The higher, the more to value future rewards.')
flags.DEFINE_float('epsilon', 1e-8,
                   'Term added to the denominator to improve stability.')
flags.DEFINE_integer(
  'num_workers', 1,
  'Number of parallel workers, the lower of this and the system.')

# Game flags.
flags.DEFINE_boolean('cartpole', False,
                     'Use CartPole environment to test algorithm.')
flags.DEFINE_string('map', 'Simple64', 'Name of a map to use.')
flags.DEFINE_integer('step_mul', 384, 'Game steps per observation.')
flags.DEFINE_string('replay_dir', '', 'Directory to save replays to.')


def main(_):
  """Run A3C."""
  manager = Manager()
  manager.train()


if __name__ == '__main__':
  app.run(main)
