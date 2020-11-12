'''Run training with DeepMind environment wrapped pysc2 and above network.
'''

from absl import app
from absl import flags

from acme import environment_loop
from acme import specs

from agent import ArImpala
from env_pysc2 import create_pysc2_env
from env_wrapper import DmEnvStarCraftWrapper
from network import ProbiusNetwork

flags.DEFINE_integer('state_size', 2, '')
flags.DEFINE_string('map_name', 'Simple64', '')
flags.DEFINE_integer('resolution', 64, '')
flags.DEFINE_integer('game_steps_per_episode', 4000, '')
flags.DEFINE_integer('step_mul', 500, '')
flags.DEFINE_string('replay_prefix', 'simple_long', '')
FLAGS = flags.FLAGS


def main(_):
  environment = create_pysc2_env()
  wrapped_environment = DmEnvStarCraftWrapper(environment)
  environment_spec = specs.make_environment_spec(wrapped_environment)
  print('actions:\n', environment_spec.actions, '\n')
  print('observations:\n', environment_spec.observations, '\n')
  print('rewards:\n', environment_spec.rewards, '\n')

  network = ProbiusNetwork(environment_spec.actions.num_values)
  agent = ArImpala(
    environment_spec=environment_spec,
    network=network,
    sequence_length=10,
    sequence_period=10,
    batch_size=2,
  )

  loop = environment_loop.EnvironmentLoop(wrapped_environment, agent)
  loop.run(num_episodes=10)


if __name__ == '__main__':
  app.run(main)
