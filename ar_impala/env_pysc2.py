'''Initialize pysc2 environment with flags.
'''

from absl import flags

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features

FLAGS = flags.FLAGS


def create_pysc2_env():
  return sc2_env.SC2Env(
    map_name=FLAGS.map_name,
    visualize=False,
    players=[
      sc2_env.Agent(sc2_env.Race.protoss, 'Agent'),
      sc2_env.Agent(sc2_env.Race.protoss, 'Opponent'),
    ],
    agent_interface_format=features.AgentInterfaceFormat(
      feature_dimensions=features.Dimensions(screen=FLAGS.resolution,
                                             minimap=FLAGS.resolution),
      action_space=actions.ActionSpace.RAW,
      use_raw_units=True,
      raw_resolution=FLAGS.resolution,
      # Make sure build coordinates the same as map coordinates.
      crop_to_playable_area=True,
      raw_crop_to_playable_area=True,
    ),
    game_steps_per_episode=FLAGS.game_steps_per_episode,
    step_mul=FLAGS.step_mul,
    disable_fog=True,
  )
