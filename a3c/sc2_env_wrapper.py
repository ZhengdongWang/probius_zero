"""Environment wrapper.

"""

import logging

import numpy as np
from absl import flags
from pysc2.env import base_env_wrapper
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

FLAGS = flags.FLAGS


class SC2EnvWrapper(base_env_wrapper.BaseEnvWrapper):
  """Environment wrapper."""
  def __init__(self):
    # Do not initialize parent with an environment until training starts.
    super().__init__(None)

    self.state_dim = 5

    self.actions = self.actions_builder()
    self.action_dim = len(self.actions)

  def reset(self):  # pylint: disable=arguments-differ
    """Reset the environment."""

    if self._env is None:
      self.init_env()

    agent_obs, _ = super().reset()

    state = self.obs_to_state(agent_obs)

    return state, agent_obs

  def step(self, action, obs):  # pylint: disable=arguments-differ
    """Step the environment."""

    agent_action = self.action_handler(action, obs)
    agent_obs, _ = super().step([agent_action, actions.FUNCTIONS.no_op()])

    state = self.obs_to_state(agent_obs)

    return state, agent_obs.reward, agent_obs.last()

  def close(self):  # pylint: disable=arguments-differ
    """Close the internal environment if one exists."""
    if self._env is not None:
      super().close()

  def init_env(self):
    """Initialize internal environment."""
    self._env = sc2_env.SC2Env(
      map_name=FLAGS.map,
      step_mul=FLAGS.step_mul,
      disable_fog=True,
      visualize=False,
      players=[
        sc2_env.Agent(sc2_env.Race.protoss, 'Agent'),
        sc2_env.Agent(sc2_env.Race.protoss, 'Opponent')
      ],
      agent_interface_format=features.AgentInterfaceFormat(
        action_space=actions.ActionSpace.RAW,
        use_raw_units=True,
        raw_resolution=64,
      ),
    )

  def train_handler(self, split_action, obs):
    """Handle training actions."""

    if split_action[1] in ['Probe']:
      building = self.get_agent_units_by_type(units.Protoss.Nexus, obs)[0]

    return getattr(actions.RAW_FUNCTIONS, split_action[0] + '_' +
                   split_action[1] + '_quick')('now', building.tag)

  def build_handler(self, split_action, obs):
    """Handle building actions."""

    probe = self.get_agent_units_by_type(units.Protoss.Probe, obs)[0]

    point = (int(split_action[2]), int(split_action[3]))

    return getattr(actions.RAW_FUNCTIONS,
                   split_action[0] + '_' + split_action[1] + '_pt')('now',
                                                                    probe.tag,
                                                                    point)

  def action_handler(self, action, obs):
    """Delegate actions to subhandlers."""

    action_name = self.actions[action]
    split_action = action_name.split('_')
    logging.info(split_action)

    if split_action[0] == 'Train':
      return self.train_handler(split_action, obs)
    if split_action[0] == 'Build':
      return self.build_handler(split_action, obs)
    return actions.RAW_FUNCTIONS.no_op()

  def obs_to_state(self, obs):
    """Get a simplified state from an internal environment observation."""

    # Currently state space has dimension 5.
    state = []

    # Spawn location.
    nexus = self.get_agent_units_by_type(units.Protoss.Nexus, obs)[0]
    if nexus.x < 32:
      state.append(0)
    else:
      state.append(1)

    # Number of probes.
    state.append(len(self.get_agent_units_by_type(units.Protoss.Probe, obs)))
    # Number of pylons.
    state.append(len(self.get_agent_units_by_type(units.Protoss.Pylon, obs)))
    # Number of forges.
    state.append(len(self.get_agent_units_by_type(units.Protoss.Forge, obs)))
    # Number of cannons.
    state.append(
      len(self.get_agent_units_by_type(units.Protoss.PhotonCannon, obs)))

    logging.info(state)

    return np.asarray(state)

  @staticmethod
  def actions_builder():
    """Create action space."""

    action_build = ['NO_OP', 'Train_Probe', 'BACK_TO_MINING']

    # Doing it like this for easy exclude.
    for x_coord in range(0, 64, 4):
      for y_coord in range(0, 64, 4):
        action_build.append('Build_Pylon_%d_%d' % (x_coord, y_coord))
    for x_coord in range(0, 64, 4):
      for y_coord in range(0, 64, 4):
        action_build.append('Build_Forge_%d_%d' % (x_coord, y_coord))
    for x_coord in range(0, 64, 4):
      for y_coord in range(0, 64, 4):
        action_build.append('Build_PhotonCannon_%d_%d' % (x_coord, y_coord))

    logging.info('Actions built.')

    return action_build

  @staticmethod
  def get_agent_units_by_type(unit_type, obs):
    """Get agent units of the same type."""
    return [
      unit for unit in obs.observation.raw_units if unit.unit_type == unit_type
      and unit.alliance == features.PlayerRelative.SELF
    ]
