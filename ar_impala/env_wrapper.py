'''DeepMind environment wrapper, to conform to Acme.
This environment also defines the relevant Acme state and action space.
'''

import dm_env
import numpy as np
from absl import flags

from acme import specs
from acme import types
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

FLAGS = flags.FLAGS


class DmEnvStarCraftWrapper(dm_env.Environment):
  def __init__(self, environment):
    self._environment = environment
    self._reset_next_step = True
    self.actions = self.actions_builder()
    self.last_obs = None
    # Note dtype for all specs.
    # The dimensions of action_spec are for action_type only.
    self._action_spec = specs.DiscreteArray(num_values=len(self.actions),
                                            dtype=np.int32)
    # 3 + FLAGS.resolution*FLAGS.resolution
    self._observation_spec = specs.Array(
      (FLAGS.state_size + FLAGS.resolution * FLAGS.resolution, ),
      dtype=np.float32)
    self._reward_spec = specs.Array((), dtype=np.float32)

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    agent, opp = self._environment.reset()
    self.last_obs = agent
    state = self.obs_to_state(agent)
    # Only return the agent observation.
    return dm_env.restart(state)

  def step(self, action_dict: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()
    pysc2_action = self.action_handler(action_dict, self.last_obs)
    agent, opp = self._environment.step(
      [pysc2_action, actions.RAW_FUNCTIONS.no_op()])
    self.last_obs = agent
    state = self.obs_to_state(agent)
    done = agent.last()
    self._reset_next_step = done
    # Return reward at the end of an episode.
    reward = np.asarray(0, dtype=np.float32)
    if done:
      reward = self.state_to_reward(state)
    if done:
      # Save a replay.
      self._environment.save_replay('', FLAGS.replay_prefix)
      return dm_env.termination(reward, state)
    return dm_env.transition(reward, state)

  def state_to_reward(self, state):
    """State to reward."""
    # Number of pylons.
    # reward = state[1]
    # Number of forges.
    reward = state[2]
    return np.asarray(reward, dtype=np.float32)

  def obs_to_state(self, obs):
    """Get a simplified state from an internal environment observation."""
    state = []
    # Number of probes.
    state.append(len(self.get_agent_units_by_type(units.Protoss.Probe, obs)))
    # Number of pylons.
    state.append(len(self.get_agent_units_by_type(units.Protoss.Pylon, obs)))
    # # Number of forges.
    # state.append(len(self.get_agent_units_by_type(units.Protoss.Forge, obs)))
    '''
    SCREEN_FEATURES = ScreenFeatures(
      height_map=(256, FeatureType.SCALAR, colors.height_map, False),
      visibility_map=(4, FeatureType.CATEGORICAL,
                      colors.VISIBILITY_PALETTE, False),
      creep=(2, FeatureType.CATEGORICAL, colors.CREEP_PALETTE, False),
      power=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
      player_id=(17, FeatureType.CATEGORICAL,
                colors.PLAYER_ABSOLUTE_PALETTE, False),
      player_relative=(5, FeatureType.CATEGORICAL,
                      colors.PLAYER_RELATIVE_PALETTE, False),
      unit_type=(max(static_data.UNIT_TYPES) + 1, FeatureType.CATEGORICAL,
                colors.unit_type, False),
      selected=(2, FeatureType.CATEGORICAL, colors.SELECTED_PALETTE, False),
      unit_hit_points=(1600, FeatureType.SCALAR, colors.hot, True),
      unit_hit_points_ratio=(256, FeatureType.SCALAR, colors.hot, False),
      unit_energy=(1000, FeatureType.SCALAR, colors.hot, True),
      unit_energy_ratio=(256, FeatureType.SCALAR, colors.hot, False),
      unit_shields=(1000, FeatureType.SCALAR, colors.hot, True),
      unit_shields_ratio=(256, FeatureType.SCALAR, colors.hot, False),
      unit_density=(16, FeatureType.SCALAR, colors.hot, True),
      unit_density_aa=(256, FeatureType.SCALAR, colors.hot, False),
      effects=(16, FeatureType.CATEGORICAL, colors.effects, False),
      hallucinations=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
      cloaked=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
      blip=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
      buffs=(max(static_data.BUFFS) + 1, FeatureType.CATEGORICAL,
            colors.buffs, False),
      buff_duration=(256, FeatureType.SCALAR, colors.hot, False),
      active=(2, FeatureType.CATEGORICAL, colors.POWER_PALETTE, False),
      build_progress=(256, FeatureType.SCALAR, colors.hot, False),
      pathable=(2, FeatureType.CATEGORICAL, colors.winter, False),
      buildable=(2, FeatureType.CATEGORICAL, colors.winter, False),
      placeholder=(2, FeatureType.CATEGORICAL, colors.winter, False),
    )
    '''
    map = obs.observation.feature_screen.buildable
    map = np.reshape(map, (FLAGS.resolution * FLAGS.resolution))

    state = np.concatenate((state, map))
    return np.asarray(state, dtype=np.float32)

  def action_spec(self) -> specs.DiscreteArray:
    return self._action_spec

  def observation_spec(self) -> specs.BoundedArray:
    return self._observation_spec

  def reward_spec(self) -> specs.Array:
    return self._reward_spec

  def close(self):
    self._environment.close()

  def train_handler(self, split_action, obs):
    """Handle training actions."""
    if split_action[1] in ['Probe']:
      building = self.get_agent_units_by_type(units.Protoss.Nexus, obs)[0]
    return getattr(actions.RAW_FUNCTIONS, split_action[0] + '_' +
                   split_action[1] + '_quick')('now', building.tag)

  def build_handler(self, split_action, point, obs):
    """Handle building actions."""
    probe = self.get_agent_units_by_type(units.Protoss.Probe, obs)[0]
    return getattr(actions.RAW_FUNCTIONS,
                   split_action[0] + '_' + split_action[1] + '_pt')('now',
                                                                    probe.tag,
                                                                    point)

  def action_handler(self, action_dict, obs):
    """Delegate actions to subhandlers."""
    # Decompose the action.
    action_type = action_dict['action_type']
    location = action_dict['location']

    action_name = self.actions[action_type]
    point = (location // FLAGS.resolution, location % FLAGS.resolution)
    split_action = action_name.split('_')

    print(action_name, point)

    if split_action[0] == 'Train':
      return self.train_handler(split_action, obs)
    if split_action[0] == 'Build':
      return self.build_handler(split_action, point, obs)
    return actions.RAW_FUNCTIONS.no_op()

  @staticmethod
  def actions_builder():
    """Create action space."""
    # action_build = ['Build_Pylon']
    action_build = ['NO_OP', 'Train_Probe', 'Build_Pylon']
    # action_build = ['NO_OP', 'Train_Probe', 'Build_Pylon', 'Build_Forge']
    return action_build

  @staticmethod
  def get_agent_units_by_type(unit_type, obs):
    """Get agent units of the same type."""
    return [
      unit for unit in obs.observation.raw_units if unit.unit_type == unit_type
      and unit.alliance == features.PlayerRelative.SELF
    ]
