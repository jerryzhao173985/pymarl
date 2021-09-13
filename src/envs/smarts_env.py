
import logging

import gym
import numpy as np
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np

import smarts
from envision.client import Client as Envision
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.visdom_client import VisdomClient

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, AgentPolicy
from smarts.core.utils.episodes import episodes

from smarts.core.agent import AgentSpec, AgentPolicy
# from envs.smarts_observations import lane_ttc_observation_adapter
from typing import Callable
from dataclasses import dataclass

import numpy as np
import gym

from smarts.core.utils.math import vec_2d, vec_to_radians, squared_dist
from smarts.core.coordinates import Heading


from glob import glob

@dataclass
class Adapter:
    space: gym.Space
    transform: Callable


def scan_for_vehicle(
    target_prefix,
    angle_a,
    angle_b,
    activation_dist_squared,
    self_vehicle_state,
    other_vehicle_state,
):
    if target_prefix and not other_vehicle_state.id.startswith(target_prefix):
        return False

    min_angle, max_angle = min(angle_a, angle_b), max(angle_a, angle_b)
    sqd = squared_dist(self_vehicle_state.position, other_vehicle_state.position)
    # check for activation distance
    if sqd < activation_dist_squared:
        direction = np.sum(
            [other_vehicle_state.position, -self_vehicle_state.position], axis=0
        )
        angle = Heading(vec_to_radians(direction[:2]))
        rel_angle = angle.relative_to(self_vehicle_state.heading)
        return min_angle <= rel_angle <= max_angle
    return False


def scan_for_vehicles(
    target_prefix,
    angle_a,
    angle_b,
    activation_dist_squared,
    self_vehicle_state,
    other_vehicle_states,
    short_circuit: bool = False,
):
    if target_prefix:
        other_vehicle_states = filter(
            lambda v: v.id.startswith(target_prefix), other_vehicle_states
        )

    min_angle, max_angle = min(angle_a, angle_b), max(angle_a, angle_b)
    vehicles = []

    for vehicle_state in other_vehicle_states:
        sqd = squared_dist(self_vehicle_state.position, vehicle_state.position)
        # check for activation distance
        if sqd < activation_dist_squared:
            direction = np.sum(
                [vehicle_state.position, -self_vehicle_state.position], axis=0
            )
            angle = Heading(vec_to_radians(direction[:2]))
            rel_angle = angle.relative_to(self_vehicle_state.heading)
            if min_angle <= rel_angle <= max_angle:
                vehicles.append(vehicle_state)
                if short_circuit:
                    break
    return vehicles


_LANE_TTC_OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        # "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        # "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)


def _lane_ttc_observation_adapter(env_observation):
    ego = env_observation.ego_vehicle_state
    waypoint_paths = env_observation.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    ego_ttc, ego_lane_dist = _ego_ttc_lane_dist(env_observation, closest_wp.lane_index)

    return {
        "distance_from_center": np.array([norm_dist_from_center]),
        "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
        "speed": np.array([ego.speed]),
        "steering": np.array([ego.steering]),
        # "ego_ttc": np.array(ego_ttc),
        # "ego_lane_dist": np.array(ego_lane_dist),   ##FIXME: Delete last two observations (6 values), only perserve 4 sensor values
    }


lane_ttc_observation_adapter = Adapter(
    space=_LANE_TTC_OBSERVATION_SPACE, transform=_lane_ttc_observation_adapter
)


def _ego_ttc_lane_dist(env_observation, ego_lane_index):
    ttc_by_p, lane_dist_by_p = _ttc_by_path(env_observation)

    return _ego_ttc_calc(ego_lane_index, ttc_by_p, lane_dist_by_p)


def _ttc_by_path(env_observation):
    ego = env_observation.ego_vehicle_state
    waypoint_paths = env_observation.waypoint_paths
    neighborhood_vehicle_states = env_observation.neighborhood_vehicle_states

    # first sum up the distance between waypoints along a path
    # ie. [(wp1, path1, 0),
    #      (wp2, path1, 0 + dist(wp1, wp2)),
    #      (wp3, path1, 0 + dist(wp1, wp2) + dist(wp2, wp3))]

    wps_with_lane_dist = []
    for path_idx, path in enumerate(waypoint_paths):
        lane_dist = 0.0
        for w1, w2 in zip(path, path[1:]):
            wps_with_lane_dist.append((w1, path_idx, lane_dist))
            lane_dist += np.linalg.norm(w2.pos - w1.pos)
        wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

    # next we compute the TTC along each of the paths
    ttc_by_path_index = [1000] * len(waypoint_paths)
    lane_dist_by_path_index = [1] * len(waypoint_paths)
    if neighborhood_vehicle_states is not None:
        for v in neighborhood_vehicle_states:
            # find all waypoints that are on the same lane as this vehicle
            wps_on_lane = [
                (wp, path_idx, dist)
                for wp, path_idx, dist in wps_with_lane_dist
                if wp.lane_id == v.lane_id
            ]

            if not wps_on_lane:
                # this vehicle is not on a nearby lane
                continue

            # find the closest waypoint on this lane to this vehicle
            nearest_wp, path_idx, lane_dist = min(
                wps_on_lane, key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position))
            )

            if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
                # this vehicle is not close enough to the path, this can happen
                # if the vehicle is behind the ego, or ahead past the end of
                # the waypoints
                continue

            relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
            if abs(relative_speed_m_per_s) < 1e-5:
                relative_speed_m_per_s = 1e-5

            ttc = lane_dist / relative_speed_m_per_s
            ttc /= 10
            if ttc <= 0:
                # discard collisions that would have happened in the past
                continue

            lane_dist /= 100
            lane_dist_by_path_index[path_idx] = min(
                lane_dist_by_path_index[path_idx], lane_dist
            )
            ttc_by_path_index[path_idx] = min(ttc_by_path_index[path_idx], ttc)

    return ttc_by_path_index, lane_dist_by_path_index


def _ego_ttc_calc(ego_lane_index, ttc_by_path, lane_dist_by_path):
    ego_ttc = [0] * 3
    ego_lane_dist = [0] * 3

    ego_ttc[1] = ttc_by_path[ego_lane_index]
    ego_lane_dist[1] = lane_dist_by_path[ego_lane_index]

    max_lane_index = len(ttc_by_path) - 1
    min_lane_index = 0
    if ego_lane_index + 1 > max_lane_index:
        ego_ttc[2] = 0
        ego_lane_dist[2] = 0
    else:
        ego_ttc[2] = ttc_by_path[ego_lane_index + 1]
        ego_lane_dist[2] = lane_dist_by_path[ego_lane_index + 1]
    if ego_lane_index - 1 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
    else:
        ego_ttc[0] = ttc_by_path[ego_lane_index - 1]
        ego_lane_dist[0] = lane_dist_by_path[ego_lane_index - 1]
    return ego_ttc, ego_lane_dist

       

def observation_adapter(env_observation):
    obs =  lane_ttc_observation_adapter.transform(env_observation)
    obs_flatten = np.concatenate(list(obs.values()), axis=0)
    # print("-----------------------------------------")
    # print("-----------------------------------------")
    # print("-----------------------------------------")
    # print("-----------------------------------------")
    # print(len(obs_flatten), obs_flatten)
    # print("-----------------------------------------")
    # print("-----------------------------------------")
    # print("-----------------------------------------")
    # print("-----------------------------------------")
    # print("-----------------------------------------")
    return obs_flatten

def reward_adapter(env_obs, env_reward):
    return env_reward

def action_adapter(policy_action):
    
    # print("policy_action:: ", policy_action)
    
    if isinstance(policy_action, (list, tuple, np.ndarray)):
        action = np.argmax(policy_action)
    else:
        action = policy_action
    action_dict = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
    return action_dict[action]

class Policy(AgentPolicy):
    def act(self, obs):
        return 0

def get_agent_spec(i):
    pass


import gym
from gym import ObservationWrapper, spaces
from gym.wrappers import TimeLimit as GymTimeLimit
from gym.spaces import flatdim



class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        
        # self.observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(10,))] * self.env.n_agents
        # self.action_space = [gym.spaces.Discrete(4)] * self.env.n_agents
        
        # if max_episode_steps is None and self.env.agent_specs is not None:
        #     max_episode_steps = env.agent_specs.get(env.agent_ids[0]).interface.max_episode_steps
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        
        
        self.n_agents = 2
        self.agent_ids = ["Agent %i" % i for i in range(self.n_agents)]
        self.observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(4,))] * self.n_agents
        self.action_space = [gym.spaces.Discrete(4)] * self.n_agents
        self.longest_action_space = max(self.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self.observation_space, key=lambda x: x.shape
        )
        

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        print("STEP: ", self._elapsed_steps)
        
        # if self._elapsed_steps >= self._max_episode_steps:
        #     info["TimeLimit.truncated"] = not all(done)
        #     done = len(observation) * [True]
        
        
        # observation = [
        #     np.pad(
        #         o,
        #         (0, self.longest_observation_space.shape[0] - len(o)),
        #         "constant",
        #         constant_values=0,
        #     )
        #     for o in observation
        # ]
        obs_n = []
        # covert dict observations to a list of np.arrays
        for agent_id in self.agent_ids:
            obs_n.append(observation.get(agent_id, np.zeros(4)))
            # this is same as: obs_n.append(self.current_observations.get(agent_id))
            # This helps when one agent i.e. 'Agent 0 ' is already finished  
        
        r_n = []
        d_n = []
        for agent_id in self.agent_ids:
            r_n.append(reward.get(agent_id, 0.))
            d_n.append(done.get(agent_id, True))
            
        if self._elapsed_steps >= self._max_episode_steps:
            d_n = self.n_agents * [True]
        
        
        print("all---true dones:",d_n , r_n, obs_n)
        
        return obs_n, r_n, d_n, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        
        self.n_agents = 2
        self.agent_ids = ["Agent %i" % i for i in range(self.n_agents)]
        self.observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(4,))] * self.n_agents
        self.action_space = [gym.spaces.Discrete(4)] * self.n_agents
        
        ma_spaces = []

        for sa_obs in self.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        
        # print(",,,,,,,,,,,,,,,", self.observation_space)
        
        # obs_n = []
        # # covert dict observations to a list of np.arrays
        # for agent_id in self.agent_ids:
        #     obs_n.append(observation.get(agent_id, np.zeros(10)))
        
        
        # print("|||||||||||||||", obs_n)
        # print(zip(self.observation_space, obs_n))
        
        
        # return tuple(
        #     [
        #         spaces.flatten(obs_space, obs)
        #         for obs_space, obs in zip(self.observation_space, observation)
        #     ]
        # )
        
        # print("------------------!!!!!", observation)
        return observation




class SMARTSEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        print(kwargs)
        
        # self.shuffle_scenarios = kwargs['shuffle_scenarios']
        self.shuffle_scenarios=True
        # self.envision_record_data_replay_path = kwargs['envision_record_data_replay_path']
        self.envision_record_data_replay_path=None
        
        # self.envision_endpoint = kwargs['envision_endpoint']
        self.envision_endpoint=None
        # self.sim_name = kwargs['sim_name']
        self.sim_name=None
        
        # self.visdom = kwargs['visdom']
        self.visdom=False
        
        # self.timestep_sec = kwargs['timestep_sec']
        self.timestep_sec=0.1
        # self.zoo_addrs = kwargs['zoo_addrs']
        self.zoo_addrs=None
        
        
        
        self.episode_limit = kwargs['episode_limit']
        self.n_agents = kwargs['agent_num']
        self.observation_space = [gym.spaces.Box(low=-1e10, high=1e10, shape=(4,))] * self.n_agents
        self.action_space = [gym.spaces.Discrete(4)] * self.n_agents
        self.agent_ids = ["Agent %i" % i for i in range(self.n_agents)]
        self.n_actions = 4
        
        
        self.scenarios = [
            kwargs['scenarios']
        ]

        self.headless = kwargs['headless']
        # self.headless = False
        
        # num_episodes = 50
        self.seed = kwargs['seed']
        
        
        # self._log = logging.getLogger(self.__class__.__name__)
        smarts.core.seed(self.seed)
        
        
        self.agent_specs = {
            agent_id: AgentSpec(
                interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=self.episode_limit),
                observation_adapter=observation_adapter,
                reward_adapter=reward_adapter,
                action_adapter=action_adapter,
            )
            for agent_id in self.agent_ids
        }
        
        
        # make env
        self._env = gym.make(
                "smarts.env:hiway-v0", # env entry name
                scenarios=self.scenarios, # a list of paths to folders of scenarios
                agent_specs=self.agent_specs, # dictionary of agents to interact with the environment
                headless=self.headless, # headless mode. False to enable Envision visualization of the environment
                visdom=False, # Visdom visualization of observations. False to disable. only supported in HiwayEnv.
                seed=self.seed, # RNG Seed, seeds are set at the start of simulation, and never automatically re-seeded.
            )
        
        # # agent_specs = {
        # #     agent_id: AgentSpec(
        # #         interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        # #         agent_builder=RuleBasedAgent,
        # #     )
        # #     for agent_id in agent_ids
        # # }

        # self.agent_specs = {
        #     agent_id: AgentSpec(
        #         interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=5000),
        #         observation_adapter=observation_adapter,
        #         reward_adapter=reward_adapter,
        #         action_adapter=action_adapter,
        #     )
        #     for agent_id in self.agent_ids
        # }
        
        # self._dones_registered = 0

        # self._scenarios_iterator = Scenario.scenario_variations(
        #     self.scenarios,
        #     list(self.agent_specs.keys()),
        #     self.shuffle_scenarios,
        # )
        
    
        # agent_interfaces = {
        #     agent_id: agent.interface for agent_id, agent in self.agent_specs.items()
        # }
        
        # # print("!!!agent interfaces: ", self.agent_specs.get(self.agent_ids[0]).interface.max_episode_steps)

        # envision_client = None
        # if not self.headless or self.envision_record_data_replay_path:
        #     envision_client = Envision(
        #         endpoint=self.envision_endpoint,
        #         sim_name=self.sim_name,
        #         output_dir=self.envision_record_data_replay_path,
        #         headless=self.headless,
        #     )

        # visdom_client = None
        # if self.visdom:
        #     visdom_client = VisdomClient()
            

        # # self.base_env = gym.make(
        # #     "smarts.env:hiway-v0",
        # #     scenarios=self.scenarios,
        # #     agent_specs=self.agent_specs,
        # #     headless=self.headless,
        # #     seed=self.seed,
        # # )
        # self.base_env = SMARTS(
        #     agent_interfaces=agent_interfaces,
        #     traffic_sim=SumoTrafficSimulation(
        #         headless=True,         #sumo_headless,
        #         time_resolution=self.timestep_sec,
        #         num_external_sumo_clients=0,      #num_external_sumo_clients,
        #         sumo_port=None,        #sumo_port,
        #         auto_start=True,       #sumo_auto_start,
        #         endless_traffic=True,  #endless_traffic,
        #     ),
        #     envision=envision_client,
        #     visdom=visdom_client,
        #     timestep_sec=self.timestep_sec,
        #     zoo_addrs=self.zoo_addrs,
        # )
        
        # self.max_episode_steps = 50
        self._env = TimeLimit(self._env, max_episode_steps=self.episode_limit)
        
        
        self._env = FlattenObservation(self._env)
        
        self.n_agents = self._env.n_agents
        self._obs = None
        
        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)
        
        
        # scenario = next(self._scenarios_iterator)
        # self.current_scenario = scenario 
        
        # env_obs = self.base_env.reset(scenario)
        # self.current_observations = {
        #     agent_id: self.agent_specs[agent_id].observation_adapter(obs)
        #     for agent_id, obs in env_obs.items()
        # }
        # print(self.current_observations)
        print("-----------------finished initialization of SmartsENV-----------------------")
        
        
        
    def step(self, actions):
        """ Returns reward, terminated, info """
        # print(actions)   # tensor([1, 2], device='cuda:0')
        actions = [int(a) for a in actions]  #[1,2] 
        agent_actions = {
            self.agent_ids[agent_id]: action   #self.agent_specs[self.agent_ids[agent_id]].action_adapter(action)
            for agent_id, action in enumerate(actions)  #actions.items()
        }
        print(agent_actions)
          
        self._obs, reward, done, info = self._env.step(agent_actions)
        # self._obs = [
        #     np.pad(
        #         o,
        #         (0, self.longest_observation_space.shape[0] - len(o)),
        #         "constant",
        #         constant_values=0,
        #     )
        #     for o in self._obs
        # ]
        
        # _obs = []
        # for agent_id in self.agent_ids:
        #     _obs.append(self._obs.get(agent_id, np.zeros(10)))
        
        # self._obs = _obs
        # print("xoxo: ", self._obs, reward, done, info)
    
    
        # r_n = []
        # d_n = []
        # for agent_id in self.agent_ids:
        #     r_n.append(reward.get(agent_id, 0.))
        #     d_n.append(done.get(agent_id, True))
        # print(r_n, d_n)
        # print(done)    # {'Agent 1': False, 'Agent 0': False, '__all__': False}
        print(float(sum(reward)), all(done), {})
        return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        # print("XXXXXX obs: ",self._obs)
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        # self._obs = [
        #     np.pad(
        #         o,
        #         (0, self.longest_observation_space.shape[0] - len(o)),
        #         "constant",
        #         constant_values=0,
        #     )
        #     for o in self._obs
        # ]
        _obs = []
        for agent_id in self.agent_ids:
            _obs.append(self._obs.get(agent_id, np.zeros(4)))
        
        self._obs = _obs
        
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}






    # @property
    # def scenario_log(self):
    #     # self.base_env.scenario_log
    #     """Simulation step logs.

    #     Returns:
    #         A dictionary with the following:
    #             timestep_sec:
    #                 The timestep of the simulation.
    #             scenario_map:
    #                 The name of the current scenario.
    #             scenario_routes:
    #                 The routes in the map.
    #             mission_hash:
    #                 The hash identifier for the current scenario.
    #     """

    #     scenario = self.base_env.scenario
    #     return {
    #         "timestep_sec": self.base_env.timestep_sec,
    #         "scenario_map": scenario.name,
    #         "scenario_routes": scenario.route or "",
    #         "mission_hash": str(hash(frozenset(scenario.missions.items()))),
    #     }


    # def reset(self):
    #     """ Returns initial observations and states"""
    #     print("#########################  RESETTING  #############################")
    #     # try:
    #     #     self.current_observations = self.base_env.reset()
    #     # except:
    #     #     self.base_env.close()
    #     #     self.base_env = gym.make(
    #     #         "smarts.env:hiway-v0",
    #     #         scenarios=self.scenarios,
    #     #         agent_specs=self.agent_specs,
    #     #         headless=self.headless,
    #     #         seed=self.seed,
    #     #     )
    #     #     self.current_observations = self.base_env.reset()
        
    #     # return self.get_obs(), self.get_state()
        
        
    #     # scenario = next(self._scenarios_iterator)
    #     self._dones_registered = 0
    #     env_observations = self.base_env.reset(self.current_scenario)

    #     self.current_observations = {
    #         agent_id: self.agent_specs[agent_id].observation_adapter(obs)
    #         for agent_id, obs in env_observations.items()
    #     }

    #     return self.get_obs(), self.get_state()



    # def close(self):
    #     # self.base_env.close()
    #     if self.base_env is not None:
    #         self.base_env.destroy()

    # def step(self, n_actions):
    #     """ Returns reward, terminated, info """
    #     print("------going on a step-------")
    #     # actions = dict(zip(self.agent_ids, action_n))
    #     # self.current_observations, rewards, dones, infos = self.base_env.step(actions)
    #     # r_n = []
    #     # d_n = []
    #     # for agent_id in self.agent_ids:
    #     #     r_n.append(rewards.get(agent_id, 0.))
    #     #     d_n.append(dones.get(agent_id, True))
        
    #     # return np.sum(r_n), d_n, {} 
        
    #     print(n_actions)
    #     actions = n_actions.cpu().numpy()
    #     print(actions)
        
    #     agent_actions = {
    #         self.agent_ids[agent_id]: self.agent_specs[self.agent_ids[agent_id]].action_adapter(action)
    #         for agent_id, action in enumerate(actions)  #actions.items()
    #     }
    #     print(agent_actions)
        
    #     observations, rewards, agent_dones, extras = self.base_env.step(agent_actions)

    #     infos = {
    #         agent_id: {"score": value, "env_obs": observations[agent_id]}
    #         for agent_id, value in extras["scores"].items()
    #     }

    #     for agent_id in observations:
    #         agent_spec = self.agent_specs[agent_id]
    #         observation = observations[agent_id]
    #         reward = rewards[agent_id]
    #         info = infos[agent_id]

    #         rewards[agent_id] = agent_spec.reward_adapter(observation, reward)
    #         observations[agent_id] = agent_spec.observation_adapter(observation)
    #         infos[agent_id] = agent_spec.info_adapter(observation, reward, info)

    #     for done in agent_dones.values():
    #         self._dones_registered += 1 if done else 0

    #     agent_dones["__all__"] = self._dones_registered == len(self.agent_specs)

    #     print(agent_dones)
        
    #     d_n = [ agent_dones[agentid] for agentid in self.agent_ids]
    #     print(d_n)
        
    #     r_n = [rewards[agentid] for agentid in self.agent_ids]
    #     print("rewards: ", r_n)
        
    #     # return observations, rewards, agent_dones, infos
    #     print(observations)
    #     self.current_observations = observations
        
    #     return np.sum(r_n), d_n, {}





    # ## This following should be the same as the original pymarl MultiAgent script warpper
    # def get_obs(self):
    #     """ Returns all agent observations in a list """
    #     obs_n = []
    #     # covert dict observations to a list of np.arrays
    #     for agent_id in self.agent_ids:
    #         obs_n.append(self.current_observations.get(agent_id, np.zeros(10)))
    #         # this is same as: obs_n.append(self.current_observations.get(agent_id))
            
    #     return obs_n

    # def get_obs_agent(self, agent_id):
    #     """ Returns observation for agent_id """
    #     return self.get_obs()[agent_id]

    # def get_obs_size(self):
    #     """ Returns the shape of the observation """
    #     return len(self.get_obs_agent(0))

    # def get_state(self):
    #     return np.asarray(self.get_obs()).flatten()

    # def get_state_size(self):
    #     """ Returns the shape of the state"""
    #     return self.get_obs_size() * self.n_agents

    # def get_avail_actions(self):
    #     avail_actions = []
    #     for agent_id in range(self.n_agents):
    #         avail_agent = self.get_avail_agent_actions(agent_id)
    #         avail_actions.append(avail_agent)
    #     return avail_actions

    # def get_avail_agent_actions(self, agent_id):
    #     """ Returns the available actions for agent_id """
    #     return np.ones(self.n_actions)

    # def get_total_actions(self):
    #     """ Returns the total number of actions an agent could ever take """
    #     return self.n_actions

    # def get_stats(self):
    #     return None

    # def render(self):
    #     raise NotImplementedError

    # def close(self):
    #     pass

    # def seed(self):
    #     raise NotImplementedError
    
    
        

if __name__ == "__main__":
    env = SMARTSEnv()
    base_nev = env.base_env

    for episode in episodes(n=100):
        observations = env.reset()
        episode.record_scenario(env.base_env.scenario_log)
        
        dones = {"__all__": False}
        while not np.all(dones.values()):
            observations, rewards, dones, infos = env.step([0,1])
            episode.record_step(observations, rewards, dones, infos)

