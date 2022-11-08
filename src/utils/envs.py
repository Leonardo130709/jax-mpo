import os

import dm_env
from dm_env import specs
import jax
import numpy as np
import gym

from src import GOAL_KEY


class DMC(dm_env.Environment):
    def __init__(self,
                 task: str,
                 seed: int | np.random.RandomState,
                 size: tuple[int, int],
                 camera: str | int = 0,
                 pn_number: int = 1000,
                 ):
        os.environ["MUJOCO_GL"] = "egl"
        from src.utils.dm_control_pcd_generator import PointCloudGenerator

        domain, task = task.split("_", 1)
        self._is_manip = False

        if domain == "manip":
            from dm_control import manipulation
            self._env = manipulation.load(task + "_features", seed)
            self._is_manip = True
        else:
            from dm_control import suite
            self._env = suite.load(
                domain, task,
                task_kwargs={"random": seed},
                environment_kwargs={"flat_observation": True}
            )
        self.camera = camera
        self.size = size
        self.pn_number = pn_number

        self._pcg = PointCloudGenerator(
            pn_number,
            [dict(camera_id=camera, width=320, height=240)]
        )

    def reset(self):
        timestep = self._env.reset()
        self._update_obs(timestep.observation)
        return timestep

    def step(self, action):
        timestep = self._env.step(action)
        self._update_obs(timestep.observation)
        return timestep

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        obs_spec = self._env.observation_spec()

        if self._is_manip:

            def _replace_shape(spec):
                shape = spec.shape
                return spec.replace(shape=(np.prod(shape),))

            obs_spec = jax.tree_util.tree_map(
                _replace_shape,
                obs_spec,
                is_leaf=lambda x: isinstance(x, specs.Array)
            )
            # obs_spec[GOAL_KEY] = obs_spec["target_position"]
        obs_spec.update(
        #     depth_map=Array(self.size + (1,), np.float32),
        #     point_cloud=Array((self.pn_number, 3), np.float32),
            image=specs.Array(self.size + (3,), np.uint8)
        )
        return obs_spec

    def _update_obs(self, obs):
        if self._is_manip:
            for k, v in obs.items():
                obs[k] = v.flatten()
            # obs[GOAL_KEY] = obs["target_position"]
        physics = self._env.physics
        # depth_map = physics.render(*self.size, camera_id=self.camera, depth=True)
        # depth_map = depth_map[..., None]
        obs.update(
            # point_cloud=self._pcg(physics).astype(np.float32),
            image=physics.render(*self.size, camera_id=self.camera),
            # depth_map=depth_map
        )
        return obs


class UR5(dm_env.Environment):
    def __init__(self,
                 address: tuple[str, str]
                 ):
        from ur_env.remote import RemoteEnvClient
        self._env = RemoteEnvClient(address)

    def action_spec(self):
        act_spec = self._env.action_space
        return specs.BoundedArray(
            act_spec.shape, act_spec.dtype, act_spec.low, act_spec.high)

    def observation_spec(self):
        obs_spec = self._env.observation_space
        return jax.tree_map(
            lambda sp: specs.Array(sp.shape, sp.dtype), obs_spec,
            is_leaf=lambda x: isinstance(x, gym.spaces.Box)
        )

    def reset(self):
        obs = self._env.reset().observation
        return dm_env.TimeStep(
            observation=obs, reward=0, discount=1,
            step_type=dm_env.StepType.FIRST
        )

    def step(self, action):
        obs, reward, done, _ = self._env.step(action)
        return dm_env.TimeStep(
            observation=obs,
            reward=reward,
            discount=float(not done),
            step_type=dm_env.StepType.LAST if done else dm_env.StepType.MID
        )
