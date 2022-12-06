import os

import dm_env
from dm_env import specs
import jax
import numpy as np
import gym


class DMC(dm_env.Environment):
    def __init__(self,
                 task: str,
                 seed: int | np.random.RandomState,
                 size: tuple[int, int],
                 camera: str | int = 0,
                 pn_number: int = 1000,
                 ):
        os.environ["MUJOCO_GL"] = "egl"
        from rltools.dmc_wrappers.utils.point_cloud_generator import \
            PointCloudGenerator

        domain, task = task.split("_", 1)
        self._is_manip = False

        if domain == "manip":
            from dm_control import manipulation
            self._is_manip = True
            self._env = manipulation.load(task, seed)
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
            def replace_shape(spec):
                shape = spec.shape
                return spec.replace(shape=shape[1:])

            obs_spec = jax.tree_util.tree_map(
                replace_shape,
                obs_spec.copy(),
                is_leaf=lambda sp: isinstance(sp, specs.Array)
            )
            obs_spec['jaco_arm/joints_pos'] =\
                obs_spec['jaco_arm/joints_pos'].replace(shape=(12,))
            # obs_spec[GOAL_KEY] = obs_spec["target_position"]
        else:
            img_shape = self.size + (3,)
            # obs_spec.update(
            #     depth_map=Array(self.size + (1,), np.float32),
            #     point_cloud=Array((self.pn_number, 3), np.float32),
            #     image=specs.BoundedArray(
            #         img_shape, np.uint8,
            #         np.full(img_shape, 0), np.full(img_shape, 255)
            #     )
            # )
        return obs_spec

    def _update_obs(self, obs):
        if self._is_manip:
            for k, v in obs.items():
                if k == 'jaco_arm/joints_pos':
                    obs[k] = v.flatten()
                else:
                    obs[k] = np.squeeze(v, 0)
            # obs[GOAL_KEY] = obs["target_position"]
        else:
            physics = self._env.physics
            # depth_map = physics.render(*self.size, camera_id=self.camera, depth=True)
            # depth_map = depth_map[..., None]
            # obs.update(
            #     # point_cloud=self._pcg(physics).astype(np.float32),
            #     image=physics.render(*self.size, camera_id=self.camera),
            #     # depth_map=depth_map
            # )
        return obs


class UR5(dm_env.Environment):
    def __init__(self,
                 address: tuple[str, str],
                 size: tuple[int, int],
                 pn_number: int = 1000
                 ):
        from ur_env.remote import RemoteEnvClient
        self._env = RemoteEnvClient(address)
        self.size = size
        self.pn_number = pn_number

    def action_spec(self):
        act_spec = self._env.action_space
        return specs.BoundedArray(
            act_spec.shape, act_spec.dtype, act_spec.low, act_spec.high)

    def observation_spec(self):
        obs_spec = self._env.observation_space.copy()
        obs_spec = jax.tree_map(
            lambda sp: specs.Array(sp.shape, sp.dtype), obs_spec,
            is_leaf=lambda x: isinstance(x, gym.spaces.Box)
        )
        im_spec = obs_spec['image']
        shape = (im_spec.shape[:-1], im_spec.shape[-1] + 3)
        del obs_spec['image']
        del obs_spec['point_cloud']
        obs_spec['pcd_with_image'] = specs.Array(shape, np.float32)

    def reset(self):
        obs = self._env.reset().observation
        obs = self._get_observation(obs)
        return dm_env.TimeStep(
            observation=obs, reward=0, discount=1,
            step_type=dm_env.StepType.FIRST
        )

    def step(self, action):
        obs, reward, done, _ = self._env.step(action)
        return dm_env.TimeStep(
            observation=self._get_observation(obs),
            reward=reward,
            discount=float(not done),
            step_type=dm_env.StepType.LAST if done else dm_env.StepType.MID
        )

    def _get_observation(self, obs: dict):
        obs = obs.copy()
        image = obs['image'] / 255. - 0.5
        pcd = np.tanh(obs['point_cloud'])
        new_pcd = np.concatenate([pcd, image], -1)
        channels = image.shape[-1] + pcd.shape[-1]
        new_pcd = new_pcd.reshape((-1, channels))
        stride = new_pcd.shape[0] // self.pn_number
        new_pcd = new_pcd[::stride]
        new_pcd = new_pcd[:self.pn_number]
        del obs['image']
        del obs['point_cloud']
        obs['pcd_with_image'] = new_pcd
        return obs
