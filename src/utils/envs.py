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
