import os

import dm_env
from dm_env.specs import Array
import numpy as np
# import PIL


class DMC(dm_env.Environment):
    def __init__(self,
                 task: str,
                 seed: int,
                 size: tuple[int],
                 camera: str | int = 0,
                 pn_number: int = 500,
                 ):
        os.environ["MUJOCO_GL"] = "egl"
        from src.utils.dm_control_pcd_generator import PointCloudGenerator

        domain, task = task.split("_", 1)
        if domain == "manip":
            from dm_control import manipulation
            self._env = manipulation.load(task+"_vision", seed)
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
        # TODO: restore this.
        # obs_spec.update(
        #     depth_map=Array(self.size + (1,), np.float32),
        #     point_cloud=Array((self.pn_number, 3), np.float32),
        #     image=Array(self.size + (3,), np.uint8)
        # )
        return obs_spec

    def _update_obs(self, obs):
        return obs
        physics = self._env.physics
        depth_map = physics.render(*self.size, camera_id=self.camera, depth=True)
        depth_map = depth_map[..., None]
        obs.update(
            point_cloud=self._pcg(physics).astype(np.float32),
            image=physics.render(*self.size, camera_id=self.camera),
            depth_map=depth_map
        )
        return obs





# class UR5(dm_env.Environment):
#     def __init__(self, size):
#         from ur_env.remote import RemoteEnvClient