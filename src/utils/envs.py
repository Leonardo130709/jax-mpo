import dm_env
import PIL


class DMC(dm_env.Environment):
    def __init__(self,
                 task: str,
                 seed: int,
                 size: tuple[int],
                 action_repeat: int,
                 ):
        domain, task = task.split("_", 1)
        if domain == "manip":
            from dm_control import manipulation
            self._env = manipulation.load(task, seed)
        else:
            from dm_control import suite
            self._env = suite.load(
                domain, task,
                task_kwargs={"random": seed},
                environment_kwargs={"flat_observation": True}
            )

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()


class UR5(dm_env.Environment):
    def __init__(self, size):
        from ur_env.remote import RemoteEnvClient