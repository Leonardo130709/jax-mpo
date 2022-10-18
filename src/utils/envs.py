import dm_env


class DMC(dm_env.Environment):
    def __init__(self,
                 task: str,
                 seed: int,
                 size: tuple[int],
                 action_repeat: int,
                 ):
        from dm_control import suite
        domain, task = task.split('_', 1)
        if domain == 'ball':
            domain = 'ball_in_cup'
            task = 'catch'
        self._env = suite.load(domain, task,
                               task_kwargs={'random': seed})

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()

    def __getattr__(self, item):
        return getattr(self._env, item)
