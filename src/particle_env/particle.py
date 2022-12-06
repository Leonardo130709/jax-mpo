from typing import NamedTuple
from collections import OrderedDict
import os

import numpy as np
from dm_env import specs

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.manipulation.shared import workspaces


_XML_PATH = os.path.join(os.path.dirname(__file__), 'particle.xml')
SCENE_LIM = .5
HEIGHT_OFFSET = .1
TARGET_SITE_SIZE = .02
THRESHOLD = .005
CTRL_LIMIT = .05
_WIDTH, _HEIGHT = IMG_SIZE = (84, 84)

DEFAULT_BBOX = workspaces.BoundingBox(
    lower=np.array([-SCENE_LIM, -SCENE_LIM, HEIGHT_OFFSET]),
    upper=np.array([SCENE_LIM, SCENE_LIM, HEIGHT_OFFSET])
)


class Particle(composer.Entity):
    def _build(self):
        self._mjcf_model = mjcf.from_path(_XML_PATH)
        self.particle = self.mjcf_model.find('body', 'particle')
        self.camera = self.mjcf_model.find('camera', 'fixed')

    def _build_observables(self):
        return ParticleObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_model


class ParticleObservables(composer.Observables):

    @composer.observable
    def pos(self):
        return observable.MJCFFeature('xpos', self._entity.particle)

    @composer.observable
    def image(self):
        return observable.MJCFCamera(
            self._entity.camera, width=_WIDTH, height=_HEIGHT)


class ParticleReach(composer.Task):

    def __init__(self,
                 scene_bbox: workspaces.BoundingBox = DEFAULT_BBOX,
                 ):
        self._root_entity = Particle()
        self._particle = self._root_entity.particle

        self._bbox = scene_bbox
        self._target_site = workspaces.add_target_site(
            self.root_entity.mjcf_model.worldbody,
            radius=TARGET_SITE_SIZE,
            visible=False, rgba="0 0 1 0.3",
            name="target_site"
        )
        self._rng_fn = lambda rng: rng.uniform(
            scene_bbox.lower, scene_bbox.upper)
        self._goal_image = np.zeros(IMG_SIZE + (3,), dtype=np.uint8)
        self._task_observables = OrderedDict()

        def goal_pos(physics):
            return physics.bind(self._target_site).xpos

        self._task_observables["goal_pos"] = observable.Generic(goal_pos)
        self._task_observables["goal_image"] = observable.Generic(
            lambda _: self._goal_image)

        self.root_entity.observables.enable_all()
        for obs in self._task_observables.values():
            obs.enabled = True

    def initialize_episode(self, physics, random_state):
        # Sample new goal.
        target_site = physics.bind(self._target_site)
        target_pos = self._rng_fn(random_state)
        target_site.pos = target_pos

        # Prepare goal image.
        self._set_pos(physics, target_pos)
        physics.forward()
        self._goal_image = self.root_entity.observables.image(
            physics, random_state).copy()

        # Sample initial pos.
        self._set_pos(physics, self._rng_fn(random_state))
        physics.forward()

    def before_step(self, physics, action, random_state):
        del random_state
        pos = self._get_pos(physics)
        pos += np.concatenate([CTRL_LIMIT*action, [0]])
        self._set_pos(physics, pos)

    def get_reward(self, physics):
        pos = self._get_pos(physics)
        target_site = physics.bind(self._target_site)
        dist = np.linalg.norm(target_site.xpos - pos)
        return float(dist < THRESHOLD)

    def should_terminate_episode(self, physics):
        goal_achieved = self.get_reward(physics) == 1.
        return goal_achieved

    def action_spec(self, physics):
        lim = np.float32([1, 1])
        return specs.BoundedArray(
            shape=lim.shape,
            dtype=lim.dtype,
            minimum=-lim,
            maximum=lim
        )

    def _set_pos(self, physics, pos):
        particle = physics.bind(self._particle)
        particle.mocap_pos = np.clip(
            pos,
            a_min=self._bbox.lower,
            a_max=self._bbox.upper
        )

    def _get_pos(self, physics):
        particle = physics.bind(self._particle)
        return particle.mocap_pos

    @property
    def root_entity(self):
        return self._root_entity

    @property
    def task_observables(self):
        return self._task_observables

    def is_ready(self):
        return self._goal_image is not None


class ParticleEnv(composer.Environment):
    def __init__(self,
                 task=ParticleReach(),
                 time_limit=float('inf'),
                 random_state=None,
                 **kwargs
                 ):
        super().__init__(task, time_limit, random_state,
                         strip_singleton_obs_buffer_dim=True, **kwargs)
