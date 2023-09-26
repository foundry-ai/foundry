from stanza.envs import Environment
from stanza.dataclasses import dataclass, field, replace
from stanza.util.attrdict import AttrMap
from typing import Callable, List
from jax.random import PRNGKey

import jax.numpy as jnp
import jax
import stanza.graphics.canvas as canvas
import pymunk

@dataclass(jax=True)
class BodyState:
    position: jnp.array
    velocity: jnp.array
    angle: jnp.array
    angular_velocity: jnp.array

class PyMunkState(AttrMap):
    pass

@dataclass(jax=True, kw_only=True)
class PyMunkWrapper(Environment):
    sim_hz: float
    width: float
    height: float

    def _make_state(self, space):
        state = PyMunkState()
        for body in space.bodies:
            if hasattr(body, 'name'):
                body_state = BodyState(
                    jnp.array(body.position),
                    jnp.array(body.velocity),
                    jnp.array(body.angle),
                    jnp.array(body.angular_velocity)
                )
                state = state.set(body.name, body_state)
        return state

    @jax.jit
    def reset(self, rng_key):
        with jax.ensure_compile_time_eval():
            space = self._build_space(PRNGKey(0))
            _state = self._make_state(space)
        return jax.pure_callback(type(self)._reset_callback, 
                                 _state, self, rng_key)

    def _reset_callback(self, rng_key):
        space = self._build_space(rng_key)
        return self._make_state(space)

    @jax.jit
    def step(self, state, action, rng_key):
        with jax.ensure_compile_time_eval():
            space = self._build_space(PRNGKey(0))
            _state = self._make_state(space)
        return jax.pure_callback(type(self)._step_callback,
                                 _state, self, state, action, rng_key) 

    def _step_callback(self, state, action, rng_key):
        space = self._build_state(state)
        self._space_action(space, action, rng_key)
        space.step(1/self.sim_hz)
        return self._make_state(space)
    
    @jax.jit
    def render(self, state, width=256, height=256):
        img = jnp.ones((width, height, 3))
        with jax.ensure_compile_time_eval():
            space = self._build_space(PRNGKey(0))

        shapes = []
        for shape in space.shapes:
            if isinstance(shape, pymunk.Circle):
                sdf = canvas.circle(jnp.array(shape.offset),
                                    radius=shape.radius)
            elif isinstance(shape, pymunk.Poly):
                points = [
                    jnp.array([p.x, p.y]) \
                        for p in shape.get_vertices()
                ]
                points = jnp.array(points)
                sdf = canvas.polygon(points)
            elif isinstance(shape, pymunk.Segment):
                a = jnp.array((shape.a.x, shape.a.y))
                b = jnp.array((shape.b.x, shape.b.y))
                sdf = canvas.segment(a, b, thickness=shape.radius)
            color = jnp.array(shape.color) if hasattr(shape, 'color') else jnp.array([0.3, 0.3, 0.3])
            render = canvas.fill(sdf, color)
            # do the body transform
            if shape.body is not None:
                body = shape.body
                if hasattr(body, "name") and body.name is not None:
                    bs = state[body.name]
                    translation = bs.position
                    rotation = bs.angle
                else:
                    translation = jnp.array(body.position)
                    rotation = jnp.array(body.angle)

                render = canvas.transform(render,
                    translation=translation,
                    rotation=rotation
                )
            shapes.append(render)
        render = canvas.stack(*shapes)
        # flip y axis
        # transform
        render = canvas.transform(render,
            scale=jnp.array([width, height]) / jnp.array([self.width, self.height]),
        )
        render = canvas.transform(render,
            scale=jnp.array([1, -1]),
            translation=jnp.array([0, -height])
        )
        img = canvas.paint(img, render)
        return img
    
    def _build_state(self, state):
        with jax.ensure_compile_time_eval():
            space = self._build_space(PRNGKey(0))
        for b in space.bodies:
            if hasattr(b, 'name'):
                bs = state[b.name]
                b.position = tuple(bs.position)
                b.velocity = tuple(bs.velocity)
                b.angle = bs.angle
                b.angular_velocity = bs.angular_velocity
        return space

    # Perform an action on the system
    def _space_action(self, space, action, rng_key):
        pass

    def _build_space(self, rng_key):
        pass

@dataclass(jax=True, kw_only=True)
class PyMunkEnv(PyMunkWrapper):
    space_builder : Callable
    space_action : Callable = None

    def _space_action(self, space, action, rng_key):
        if self.space_action is not None:
            self.space_action(space, action, rng_key)

    def _build_space(self, rng_key):
        return self.space_builder(rng_key)