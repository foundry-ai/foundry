from stanza.envs import Environment
from stanza.util.dataclasses import dataclass, field
from stanza.dataset import Dataset
from functools import partial

import jax.numpy as jnp
import jax.random
import numpy as np

import pymunk

@dataclass(jax=True)
class AgentState:
    pos: jnp.array
    vel: jnp.array

@dataclass(jax=True)
class BlockState:
    pos: jnp.array
    vel: jnp.array
    rot: jnp.array

@dataclass(jax=True)
class PushTState:
    # for rendering purposes
    last_target_pos : jnp.array
    agent: AgentState
    block: BlockState

@dataclass(jax=True)
class PushTAction:
    # the target position
    target_pos: jnp.array

@dataclass(jax=True)
class PushTObservation:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_rot: jnp.array

@dataclass(jax=True)
class PushTEnv(Environment):
    sim_hz: float = 100
    control_hz: float = 5

    # PID controller
    k_p: float = 100
    k_v: float = 20

    goal_pose : BlockState = field(
        default_factory=lambda: BlockState(
            jnp.array([256.,256.]),
            jnp.array([0.,0.]),
            jnp.array(jnp.pi/4)
        ))

    def sample_action(self, rng_key):
        pos_agent = jax.random.randint(rng_key, (2,), 50, 450).astype(float)
        return PushTAction(pos_agent)

    def sample_state(self, rng_key):
        return self.reset(rng_key)

    def reset(self, rng_key):
        pos_key, block_key, rot_key = jax.random.split(rng_key, 3)
        pos_agent = jax.random.randint(pos_key, (2,), 50, 450).astype(float)
        agent = AgentState(pos_agent, jnp.zeros((2,)))
        pos_block = jax.random.randint(block_key, (2,), 100, 400).astype(float)
        rot_block = jax.random.uniform(rot_key, minval=-jnp.pi, maxval=jnp.pi)
        block = BlockState(pos_block, jnp.zeros((2,)), rot_block)
        return PushTState(jnp.zeros((2,)), agent, block)
    
    def observe(self, state):
        return PushTObservation(state.agent.pos,
                                state.block.pos,
                                state.block.rot)

    def render(self, state, width=500, height=500):
        img = jax.pure_callback(
            partial(PushTEnv._callback_render, width=width, height=height),
            jax.ShapeDtypeStruct((3, width, height), jnp.uint8),
            self, state
        )
        return jnp.transpose(img, (1,2,0))

    def _callback_render(self, state, width=512, height=512):
        space, _, _ = self._setup_space(state)
        # add the target block to the space to render
        self._add_tee(space,
            (self.goal_pose.pos[0],self.goal_pose.pos[1]),
            (0,0), self.goal_pose.rot,
            color=(0,1,0), z=-1)
        # add a crosshairs at the control input position
        ltp = state.last_target_pos
        self._add_segment(space, (ltp[0] - 15, ltp[1]), (ltp[0]+15, ltp[1]), 2,
                          color=(1,0,0), z=1)
        self._add_segment(space, (ltp[0], ltp[1]-15), (ltp[0], ltp[1]+15), 2,
                          color=(1,0,0), z=1)
        return render_space(space, 512, 512, width, height)

    def step(self, state, action=None):
        return jax.pure_callback(PushTEnv._callback_step, state, self, state, action)
    
    def _callback_step(self, state, action):
        space, agent, block = self._setup_space(state)
        dt = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            target = action.target_pos
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (pymunk.Vec2d(target[0], target[1])- agent.position) \
                                + self.k_v * (pymunk.Vec2d(0,0) - agent.velocity)
                agent.velocity += acceleration * dt
                # Step physics.
                space.step(dt)
        # extract the end state from the space
        agent_state = AgentState(jnp.array([agent.position.x, agent.position.y]),
                                 jnp.array([agent.velocity.x, agent.velocity.y]))
        block_state = BlockState(jnp.array([block.position.x, block.position.y]),
                                 jnp.array([block.velocity.x, block.velocity.y]),
                                 jnp.array(block.angle))
        action = jnp.zeros((2,)) if action is None else action
        return PushTState(action, agent_state, block_state)

    def _setup_space(self, state):
        space = pymunk.Space()
        space.gravity = 0, 0
        space.damping = 0
        # Add walls.
        self._add_segment(space, (5, 506), (5, 5), 2)
        self._add_segment(space, (5, 5), (506, 5), 2)
        self._add_segment(space, (506, 5), (506, 506), 2)
        self._add_segment(space, (5, 506), (506, 506), 2)

        # Add the circle agent
        agent = self._add_circle(space,
                (state.agent.pos[0],state.agent.pos[1]),
                (state.agent.vel[0],state.agent.vel[1]), 15,
                (65/255, 105/255, 225/255))
        block = self._add_tee(space,
                (state.block.pos[0],state.block.pos[1]),
                (state.block.vel[0],state.block.vel[1]), state.block.rot.item(),
                color=(119/255, 136/255, 153/255))
        # Add collision handeling
        # _ = space.add_collision_handler(0, 0)
        return space, agent, block

    def _add_segment(self, space, a, b, radius, color=None, z=None):
        shape = pymunk.Segment(space.static_body, a, b, radius)
        if color:
            shape.color = color
            shape.z = z
        space.add(shape)
        return shape

    def _add_circle(self, space, position, velocity, radius, color):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.velocity = velocity
        body.friction = 1
        circle = pymunk.Circle(body, radius)
        circle.color = color
        space.add(body, circle)
        return body

    def _add_tee(self, space, position, velocity, angle,
                 scale=30, color=None, mask=pymunk.ShapeFilter.ALL_MASKS(), z=None):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.z = z
        shape2.z = z
        shape1.color = color
        shape2.color = color
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.velocity = velocity
        body.angle = angle
        body.friction = 1
        space.add(body, shape1, shape2)
        body.position = position
        return body

def render_space(space, space_width, space_height, width, height):
    from cairo import ImageSurface, Context, Format
    surface = ImageSurface(Format.ARGB32, width, height)
    ctx = Context(surface)
    ctx.rectangle(0, 0, width, height)
    ctx.set_source_rgb(0.9, 0.9, 0.9)
    ctx.fill()
    ctx.move_to(width/2, height/2)
    ctx.scale(width/space_width, height/space_height)

    # do a transform based on space_width, space_height
    # and width, height
    shapes = list(space.shapes)
    shapes.sort(key=lambda x: x.z if hasattr(x, 'z') and x.z is not None else 0)
    for shape in shapes:
        ctx.save()
        ctx.translate(shape.body.position[0], shape.body.position[1])
        ctx.rotate(shape.body.angle)
        if hasattr(shape, 'color') and shape.color:
            ctx.set_source_rgb(shape.color[0], shape.color[1], shape.color[2])
        else:
            ctx.set_source_rgb(0.3,0.3,0.3)

        if isinstance(shape, pymunk.Circle):
            # draw a circle
            ctx.arc(shape.offset[0], shape.offset[1], shape.radius, 0, 2*np.pi)
            ctx.close_path()
            ctx.fill()
        elif isinstance(shape, pymunk.Poly):
            verts = shape.get_vertices()
            ctx.move_to(verts[0].x, verts[0].y)
            for v in verts[1:]:
                ctx.line_to(v.x, v.y)
            ctx.close_path()
            ctx.fill()
        elif isinstance(shape, pymunk.Segment):
            ctx.move_to(shape.a.x, shape.a.y)
            ctx.line_to(shape.b.x, shape.b.y)
            ctx.set_line_width(shape.radius)
            ctx.stroke()
        else:
            pass
        ctx.restore()
    img = cairo_to_numpy(surface)[:3,:,:]
    # we need to make a copy otherwise it may
    # get overridden the next time we render

    return np.copy(img)

def cairo_to_numpy(surface):
    data = np.ndarray(shape=(surface.get_height(), surface.get_width(), 4),
                    dtype=np.uint8,
                    buffer=surface.get_data())
    data[:,:,[0,1,2,3]] = data[:,:,[2,1,0,3]]
    data = np.transpose(data, (2, 0, 1))
    return data

def expert_dataset():
    import gdown
    import os
    cache = os.path.join(os.getcwd(), '.cache')
    os.makedirs(cache, exist_ok=True)
    dataset_path = os.path.join(cache, 'pusht_data.zarr.zip')
    if not os.path.exists(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)
    import zarr
    with zarr.open(dataset_path, "r") as data:
        # Read in all of the data
        state = jnp.array(data['data/state'])
        actions = jnp.array(data['data/action'])
    # fill in zeros for the missing state
    # data
    last_action = jnp.roll(actions, 1, 0)
    agent_pos = state[:,:2]
    agent_vel = jnp.zeros_like(agent_pos)
    block_pos = state[:,2:4]
    block_vel = jnp.zeros_like(block_pos)
    block_rot = state[:,4]
    states = PushTState(
        last_action,
        AgentState(agent_pos, agent_vel),
        BlockState(block_pos, block_vel, block_rot))
    actions = PushTAction(actions)
    return Dataset.from_pytree((states, actions))

def builder(name):
    return PushTEnv()