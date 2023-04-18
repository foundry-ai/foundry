from stanza.envs import Environment
from stanza.policies import Policy, PolicyTransform, PolicyOutput
from stanza.util.dataclasses import dataclass, field
from stanza.dataset import Dataset
from functools import partial

import stanza
import itertools
import jax.numpy as jnp
import jax.random
from jax.random import PRNGKey
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
    agent: AgentState
    block: BlockState

@dataclass(jax=True)
class PushTPositionState:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_rot: jnp.array

@dataclass(jax=True)
class PushTEnv(Environment):
    sim_hz: float = 100
    goal_pose : BlockState = field(
        default_factory=lambda: BlockState(
            jnp.array([256.,256.]),
            jnp.array([0.,0.]),
            jnp.array(jnp.pi/4)
        ))

    def sample_action(self, rng_key):
        pos_agent = jax.random.randint(rng_key, (2,), 50, 450).astype(float)
        return pos_agent

    def sample_state(self, rng_key):
        return self.reset(rng_key)

    def reset(self, rng_key):
        pos_key, block_key, rot_key = jax.random.split(rng_key, 3)
        pos_agent = jax.random.randint(pos_key, (2,), 50, 450).astype(float)
        agent = AgentState(pos_agent, jnp.zeros((2,)))
        pos_block = jax.random.randint(block_key, (2,), 100, 400).astype(float)
        rot_block = jax.random.uniform(rot_key, minval=-jnp.pi, maxval=jnp.pi)
        block = BlockState(pos_block, jnp.zeros((2,)), rot_block)
        return PushTState(agent, block)
    
    def observe(self, state):
        return state

    def render(self, state, action=None, width=500, height=500):
        img = jax.pure_callback(
            partial(PushTEnv._callback_render, width=width, height=height),
            jax.ShapeDtypeStruct((3, width, height), jnp.uint8),
            self, state, action
        )
        return jnp.transpose(img, (1,2,0))

    def _callback_render(self, state, action, width, height):
        space, _, _ = self._setup_space(state)
        # add the target block to the space to render
        self._add_tee(space,
            (self.goal_pose.pos[0],self.goal_pose.pos[1]),
            (0,0), self.goal_pose.rot,
            color=(0,1,0), z=-1)
        # add a crosshairs at the control input position
        if action is not None and self.pos_control:
            ltp = action
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
        if action is not None:
            agent.velocity += action * dt
        space.step(dt)
        # extract the end state from the space
        agent_state = AgentState(jnp.array([agent.position.x, agent.position.y]),
                                 jnp.array([agent.velocity.x, agent.velocity.y]))
        block_state = BlockState(jnp.array([block.position.x, block.position.y]),
                                 jnp.array([block.velocity.x, block.velocity.y]),
                                 jnp.array(block.angle))
        return PushTState(agent_state, block_state)

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

def builder(name):
    return PushTEnv()

# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
class PositionalPushT(PolicyTransform):
    def __call__(self, policy, policy_init_state):
        return PushTPositionalPolicy(policy), policy_init_state
    
@dataclass(jax=True)
class PushTPositionalPolicy(Policy):
    policy: Policy

    def __call__(self, state, policy_state=None, rng_key=None):
        # extract just the position parts
        # to feed into the policy
        pass

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
    agent_pos = state[:,:2]
    agent_vel = jnp.zeros_like(agent_pos)
    block_pos = state[:,2:4]
    block_vel = jnp.zeros_like(block_pos)
    block_rot = state[:,4]
    states = PushTState(
        AgentState(agent_pos, agent_vel),
        BlockState(block_pos, block_vel, block_rot))
    return Dataset.from_pytree((states, actions))

# ----- pretrained network -------

def remap_diff_step_encoder():
    yield 'diffusion_step_encoder.1', "net/diff_embed_linear_0"
    yield 'diffusion_step_encoder.3', "net/diff_embed_linear_1"

def remap_resblock(in_prefix, out_prefix):
    for l in [0,1]:
        yield f"{in_prefix}.blocks.{l}.block.0", f"{out_prefix}/block{l}/conv"
        yield f"{in_prefix}.blocks.{l}.block.1", f"{out_prefix}/block{l}/group_norm"
    yield f'{in_prefix}.residual_conv', f'{out_prefix}/residual_conv'
    yield f'{in_prefix}.cond_encoder.1', f'{out_prefix}/cond_encoder'
def remap_downsample(in_prefix, out_prefix):
    yield f'{in_prefix}.conv', f'{out_prefix}/conv'

def remap_upsample(in_prefix, out_prefix):
    yield f'{in_prefix}.conv', f'{out_prefix}/conv_transpose'


MOD_NAME_MAP = dict(itertools.chain(
    remap_diff_step_encoder(),
    remap_resblock('mid_modules.0', 'net/mid0'),
    remap_resblock('mid_modules.1', 'net/mid1'),

    remap_resblock('down_modules.0.0', 'net/down0_res0'),
    remap_resblock('down_modules.0.1', 'net/down0_res1'),
    remap_downsample('down_modules.0.2', 'net/down0_downsample'),
    remap_resblock('down_modules.1.0', 'net/down1_res0'),
    remap_resblock('down_modules.1.1', 'net/down1_res1'),
    remap_downsample('down_modules.1.2', 'net/down1_downsample'),
    remap_resblock('down_modules.2.0', 'net/down2_res0'),
    remap_resblock('down_modules.2.1', 'net/down2_res1'),

    remap_resblock('up_modules.0.0', 'net/up0_res0'),
    remap_resblock('up_modules.0.1', 'net/up0_res1'),
    remap_upsample('up_modules.0.2', 'net/up0_upsample'),
    remap_resblock('up_modules.1.0', 'net/up1_res0'),
    remap_resblock('up_modules.1.1', 'net/up1_res1'),
    remap_upsample('up_modules.1.2', 'net/up1_upsample'),
    [('final_conv.0.block.0', 'net/final_conv_block/conv'),
     ('final_conv.0.block.1', 'net/final_conv_block/group_norm'),
     ('final_conv.1', 'net/final_conv')],
))

def pretrained_net():
    import gdown
    import os
    import torch
    cache = os.path.join(os.getcwd(), '.cache')
    os.makedirs(cache, exist_ok=True)
    model_path = os.path.join(cache, 'pusht_model.ckpt')
    if not os.path.exists(model_path):
        id = "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t"
        gdown.download(id=id, output=model_path, quiet=False)
    
    tm = torch.load(model_path, map_location=torch.device('cpu'))
    mapped_params = {}
    for (k,v) in tm.items():
        v = jnp.array(v.numpy()).T
        if k.endswith('.weight'):
            root = k[:-len('.weight')]
            # root = MOD_NAME_MAP[root]
            ext = 'scale' if 'group_norm' in root else 'w'
        elif k.endswith('.bias'):
            root = k[:-len('.bias')]
            ext = 'offset' if 'group_norm' in root else 'b'
        mapped_root = MOD_NAME_MAP[root]
        if 'group_norm' in mapped_root:
            ext = 'offset' if ext == 'b' else 'scale'
        # print(f'{k} -> {mapped_root} {ext} {v.shape}')
        # Map the root name
        if 'transpose' in mapped_root and ext == 'w':
            # for some reason the conv transposed
            # needs the kernel to be flipped but the
            # regular conv does not?
            v = jnp.flip(v, 0)
        d = mapped_params.setdefault(mapped_root,{})
        d[ext] = v

    from stanza.model.unet1d import ConditionalUnet1D
    import haiku as hk
    def model(sample, timestep, cond):
        model = ConditionalUnet1D(name='net')
        r = model(sample, timestep, cond)
        return r
    net = hk.transform(model)
    return net, mapped_params

def pretrained_policy():
    from stanza.model.diffusion import DDPMSchedule
    net, params = pretrained_net()
    model = stanza.Partial(net.apply, params, None)
    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=1)
    sample_action_trajectory = jnp.zeros((16, 2))
    def policy(obs, policy_state, rng_key):
        schedule.sample(rng_key, model)

# ----- Rendering utilities ------
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