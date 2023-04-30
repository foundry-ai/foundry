from stanza.envs import Environment
from stanza.policies import Policy, PolicyOutput, PolicyTransform, \
                            chain_transforms, \
                            SampleRateTransform, ChunkTransform
from stanza.util.dataclasses import dataclass, field, replace
from stanza.util.attrdict import attrs
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
class BodyState:
    position: jnp.array
    velocity: jnp.array
    angle: jnp.array
    angular_velocity: jnp.array

@dataclass(jax=True)
class PushTState:
    # for rendering purposes
    agent: BodyState
    block: BodyState

@dataclass(jax=True)
class PushTEnv(Environment):
    sim_hz: float = 100
    goal_pose : BodyState = field(
        default_factory=lambda: BodyState(
            jnp.array([256.,256.]),
            jnp.array([0.,0.]),
            jnp.array(jnp.pi/4),
            jnp.array(0)
        ))

    def sample_action(self, rng_key):
        pos_agent = jax.random.randint(rng_key, (2,), 50, 450).astype(float)
        return pos_agent

    def sample_state(self, rng_key):
        return self.reset(rng_key)

    def reset(self, rng_key):
        z = jnp.zeros(())
        z2 = jnp.zeros((2,))
        pos_key, block_key, rot_key = jax.random.split(rng_key, 3)
        pos_agent = jax.random.randint(pos_key, (2,), 50, 450).astype(float)
        agent = BodyState(pos_agent, z2, z, z)
        pos_block = jax.random.randint(block_key, (2,), 100, 400).astype(float)
        rot_block = jax.random.uniform(rot_key, minval=-jnp.pi, maxval=jnp.pi)
        block = BodyState(pos_block, z2, rot_block, z)
        return PushTState(agent, block)

    def render(self, state, width=500, height=500):
        img = jax.pure_callback(
            partial(PushTEnv._callback_render, width=width, height=height),
            jax.ShapeDtypeStruct((3, width, height), jnp.uint8),
            self, state,
        )
        return jnp.transpose(img, (1,2,0))

    def _callback_render(self, state, width, height):
        space, _, _ = self._setup_space(state.agent, state.block)
        # add the target block to the space to render
        self._add_tee(space, self.goal_pose, color=(0,1,0), z=-1)
        return render_space(space, 512, 512, width, height)

    def step(self, state, action, rng_key):
        return jax.pure_callback(PushTEnv._callback_step, state, self, state, action)
    
    def _get_body_state(self, body):
        pos = jnp.array([body.position.x, body.position.y])
        vel = jnp.array([body.velocity.x, body.velocity.y])
        angle = jnp.array(body.angle)
        angular_vel = jnp.array(body.angular_velocity)
        return BodyState(pos, vel, angle, angular_vel)

    def _set_body_state(self, body, state):
        pos = (state.position[0].item(), state.position[1].item())
        vel = (state.velocity[0].item(), state.velocity[1].item())
        angle = state.angle.item()
        angular_vel = state.angular_velocity.item()
        body.angle = angle
        body.position = pos
        body.velocity = vel
        body.angular_vel = angular_vel
        body._space.reindex_shapes_for_body(body)
    
    def _callback_step(self, state, action):
        space, agent, block = self._setup_space(state.agent, state.block)
        dt = 1.0 / self.sim_hz
        for i in range(5):
            if action is not None:
                agent.velocity += action * dt/5
            space.step(dt/5)
        # extract the end state from the space
        agent_state = self._get_body_state(agent)
        block_state = self._get_body_state(block)
        return PushTState(agent_state, block_state)

    def _setup_space(self, agent_state, block_state):
        space = pymunk.Space()
        space.gravity = 0, 0
        space.damping = 0
        # Add walls.
        self._add_segment(space, (5, 506), (5, 5), 2)
        self._add_segment(space, (5, 5), (506, 5), 2)
        self._add_segment(space, (506, 5), (506, 506), 2)
        self._add_segment(space, (5, 506), (506, 506), 2)

        # Add the circle agent
        agent = self._add_circle(space, agent_state, 15,
                color=(65/255, 105/255, 225/255))
        block = self._add_tee(space, block_state,
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

    def _add_circle(self, space, state, radius, color):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.friction = 1
        circle = pymunk.Circle(body, radius)
        circle.color = color
        space.add(body, circle)
        self._set_body_state(body, state)
        return body

    def _add_tee(self, space, state,
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
        body.friction = 1
        # self._set_body_state(body, state)
        space.add(body, shape1, shape2)
        self._set_body_state(body, state)
        return body

def builder(name):
    return PushTEnv()

@dataclass(jax=True)
class PushTPositionObs:
    agent_pos: jnp.array
    block_pos: jnp.array
    block_rot: jnp.array

@dataclass(jax=True)
class PositionObsTransform(PolicyTransform):
    def transform_policy(self, policy):
        return PositionObsPolicy(policy)

@dataclass(jax=True)
class PositionObsPolicy(Policy):
    policy: Policy

    @property
    def rollout_length(self):
        return self.policy.rollout_length
    
    def __call__(self, input):
        obs = input.observation
        obs = PushTPositionObs(
            obs.agent.position,
            obs.block.position,
            obs.block.angle
        )
        input = replace(input, observation=obs)
        return self.policy(input)

# A state-feedback adapter for the PushT environment
# Will run a PID controller under the hood
@dataclass(jax=True)
class PositionControlTransform(PolicyTransform):
    k_p : float = 100
    k_v : float = 20

    def transform_policy(self, policy):
        return PositionControlPolicy(policy, self.k_p, self.k_v)

@dataclass(jax=True)
class PositionControlPolicy(Policy):
    policy: Policy
    k_p : float = 100
    k_v : float = 20

    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        obs = input.observation
        output = self.policy(input)
        a = self.k_p * (output.action - obs.agent.position) + self.k_v * (-obs.agent.velocity)
        return replace(
            output, action=a,
            info=attrs(output.info, target_pos=output.action)
        )

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
    cache = os.path.join(os.getcwd(), '.cache')
    os.makedirs(cache, exist_ok=True)
    model_path = os.path.join(cache, 'pusht_model.ckpt.npy')
    if not os.path.exists(model_path):
        id = "1AMUZvH4mtkCFn4C2MJMgQkfeds2m8dsh&confirm=t"
        gdown.download(id=id, output=model_path, quiet=False)
    tm = jnp.load(model_path, allow_pickle=True).item()
    mapped_params = {}
    for (k,v) in tm.items():
        v = jnp.array(v).T
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
    @jax.jit
    def model(sample, timestep, cond):
        model = ConditionalUnet1D(name='net')
        r = model(sample, timestep, cond)
        return r
    net = hk.transform(model)
    return net, mapped_params

def pretrained_policy():
    from stanza.model.diffusion import DDPMSchedule
    net, params = pretrained_net()
    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=1)
    sample_action_trajectory = jnp.zeros((16, 2))
    obs_scale = {'min': jnp.array([1.3456424e+01, 3.2938293e+01, 5.7471767e+01, 1.0827995e+02, 2.1559125e-04], dtype=jnp.float32), 
           'max': jnp.array([496.14618, 510.9579, 439.9153, 485.6641, 6.2830877], dtype=jnp.float32)}
    action_scale = {'min': jnp.array([12., 25.], dtype=jnp.float32),
              'max': jnp.array([511., 511.], dtype=jnp.float32)}
    
    @jax.jit
    def high_level_policy(input):
        obs = input.observation
        obs = jnp.concatenate(
            (obs.agent_pos, obs.block_pos, obs.block_rot[:,jnp.newaxis]), 
            axis=-1
        )
        # normalize to [0, 1]
        obs = (obs - obs_scale['min'])/(obs_scale['max'] - obs_scale['min'])
        # shift to [-1, 1]
        obs = obs*2 - 1
        obs = obs.reshape((-1))
        model = stanza.Partial(net.apply, params, cond=obs)
        diffused = schedule.sample(input.rng_key, model,
            sample_action_trajectory, num_steps=100)
        traj = diffused[:8]
        # shift from [-1, 1] to [0, 1]
        traj = (traj + 1)/2
        # rescale
        traj = traj*(action_scale['max'] - action_scale['min']) \
                    + action_scale['min']
        return PolicyOutput(traj)

    return chain_transforms(
        ChunkTransform(input_chunk_size=2,
                       output_chunk_size=8),
        # Low-level position controller runs 20x higher
        # frequency than the high-level controller
        # which outputs target positions
        SampleRateTransform(control_interval=10),
        # The low-level controller takes a
        # target position and runs feedback gains
        PositionObsTransform(),
        PositionControlTransform()
    )(high_level_policy)