import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx

import dataclasses
from functools import partial

from stanza.dataclasses import dataclass, field
from stanza.env import (
    EnvWrapper, RenderConfig, 
    ImageRender, SequenceRender,
    HtmlRender, Environment,
    EnvironmentRegistry
)
from stanza.policy import Policy, PolicyInput
from stanza.policy.transforms import Transform, chain_transforms
from stanza.env.mujoco.utils import _quat_to_angle
from stanza import canvas
from stanza.policy.mpc import MPC


XML = """
<mujoco model="reacher">
	<compiler angle="radian" inertiafromgeom="true"/>
	<default>
		<joint armature="1" damping="1" limited="true"/>
		<geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
	</default>
	<option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
	<worldbody>
		<!-- Arena -->
		<geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>
		<geom conaffinity="0" fromto="-.3 -.3 .01 .3 -.3 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
		<geom conaffinity="0" fromto=" .3 -.3 .01 .3  .3 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
		<geom conaffinity="0" fromto="-.3  .3 .01 .3  .3 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
		<geom conaffinity="0" fromto="-.3 -.3 .01 -.3 .3 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
		<!-- Arm -->
		<geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
		<body name="body0" pos="0 0 .01">
			<geom fromto="0 0 0 0.1 0 0" name="link0" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
			<joint axis="0 0 1" limited="false" name="joint0" pos="0 0 0" type="hinge"/>
			<body name="body1" pos="0.1 0 0">
				<joint axis="0 0 1" limited="true" name="joint1" pos="0 0 0" range="-3.0 3.0" type="hinge"/>
				<geom fromto="0 0 0 0.1 0 0" name="link1" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>
				<body name="fingertip" pos="0.11 0 0">
					<geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>
				</body>
			</body>
		</body>
		<!-- Target -->
		<body name="target" pos=".1 -.1 .01">
			<joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.27 .27" ref=".1" stiffness="0" type="slide"/>
			<joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.27 .27" ref="-.1" stiffness="0" type="slide"/>
			<geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
		</body>
	</worldbody>
	<actuator>
		<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint0"/>
		<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint1"/>
	</actuator>
</mujoco>
"""


MJ_MODEL = None
MJX_MODEL = None
def load_mj_model():
    global MJ_MODEL
    if MJ_MODEL is None:
        MJ_MODEL = mujoco.MjModel.from_xml_string(XML)
    return MJ_MODEL

def load_mjx_model():
    global MJX_MODEL
    if MJX_MODEL is None:
        model = load_mj_model()
        with jax.ensure_compile_time_eval():
            MJX_MODEL = mjx.put_model(model)
    return MJX_MODEL

@dataclass
class ReacherObs:
    fingertip_pos: jnp.array
    fingertip_vel: jnp.array

    body0_pos: jnp.array
    body0_rot: jnp.array
    body0_rot_vel: jnp.array

    body1_pos: jnp.array
    body1_rot: jnp.array
    body1_rot_vel: jnp.array

@dataclass
class ReacherPosObs:
    fingertip_pos: jnp.array
    body0_pos: jnp.array
    body0_rot: jnp.array
    body1_pos: jnp.array
    body1_rot: jnp.array

@dataclass
class ReacherState:
    q: jax.Array
    qd: jax.Array

@dataclass
class ReacherEnv(Environment):

    target_id = load_mj_model().body("target").id
    body0_id = load_mj_model().body("body0").id
    body1_id = load_mj_model().body("body1").id
    fingertip_id = load_mj_model().body("fingertip").id
    target_pos = jnp.array([0.1, -0.1])

    @jax.jit
    def sample_action(self, rng_key: jax.Array):
        pos_agent = jax.random.randint(rng_key, (2,), 50, 450).astype(float)
        return pos_agent

    @jax.jit
    def sample_state(self, rng_key):
        return self.reset(rng_key)

    @jax.jit
    def reset(self, rng_key : jax.Array): 
        
        b0_rot, b1_rot = jax.random.split(rng_key, 2)
        body0_rot = jax.random.uniform(b0_rot, (), minval=-jnp.pi, maxval=jnp.pi)
        body1_rot = jax.random.uniform(b1_rot, (), minval=-jnp.pi, maxval=jnp.pi)

        qpos = jnp.concatenate([body0_rot[jnp.newaxis], body1_rot[jnp.newaxis], self.target_pos])
        return ReacherState(
            qpos,
            jnp.zeros_like(qpos)
        )

    @jax.jit
    def step(self, state, action, rng_key = None): 
        data = mjx.make_data(load_mjx_model())
        data = data.replace(qpos=state.q, qvel=state.qd)
        if action is not None:
            # apply torques to the joints
            #xfrc_applied = data.xfrc_applied.at[self.fingertip_id,0:2].set(action)
            xfrc_applied = data.xfrc_applied.at[self.body0_id,0:2].set(action[0])
            xfrc_applied = xfrc_applied.at[self.body1_id,0:2].set(action[1])
            data = data.replace(xfrc_applied=xfrc_applied)
        @jax.jit
        def step_fn(_, data):
            return mjx.step(load_mjx_model(), data)
        data = jax.lax.fori_loop(0, 6, step_fn, data)
        return ReacherState(data.qpos, data.qvel)
    
    @jax.jit
    def observe(self, state): 
        mjx_data = mjx.make_data(load_mjx_model())
        mjx_data = mjx_data.replace(qpos=state.q, qvel=state.qd)
        mjx_data = mjx.forward(load_mjx_model(), mjx_data)
        return ReacherObs(
            fingertip_pos=mjx_data.xpos[self.fingertip_id,:2],
            fingertip_vel=mjx_data.cvel[self.fingertip_id, 3:5],
            body0_pos=mjx_data.xpos[self.body0_id,:2],
            body0_rot=_quat_to_angle(mjx_data.xquat[self.body0_id,:4]),
            body0_rot_vel=mjx_data.cvel[self.body0_id, 2],
            body1_pos=mjx_data.xpos[self.body1_id,:2],
            body1_rot=_quat_to_angle(mjx_data.xquat[self.body1_id,:4]),
            body1_rot_vel=mjx_data.cvel[self.body1_id, 2]
        )
    
    @jax.jit
    def reward(self, state, action, next_state) -> jax.Array:
        obs = self.observe(next_state)
        reward_dist = -jnp.linalg.norm(obs.fingertip_pos - self.target_pos)
        reward_control = -jnp.square(action).sum()
        return reward_dist + reward_control

    

    @partial(jax.jit, static_argnums=(2,3))
    def _render_image(self, obs : ReacherObs, width : int, height : int):
        image = 0.95*jnp.ones((width, height, 3))
        target = canvas.fill(
            canvas.circle(self.target_pos * jnp.array([1, -1]), 0.009),
            color=canvas.colors.Red
        )
        body0 = canvas.transform(
            canvas.fill(
                canvas.box((0, -0.0055), (0.1, 0.0055)),
                color=canvas.colors.LightSlateGray
            ),
            translation=obs.body0_pos*jnp.array([1, -1]),
            rotation=obs.body0_rot
        )
        body1 = canvas.transform(
            canvas.fill(
                canvas.box((0, -0.0055), (0.1, 0.0055)),
                color=canvas.colors.Blue
            ),
            translation=obs.body1_pos*jnp.array([1, -1]),
            rotation=obs.body1_rot
        )
        world = canvas.stack(target, body0, body1)
        translation = (0.25,0.25)
        scale = (width/0.5, height/0.5)
        world = canvas.transform(world,
            translation=translation,
            scale=scale
        )
        image = canvas.paint(image, world)
        return image
    
    @jax.jit
    def render(self, config: RenderConfig, state: ReacherState) -> jax.Array:
        if type(config) == ImageRender or type(config) == SequenceRender:
            obs = ReacherEnv.observe(self, state)
            return self._render_image(obs, config.width, config.height)
        

# def compute_goal_rot(action_pos):
#     """
#     Compute the desired body0_rot given the action position.
#     """
#     rot_action = jnp.atan2(action_pos[1], action_pos[0])
#     # distance from joint0 to action_pos
#     dist_joint0_fingertip = jnp.min(jnp.array([jnp.linalg.norm(action_pos), 0.19]))
#     # distance from joint0 to joint1, i.e. length of body0
#     dist_body0 = 0.1 
#     # angle between vector from joint0 to fingertip and goal body0, always positive
#     rot_fingertip_goal = jnp.arccos((dist_joint0_fingertip/2)/dist_body0)
#     angle = rot_fingertip_goal + rot_action
#     return jnp.atan2(jnp.sin(angle),jnp.cos(angle))

# def compute_goal_pos(body0_rot):
#     return jnp.array([jnp.cos(body0_rot), jnp.sin(body0_rot)]) * 0.1

# def angle_between_vectors(v1, v2):
#     return jnp.arcsin(jnp.cross(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2)))



@dataclass
class MPCTransform(Transform):

    def transform_policy(self, policy):
        return MPCPolicy(policy)
    
    def transform_env(self, env):
        return MPCEnv(env)

@dataclass
class MPCPolicy(Policy):
    policy: Policy

    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        obs = input.observation
        obs = ReacherPosObs(
            fingertip_pos=obs.fingertip.position,
            body0_pos=obs.body0.position,
            body0_rot=obs.body0.rotation,
            body1_pos=obs.body1.position,
            body1_rot=obs.body1.rotation
        )
        output = self.policy(input)
        a = 0
        return dataclasses.replace(
            output, action=a
        )

@dataclass
class MPCEnv(EnvWrapper):

    def step(self, state, action, rng_key=None):
        obs = ReacherEnv.observe(self.base, state)
        cost_fn = lambda state: jnp.square(ReacherEnv.observe(self.base, state).fingertip_pos - action).sum()
        mpc = MPC(
            action_sample=(jnp.array([0]), jnp.array([0])),
            cost_fn = lambda states, actions: cost_fn(states),
            model_fn = lambda env_state, action, _: self.base.step(env_state, action),
            horizon_length=10
        )
        a = mpc(PolicyInput(state))
        res = self.base.step(state, a, rng_key)
        return res

@dataclass
class PositionalObsTransform(Transform):
    def transform_policy(self, policy):
        return PositionalObsPolicy(policy)
    
    def transform_env(self, env):
        return PositionalObsEnv(env)

@dataclass
class PositionalObsPolicy(Policy):
    policy: Policy

    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        obs = input.observation
        obs = ReacherPosObs(
            fingertip_pos=obs.fingertip.position,
            body0_pos=obs.body0.position,
            body0_rot=obs.body0.rotation,
            body1_pos=obs.body1.position,
            body1_rot=obs.body1.rotation
        )
        input = dataclasses.replace(input, observation=obs)
        return self.policy(input)

@dataclass
class PositionalObsEnv(EnvWrapper):
    def observe(self, state):
        obs = self.base.observe(state)
        return ReacherPosObs(
            fingertip_pos=obs.fingertip_pos,
            body0_pos=obs.body0_pos,
            body0_rot=obs.body0_rot,
            body1_pos=obs.body1_pos,
            body1_rot=obs.body1_rot
        )



environments = EnvironmentRegistry[ReacherEnv]()
environments.register("", ReacherEnv)
def _make_positional(**kwargs):
    env = ReacherEnv(**kwargs)
    return chain_transforms(
        MPCTransform(),
        PositionalObsTransform
    ).transform_env(env)
environments.register("positional", _make_positional)