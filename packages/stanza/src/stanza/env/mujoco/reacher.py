import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx

import dataclasses
from functools import partial, cached_property

from stanza.dataclasses import dataclass, field
from stanza.env import (
    EnvWrapper, RenderConfig, ObserveConfig,
    ImageRender, ImageRenderTraj,
    HtmlRender, Environment,
    EnvironmentRegistry
)
from stanza.env.mujoco.core import (
    MujocoEnvironment, SystemState, 
    SimulatorState, Action,
    quat_to_angle, render_2d,
    orientation_error
)
from stanza.policy import Policy, PolicyInput
from stanza.env.transforms import (
    EnvTransform, ChainedTransform,
    MultiStepTransform
)
from stanza import canvas
from stanza.util import jax_static_property
from jax.random import PRNGKey


XML = """
<mujoco model="reacher">
	<compiler angle="radian" inertiafromgeom="true"/>
	<default>
		<joint armature="1" damping="1" limited="true"/>
		<geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
	</default>
	<option iterations="1"/>
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
                    <site name="eef" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 0.5" type="sphere" group="1" />
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


@dataclass
class ReacherObs:
    fingertip_pos: jnp.array = None
    fingertip_vel: jnp.array = None

    eef_pos: jax.Array = None # (n_robots, 3,) -- end-effector position
    eef_vel: jax.Array = None # (n_robots, 3,) -- end-effector velocity
    eef_ori_mat: jax.Array = None # (n_robots, 4,) -- end-effector quaternion
    eef_ori_vel: jax.Array = None # (n_robots, 3,) -- end-effector angular velocity

    body0_pos: jnp.array = None
    body0_rot: jnp.array = None
    body0_rot_vel: jnp.array = None

    body1_pos: jnp.array = None
    body1_rot: jnp.array = None
    body1_rot_vel: jnp.array = None

@dataclass
class ReacherPosObs:
    fingertip_pos: jnp.array = None
    eef_pos: jax.Array = None
    eef_ori_mat: jax.Array = None
    body0_pos: jnp.array = None
    body0_rot: jnp.array = None
    body1_pos: jnp.array = None
    body1_rot: jnp.array = None

@dataclass
class ReacherAgentPos:
    pass

@dataclass
class ReacherEnv(MujocoEnvironment[SimulatorState]):
    body0_id: int = 1
    body1_id: int = 2
    fingertip_id: int = 3
    eef_id: int = 0
    target_id: int = 4
    target_pos: jax.Array = field(default_factory=lambda:jnp.array([-0.1, 0.1]))

    @jax_static_property
    def model(self):
        return mujoco.MjModel.from_xml_string(XML)

    @jax.jit
    def sample_action(self, rng_key: jax.Array):
        return jax.random.randint(rng_key, (12,), 0, 10).astype(jnp.float32)

    @jax.jit
    def sample_state(self, rng_key):
        return self.reset(rng_key)

    @jax.jit
    def reset(self, rng_key : jax.Array) -> SimulatorState: 
        b0_rot, b1_rot = jax.random.split(rng_key, 2)
        body0_rot = jax.random.uniform(b0_rot, (), minval=-jnp.pi, maxval=jnp.pi)
        body1_rot = jax.random.uniform(b1_rot, (), minval=-jnp.pi, maxval=jnp.pi)

        qpos = jnp.concatenate([body0_rot[jnp.newaxis], body1_rot[jnp.newaxis], self.target_pos])
        # qpos = jnp.zeros((4,), dtype=jnp.float32)
        return self.full_state(SystemState(
            jnp.zeros((), dtype=jnp.float32), 
            qpos, 
            jnp.zeros_like(qpos), 
            jnp.zeros((0,), dtype=jnp.float32)
        ))

    
    @jax.jit
    def observe(self, state, config : ObserveConfig | None = None):
        if config is None: config = ReacherObs()
        data = self.simulator.system_data(state)
        if isinstance(config, ReacherObs):
            system_state = self.simulator.reduce_state(state)
            jacp, jacr = self.native_simulator.get_jacs(system_state, self.eef_id)
            return ReacherObs(
                fingertip_pos=data.xpos[self.fingertip_id,:],
                fingertip_vel=data.cvel[self.fingertip_id, 3:5],
                eef_pos=data.site_xpos[self.eef_id, :],
                eef_vel=jnp.dot(jacp, system_state.qvel),
                eef_ori_mat=data.site_xmat[self.eef_id, :].reshape([3, 3]),
                eef_ori_vel=jnp.dot(jacr, system_state.qvel),
                body0_pos=data.xpos[self.body0_id,:2],
                body0_rot=quat_to_angle(data.xquat[self.body0_id,:4]),
                body0_rot_vel=data.cvel[self.body0_id, 2],
                body1_pos=data.xpos[self.body1_id,:2],
                body1_rot=quat_to_angle(data.xquat[self.body1_id,:4]),
                body1_rot_vel=data.cvel[self.body1_id, 2]
            )
        elif isinstance(config, ReacherAgentPos):
            return jnp.concatenate([data.xpos[self.eef_id, :], data.site_xmat[self.eef_id, :]])
        else:
            raise ValueError("Unsupported observation type")

    @jax.jit
    def reward(self, state, action, next_state) -> jax.Array:
        obs = self.observe(next_state)
        reward_dist = -jnp.linalg.norm(obs.eef_pos[:2] - self.target_pos)
        reward_control = -jnp.sum(jnp.square(action))
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
            rotation=-obs.body1_rot
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
    
    # @jax.jit
    # def render(self, state: SimulatorState, config: RenderConfig):
    #     if type(config) == ImageRender or type(config) == SequenceRender:
    #         obs = ReacherEnv.observe(self, state)
    #         return self._render_image(obs, config.width, config.height)
    #     elif type(config) == HtmlRender:
    #         if data.qpos.ndim == 1:
    #             data = jax.tree_map(lambda x: x[None], data)
    #         return MujocoEnvironment.brax_render(self.model, data)
        


'''
@dataclass
class MPCTransform(Transform):
    def transform_policy(self, policy):
        raise NotImplementedError()

    def transform_env(self, env):
        return MPCEnv(env)


@dataclass
class MPCEnv(EnvWrapper):
    def step(self, state, action, rng_key=None):
        cost_fn = lambda state: jnp.sum(jnp.square(ReacherEnv.observe(self.base, state).fingertip_pos - action))
        mpc = MPC(
            action_sample=(jnp.array([0]), jnp.array([0])),
            cost_fn = lambda states, actions: cost_fn(jax.tree.map(lambda x: x[-1], states)),
            model_fn = lambda env_state, action, _: self.base.step(env_state, action),
            horizon_length=5
        )
        a = mpc(PolicyInput(state))
        res = self.base.step(state, a.action, rng_key)
        return res
'''

@dataclass
class PositionalControlTransform(EnvTransform):
    k_p : jax.Array = jnp.array([15]*6) # (6,) -- [0:3] corresponds to position, [3:6] corresponds to orientation
    k_d : jax.Array = jnp.array([2]*6) # (6,) -- [0:3] corresponds to position, [3:6] corresponds to orientation

    def apply(self, env):
        return PositionalControlEnv(env, self.k_p, self.k_d)


@dataclass
class PositionalControlEnv(EnvWrapper):
    k_p : jax.Array
    k_d : jax.Array

    def step(self, state, action, rng_key=None):
        obs = self.base.observe(state)
        if action is not None:
            action_pos = action[:3]
            action_ori_mat = action[3:].reshape([3,3])
            data = self.simulator.system_data(state)
            qvel_indices = jnp.array([0,1])
            eef_id = self.eef_id

            system_state = self.simulator.reduce_state(state)
            jacp, jacv = self.native_simulator.get_jacs(system_state, eef_id)
            J_pos = jnp.array(jacp.reshape((3, -1))[:, qvel_indices])
            J_ori = jnp.array(jacv.reshape((3, -1))[:, qvel_indices])
            
            mass_matrix = self.native_simulator.get_fullM(system_state)
            mass_matrix = jnp.reshape(mass_matrix, (len(data.qvel), len(data.qvel)))
            mass_matrix = mass_matrix[qvel_indices, :][:, qvel_indices]

            # Compute lambda matrices
            mass_matrix_inv = jnp.linalg.inv(mass_matrix)
            lambda_pos_inv = J_pos @ mass_matrix_inv @ J_pos.T
            lambda_ori_inv = J_ori @ mass_matrix_inv @ J_ori.T
            # take the inverses, but zero out small singular values for stability
            lambda_pos = jnp.linalg.pinv(lambda_pos_inv)
            lambda_ori = jnp.linalg.pinv(lambda_ori_inv)

            pos_error = action_pos - obs.eef_pos
            vel_pos_error = -obs.eef_vel

            eef_ori_mat = obs.eef_ori_mat
            #print(action_ori_mat.shape, eef_ori_mat.shape)
            ori_error = orientation_error(action_ori_mat, eef_ori_mat)
            vel_ori_error = -obs.eef_ori_vel

            F_r = self.k_p[:3] * pos_error + self.k_d[:3] * vel_pos_error
            Tau_r = self.k_p[3:] * ori_error + self.k_d[3:] * vel_ori_error
            compensation = data.qfrc_bias[qvel_indices]

            torques = J_pos.T @ lambda_pos @ F_r + J_ori.T @ lambda_ori @ Tau_r + compensation
            a = jnp.zeros(self.model.nu, dtype=jnp.float32)
            a = a.at[qvel_indices].set(torques)
            #jax.debug.print("J_pos: {J_pos}, J_ori: {J_ori}, lambda_pos: {lambda_pos}, lambda_ori: {lambda_ori}", J_pos=J_pos, J_ori=J_ori, lambda_pos=lambda_pos, lambda_ori=lambda_ori)
            #jax.debug.print("{s}, {t}, {u}", s=J_pos.T @ lambda_pos @ F_r, t=J_ori.T @ lambda_ori @ Tau_r, u=compensation)
            #jax.debug.print("{s}", s=a)
        else: 
            a = jnp.zeros(self.model.nu, dtype=jnp.float32)
        return self.base.step(state, a, None)

@dataclass
class PositionalObsTransform(EnvTransform):
    def apply(self, env):
        return PositionalObsEnv(env)

@dataclass
class PositionalObsEnv(EnvWrapper):
    def observe(self, state, config=None):
        if config is None: config = ReacherPosObs()
        if not isinstance(config, ReacherPosObs):
            return self.base.observe(state, config)
        obs = self.base.observe(state, ReacherObs())
        return ReacherPosObs(
            fingertip_pos=obs.fingertip_pos,
            eef_pos = obs.eef_pos,
            eef_ori_mat = obs.eef_ori_mat,
            body0_pos=obs.body0_pos,
            body0_rot=obs.body0_rot,
            body1_pos=obs.body1_pos,
            body1_rot=obs.body1_rot
        )



environments = EnvironmentRegistry[ReacherEnv]()
environments.register("", ReacherEnv)
def _make_positional(**kwargs):
    env = ReacherEnv(**kwargs)
    return ChainedTransform(
        PositionalControlTransform(),
        PositionalObsTransform
    ).apply(env)
environments.register("positional", _make_positional)