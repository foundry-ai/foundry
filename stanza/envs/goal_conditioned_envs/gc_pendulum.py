import math
import jax.random as random
import jax.numpy as jnp
from stanza.goal_conditioned import GCEnvironment, EndGoal
from stanza.envs import Environment
from jax.random import PRNGKey
from stanza.envs.pendulum import State as PendulumState


def goal_reward(state, next_state, end_state):
    angle_diff = next_state.angle - state.angle
    vel_diff = next_state.vel - state.vel
    angle_rew = 32 * angle_diff * jnp.sign(end_state.angle - next_state.angle)
    vel_rew = vel_diff * jnp.sign(end_state.vel-next_state.vel)
    return angle_rew + vel_rew

def cost_to_goal( x, u, x_goal):
    x = jnp.stack((x.angle, x.vel), -1)
    x_goal = jnp.stack((x_goal.angle, x_goal.vel), -1)
    diff = (x - x_goal)
    x_cost = jnp.sum(diff[:-1]**2)
    xf_cost = jnp.sum(diff[-1]**2)
    if u == None:
        u_cost = 0
    else:
        u_cost = jnp.sum(u**2)
    return 5*xf_cost + 2*x_cost + u_cost

def gc_cost(gc_state, action):
    return cost_to_goal(gc_state.env_state, action, gc_state.goal.end_state)
      

def gc_reward(gc_state, action, next_state ):
    env_state, goal = gc_state.env_state, gc_state.goal
    end_state = goal.end_state
    #TODO penalize reward
    return goal_reward(env_state,next_state,end_state)
    #return 3 - (1 * cost_to_goal(env_state, action, end_state))

def g_done(gc_state):
    x = gc_state.env_state
    x_goal = gc_state.goal.end_state
    return (cost_to_goal(x =x,u=None,x_goal = x_goal) < .03*.03)


def make_gc_pendulum_env(env : Environment, gs_sampler):
    return GCEnvironment(env=env, gs_sampler=gs_sampler,
                            gc_reward=gc_reward, 
                            gc_cost=gc_cost,g_done=g_done)

def gc_pendulum_uniform_goal_sampler(min_angle : float = -math.pi/2, max_angle : float = math.pi):
    def sampler(key : PRNGKey):
        angle = random.uniform(key,shape=(), minval=min_angle,maxval=max_angle)
        vel = jnp.zeros(())
        return EndGoal(end_state=PendulumState(angle, vel))
    return sampler

