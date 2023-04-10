from stanza.runtime import activity
from stanza.util.dataclasses import dataclass

import stanza.envs as envs
import stanza.policies as policies
import jax.numpy as jnp

from stanza.policies.mpc import BarrierMPC
from stanza.util.logging import logger
from jax.random import PRNGKey
from functools import partial

import time
import jax
import plotly.graph_objects as go
import plotly.subplots as subplots

@dataclass(frozen=True)
class VisControllerConfig:
    env: str = "linear/di"
    constrain: bool = True

def compute_mpc(env, xs, eta):
    mpc = BarrierMPC(
        # Sample action
        action_sample=env.sample_action(PRNGKey(0)),
        barrier_sdf=env.constraints if eta is not None else None,
        center_state=jnp.zeros_like(env.sample_state(PRNGKey(0))),
        center_action=jnp.zeros_like(env.sample_action(PRNGKey(0))),
        cost_fn=env.cost,
        model_fn=env.step,
        horizon_length=5,
        eta=eta
    )
    res = jax.vmap(mpc)(xs)
    # rollout a trajectory
    ro = jax.vmap(lambda x: policies.rollout(
        env.step, x,
        policy=mpc,
        length=10
    ))
    r = ro(jnp.array([[9, -1], [-5, 0], [0, 1.6]]))
    return res, r

@activity(VisControllerConfig)
def vis_controller(config, database):
    env = envs.create(config.env)

    # generate a grid of points to evaluate
    # the controller at
    x0 = jnp.linspace(-10, 10, 100)
    x1 = jnp.linspace(-4.5, 4.5, 100)
    xs = jnp.stack(jnp.meshgrid(x0, x1))
    xs = xs.reshape((xs.shape[0], -1))
    xs = xs.transpose(1, 0)

    logger.info("Evaluating controller:")
    t = time.time()
    vmpc = jax.pmap(partial(compute_mpc, env, xs))
    eta = jnp.array([0.001, 1, 5, 10])
    res, rolls = vmpc(eta)
    logger.info("Took {} seconds", time.time() - t)

    # r = mpc(jnp.array([0, 2]))
    # logger.info("evaled: {}", r.action)
    # logger.info("cost: {}", r.extra.cost)
    inputs = jnp.reshape(res.action,(eta.shape[0], x0.shape[0], x1.shape[0]))
    costs = jnp.reshape(res.extra.cost,(eta.shape[0], x0.shape[0], x1.shape[0]))
    costs = jnp.where(costs > 1e10, jnp.nan, costs)
    cost_min, cost_max = jnp.nanmin(costs).item(), jnp.nanmax(costs).item()

    cols = 4
    rows = (eta.shape[0] + cols - 1) // cols

    inputs_fig = subplots.make_subplots(rows=rows, cols=cols,
            horizontal_spacing=0.05,
            specs=rows*[cols*[{'is_3d': True}]],
            subplot_titles=[f'$\eta={e:0.3}$' for e in eta])
    cost_fig = subplots.make_subplots(rows=rows, cols=cols,
            horizontal_spacing=0.05,
            specs=rows*[cols*[{'is_3d': True}]],
            subplot_titles=[f'$\eta={e:0.3}$' for e in eta])
    inputs_ct_fig = subplots.make_subplots(rows=rows, cols=cols,
                horizontal_spacing=0.05,
                subplot_titles=[f'$\eta={e:0.3}$' for e in eta])
    cost_ct_fig = subplots.make_subplots(rows=rows, cols=cols,
                horizontal_spacing=0.05)
                #subplot_titles=[f'$\eta={e:0.3}$' for e in eta])

    for i, (e, input, cost, t_s, t_a, t_c) in enumerate(zip(
                    eta, inputs, costs, 
                    rolls.states, rolls.actions, rolls.extras.cost
                )):
        r = (i // cols) + 1
        c = (i % cols) + 1
        inputs_fig.add_trace(go.Surface(
            x=x0, y=x1, z=input,
            showscale=i == 0, cmax=1, cmin=-1
        ), r, c)
        cost_fig.add_trace(go.Surface(
            x=x0, y=x1, z=cost,
            showscale=i == 0, cmax=cost_max, cmin=cost_min
        ), r, c)
        # make the contour-based versions as well
        inputs_ct_fig.add_trace(go.Contour(
            x=x0, y=x1, z=input,
            colorscale='pinkyl',
            showscale=i==0, zmax=1, zmin=-1
        ), r, c)
        cost_ct_fig.add_trace(go.Contour(
            x=x0, y=x1, z=cost,
            colorscale='pinkyl',
            showscale=i==0, zmax=cost_max, zmin=cost_min
        ), r, c)
        for (ti_s, ti_a, ti_c, color) in zip(t_s, t_a, t_c,
                ['#37306B', '#4C4B16', '#6E8EB1']):
            style = dict(
                line=dict(width=3, color=color),
                marker=dict(size=7),
                showlegend=False
            )
            inputs_ct_fig.add_trace(go.Scatter(
                x=ti_s[:,0],y=ti_s[:,1], **style
            ), r, c)
            cost_ct_fig.add_trace(go.Scatter(
                x=ti_s[:,0],y=ti_s[:,1], **style
            ), r, c)

        #inputs_ct_fig.update_xaxes(title_text="$x_0$", row=r, col=c)
        cost_ct_fig.update_xaxes(title_text="$x_0$", row=r, col=c)
        if c == 1:
            inputs_ct_fig.update_yaxes(title_text="$x_1$", tickvals=[-4, -2, 0, 2, 4], row=r, col=c)
            cost_ct_fig.update_yaxes(title_text="$x_1$", tickvals=[-4,-2,0,2,4], row=r, col=c)

    # inputs_ct_fig.update_layout(height=400, width=4*400,
    #               title_text="Control Landscape for Various Eta",
    #               title_x=0.5)
    # cost_ct_fig.update_layout(height=400, width=4*400,
    #               title_text="Cost Landscape for Various Eta",
    #               title_x=0.5)
    inputs_ct_fig.update_layout(height=175, width=4*200 + 30, margin=dict(l=30,r=0,b=0,t=20))
    cost_ct_fig.update_layout(height=180, width=4*200 + 30, margin=dict(l=30,r=0,b=5,t=10))
    inputs_ct_fig.write_image("inputs_eta_sweep.pdf")
    cost_ct_fig.write_image("cost_eta_sweep.pdf")
    inputs_ct_fig.update_layout(height=None, width=None, margin=None)
    cost_ct_fig.update_layout(height=None, width=None, margin=None)
    # inputs_fig.show()
    # cost_fig.show()
    # inputs_ct_fig.show()
    # cost_ct_fig.show()