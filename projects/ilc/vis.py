import plotly.graph_objects as go

from stanza.runtime import activity
from stanza.runtime.database import Figure
from stanza.util.logging import logger

from dataclasses import dataclass

import os
import pickle
import jax.numpy as jnp
import jax
from functools import partial
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class PlotConfig:
    exp: List[str] = None

def plot_main(exp):
    config = exp.get("config")
    costs = exp.get("cost")
    gt_costs = exp.get("gt_cost")
    samples = exp.get("samples")
    if samples is None:
        samples = 20*jnp.tile(jnp.expand_dims(jnp.arange(costs.shape[-1]),0), 
                        (costs.shape[0],1))
    subopt = (costs - jnp.expand_dims(gt_costs, -1))/jnp.expand_dims(gt_costs, -1)
    subopt = jnp.maximum(subopt, 1e-7)
    x = samples[0]
    y_low, y, y_high = jnp.percentile(subopt,jnp.array([30, 50, 70]), axis=0)
    # y = jnp.mean(subopt, axis=0)
    # y_std = jnp.std(subopt, axis=0)
    # y_low, y_high = y - y_std, y + y_std
    y_low = jnp.maximum(y_low, 1e-7)
    if config.use_gains:
        color = 'r'
        label = 'With Gains'
    else:
        color = 'b'
        label = 'No Gains'
    plt.plot(x, y, f'{color}-', label=label)
    plt.fill_between(x, y_low, y_high, color=color, alpha=0.2)

def plot_ilqr(exp):
    config = exp.get("config")
    subopt = exp.get("subopts")
    samples = exp.get("samples")
    x = samples
    subopt = jnp.maximum(subopt, 1e-7)
    y_low, y, y_high = jnp.percentile(subopt,jnp.array([30, 50, 70]), axis=1)
    # y = jnp.mean(subopt, axis=1)
    # y_std = jnp.std(subopt, axis=1)
    # y_low, y_high = y - y_std, y + y_std
    y_low = jnp.maximum(y_low, 1e-7)
    if config.jacobian_regularization > 0:
        if config.use_random:
            color = 'cyan'
            label = 'Learning (Rand,JacReg) + iLQR'
        else:
            color = 'g'
            label = 'Learning (JacReg) + iLQR'
    else:
        if config.use_random:
            color = 'brown'
            label = 'Learning (Rand) + iLQR'
        else:
            color = 'purple'
            label = 'Learning + iLQR'
    plt.plot(x, y, '-', color=color, label=label)
    plt.fill_between(x, y_low, y_high, color=color, alpha=0.2)

@activity(PlotConfig)
def make_plot(config, database):
    sns.set()
    plots = database.open("plots")
    for exp in config.exp:
        logger.info(f"Reading [blue]{exp}[/blue]")
        exp = database.open(exp)
        if 'iterations' in exp.children:
            plot_main(exp)
        else:
            plot_ilqr(exp)
    plt.yscale('log')
    plt.ylabel('Cost Suboptimality')
    plt.xlabel('Trajectories')
    plt.legend()
    plots.add("plot", Figure(plt.gcf()))
