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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("Set2")

@dataclass
class PlotConfig:
    exp: List[str] = None
    name: str = None

def plot_main(exp):
    config = exp.get("config")
    costs = exp.get("cost")
    gt_costs = exp.get("gt_cost")
    samples = exp.get("samples")
    if samples is None:
        samples = 10*jnp.tile(jnp.expand_dims(jnp.arange(costs.shape[-1]),0), 
                        (costs.shape[0],1))
    subopt = (costs - jnp.expand_dims(gt_costs, -1))/jnp.expand_dims(gt_costs, -1)
    subopt = jnp.maximum(subopt, 1e-6)
    x = samples[0]
    # we use 28, 72 so that with 20 datapoints, the "inner" ones are chosen
    # rather than the outer ones (i.e the max, min in the 25 - 75 percentile range)
    y_low, y, y_high = jnp.percentile(subopt,jnp.array([28, 50, 72]), axis=0)
    # y = jnp.mean(subopt, axis=0)
    # y_std = jnp.std(subopt, axis=0)
    # y_low, y_high = y - y_std, y + y_std
    y_low = jnp.maximum(y_low, 1e-7)
    if config.use_gains:
        color = (0.5, 0.5, 1)
        label = 'With Gains'
    else:
        color = (1, 0.3, 0.3)
        label = 'No Gains'
    plt.plot(x, y, '-', color=color, label=label)
    plt.fill_between(x, y_low, y_high, color=color, alpha=0.2)

def plot_ilqr(exp):
    config = exp.get("config")
    subopt = exp.get("subopts")
    samples = exp.get("samples")
    x = samples
    subopt = jnp.maximum(subopt, 1e-6)
    y_low, y, y_high = jnp.percentile(subopt,jnp.array([28, 50, 72]), axis=1)
    # y = jnp.mean(subopt, axis=1)
    # y_std = jnp.std(subopt, axis=1)
    # y_low, y_high = y - y_std, y + y_std
    y_low = jnp.maximum(y_low, 1e-7)
    if config.jacobian_regularization > 0:
        if config.use_random:
            color = (148/255, 3/255, 252/255)
            label = 'Learning (Rand, JacReg) + iLQR'
        else:
            color = colors[1]
            label = 'Learning (Agg, JacReg) + iLQR'
    else:
        if config.use_random:
            color = (47/255, 212/255, 206/255)
            label = 'Learning (Rand) + iLQR'
        else:
            color = (43/255, 155/255, 1/255)
            label = 'Learning (Agg) + iLQR'
    plt.plot(x, y, '-', color=color, label=label)
    plt.fill_between(x, y_low, y_high, color=color, alpha=0.2)

@activity(PlotConfig)
def make_plot(config, database):
    sns.set()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({'font.size': 24})
    plots = database.open("plots")
    for exp in config.exp:
        logger.info(f"Reading [blue]{exp}[/blue]")
        exp = database.open(exp)
        if 'iterations' in exp.children:
            plot_main(exp)
        elif exp.children:
            plot_ilqr(exp)
    plt.yscale('log')
    plt.ylabel('Cost Suboptimality')
    plt.xlabel('Trajectories')
    if config.name == 'pendulum':
        plt.title("Pendulum")
        plt.xlim([0, 10000])
    else:
        plt.title("Quadrotor")
        plt.xlim([0, 20000])
    plt.legend()
    plots.add(f"plot_{config.name}", Figure(plt.gcf(), width=7, height=5))
