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
        samples = jnp.tile(jnp.expand_dims(jnp.arange(costs.shape[-1]),0), 
                        (costs.shape[0],1))
    subopt = (costs - jnp.expand_dims(gt_costs, -1))/jnp.expand_dims(gt_costs, -1)

    x = samples[0]
    y = jnp.mean(subopt, axis=0)
    y_std = jnp.std(subopt, axis=0)
    y_low, y_high = y - y_std/2, y + y_std/2
    #y_low, y, y_high = jnp.percentile(subopt,jnp.array([25, 50, 75]), axis=0)
    if config.use_gains:
        color = 'r'
        label = 'With Gains'
    else:
        color = 'b'
        label = 'No Gains'
    plt.plot(x, y, f'{color}-', label=label)
    plt.fill_between(x, y_low, y_high, color=color, alpha=0.2)

def plot_ilqr(exp):
    subopt = exp.get("subopts")
    samples = exp.get("samples")
    print(subopt.shape)
    x = samples
    y_low, y, y_high = jnp.percentile(subopt,jnp.array([10, 50, 90]), axis=1)
    plt.plot(x, y, 'g-', label='iLQR')
    plt.fill_between(x, y_low, y_high, color='g', alpha=0.2)

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
