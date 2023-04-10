import plotly.express as px
import plotly.graph_objects as go

from jinx.experiment.runtime import activity, EmptyConfig

from dataclasses import dataclass

import os
import pickle
import seaborn as sns
import jax.numpy as jnp
import jax
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

def line_style(config):
    return {
        'name': f'{config.samples} samples' + ('' if config.use_gains else ', no gains')
    }

@activity("cost_history", EmptyConfig)
def cost_history(config, data):
    fig = go.Figure()
    for d in data:
        fig.add_trace(go.Scatter(x=d.sample_history, y=d.cost_history, **line_style(d.config)))
    return fig

@dataclass
class PlotConfig:
    env_type: str = "pendulum"
    folder: str = "results"
    max_samples: int = 10000


def make_ilqr_command(config):
    cmd = [
        "launch",
        "exp",
        "ilqr_learning",
        "--rng_seed", f"{config.rng_seed}",
        "--traj_seed", f"{config.traj_seed}",
        "--trajectories", f"{config.trajectories}",
        "--jacobian_regularization", f"{config.jacobian_regularization}",
        "--sample_strategy", config.sample_strategy,
        "--env_type", config.env_type,
        "--show_pbar", "True",
        "--save_file", config.save_file,
    ]
    return " ".join(cmd)

@activity("plot_sweep", PlotConfig)
def plot_sweep(config, data):
    # read in ilqr and iter stuff
    root_dir = os.path.abspath(os.path.join(__file__,'..','..'))
    iter = []
    ilqr = {}
    for fl in os.listdir(os.path.join(root_dir, config.folder)):
        path = os.path.join(root_dir, config.folder, fl)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if fl.startswith(f'iter_sweep-{config.env_type}'):
            iter.append(data)
        elif fl.startswith(f'ilqr_sweep-{config.env_type}'):
            key = (data.config.sample_strategy, data.config.jacobian_regularization)
            buf = ilqr.get(key, [])
            buf.append(data)
            ilqr[key] = buf

    print("ilqr keys", ilqr.keys())

    ilqr_dataframes = {}
    ilqr_baseline = {}
    for ilqr_key, ilqr_buf in ilqr.items():
        for x in ilqr_buf:
            if x.config.trajectories == 0:
                print(x.config.traj_seed)
                ilqr_baseline[x.config.traj_seed] = x.final_cost.item()

    for ilqr_key, ilqr_buf in ilqr.items():
        ilqr_data = {'Samples': [], 'Cost': []}
        for x in ilqr_buf:
            bl = ilqr_baseline[x.config.traj_seed]
            if x.config.trajectories > 0:
                ilqr_data['Samples'].append(x.config.trajectories)
                subopt = (x.final_cost.item() - bl)/bl
                ilqr_data['Cost'].append(subopt)
                # if subopt > 0.01:
                if subopt > 0.05:
                # if subopt > 0.001 and x.config.jacobian_regularization > 0 \
                #         and x.config.sample_strategy == 'perturb_ilqr':
                # if x.config.sample_strategy == 'perturb_ilqr':
                    print(subopt, x.final_cost.item())
                    print(make_ilqr_command(x.config))
        ilqr_data = {k: jnp.array(v) for k,v in ilqr_data.items()}
        ilqr_data = pd.DataFrame(ilqr_data)
        ilqr_dataframes[ilqr_key] = ilqr_data

    # consolidate iter data
    iter_gains = {'Samples': [], 'Cost': []}
    iter_no_gains = {'Samples': [], 'Cost': []}
    for x in iter:
        bl = ilqr_baseline[x.config.traj_seed]
        data = iter_gains if x.config.use_gains else iter_no_gains
        if x.sample_history is None:
            data['Samples'].extend([10*x for x in range(x.cost_history.shape[0])])
        else:
            data['Samples'].extend([y.item() for y in x.sample_history])
        data['Cost'].extend([(y.item() - bl)/bl for y in x.cost_history])
    iter_gains = pd.DataFrame(iter_gains)
    iter_no_gains = pd.DataFrame(iter_no_gains)

    # only include up to "samples" part of the plot
    iter_gains = iter_gains[iter_gains['Samples'] <= config.max_samples]
    iter_no_gains = iter_no_gains[iter_no_gains['Samples'] <= config.max_samples]

    #lineplot = partial(sns.lineplot, estimator='median', errorbar=('pi', 50))
    lineplot = sns.lineplot

    THRESH = 1e-4

    iter_gains.Cost = iter_gains.Cost.mask(iter_gains.Cost.lt(THRESH), THRESH)
    iter_no_gains.Cost = iter_no_gains.Cost.mask(iter_no_gains.Cost.lt(THRESH), THRESH)

    lineplot(data=iter_gains, x='Samples', y='Cost', label='Gains')
    lineplot(data=iter_no_gains, x='Samples', y='Cost', label='No Gains')

    def ilqr_key_to_label(ilqr_key):
        sampling_strategy, jac_reg = ilqr_key
        jac_reg = float(jac_reg)
        has_jac_reg = jac_reg > 0.0
        sampling_strategy_to_label = {
            "noisy_ilqr": "Opt",
            "perturb_ilqr": "Pert",
            "random_policy": "Rand",
        }
        if has_jac_reg:
            return "Learning ({}+JacReg) + iLQR".format(sampling_strategy_to_label[sampling_strategy])
        else:
            return "Learning ({}) + iLQR".format(sampling_strategy_to_label[sampling_strategy])

    for ilqr_key, ilqr_data in ilqr_dataframes.items():
        ilqr_data.Cost = ilqr_data.Cost.mask(ilqr_data.Cost.lt(THRESH), THRESH)
        lineplot(data=ilqr_data, x='Samples', y='Cost', label=ilqr_key_to_label(ilqr_key))

    env_type_to_title = {
        'pendulum': 'Pendulum',
        'quadrotor': 'Quadrotor',
    }

    plt.ylabel('Cost Suboptimality')
    plt.xlabel('Trajectories')
    plt.yscale('log')
    #plt.ylim((-0.1, 0.1))
    plt.title(env_type_to_title[config.env_type])
    plt.savefig(f'sweep_plot_{config.env_type}.pdf')
    plt.savefig(f'sweep_plot_{config.env_type}.png')

@dataclass
class PhaseConfig:
    env_type: str = "pendulum"
    rng_seed: int = 42

from .main import iterative_learning, Config as IterConfig
from .ilqr import ilqr_learning, Config as ILQRConfig
from jinx.experiment.dummy import DummyRun
from jinx.logging import logger

# This "analysis" will actually run the experiments
# because it is pretty cheap
@activity("plot_phase", PhaseConfig)
def plot_phase(config, data):
    ilqr_configs = []
    gt_ilqr = ILQRConfig(
        env_type = config.env_type,
        receed=False,
        learned_model=False,
        rng_seed=config.rng_seed,
        traj_seed=config.rng_seed
    )
    ilqr_configs.append(gt_ilqr)

    iter_configs = []
    for t in [1000, 2000, 3000]:
        learned_ilqr = ILQRConfig(
            env_type = config.env_type,
            receed=True,
            trajectories=t,
            learned_model=True,
            rng_seed=config.rng_seed,
            traj_seed=config.rng_seed
        )
        ilqr_configs.append(learned_ilqr)
        gains = IterConfig(
            env_type = config.env_type,
            estimate_grad = True,
            use_gains = True,
            trajectories=t,
            rng_seed=config.rng_seed,
            traj_seed=config.rng_seed
        )
        iter_configs.append(gains)
        no_gains = IterConfig(
            env_type = config.env_type,
            estimate_grad = True,
            use_gains = False,
            trajectories = t,
            rng_seed=config.rng_seed,
            traj_seed=config.rng_seed
        )
        iter_configs.append(no_gains)
    
    ilqr = [ilqr_learning(x, DummyRun()) for x in ilqr_configs]
    iter = [iterative_learning(x, DummyRun()) for x in iter_configs]
    logger.info("plot_phase", "Done with all experiments")

    if config.env_type == "pendulum":
        def plot(xs, us, *args, **kwargs):
            sns.lineplot(x=jnp.squeeze(xs.angle), y=jnp.squeeze(xs.vel), *args, **kwargs)
    
    for res in ilqr:
        name = 'GT' if not res.config.learned_model else f'{res.config.trajectories} traj'
        plot(res.xs, res.us, label=f'{name} iLQR')
    
    for res in iter:
        name = 'Gains' if res.config.use_gains else 'No Gains'
        plot(res.xs, res.us, label=f'{res.config.trajectories} traj {name}')

    plt.title('Phase Plot')
    plt.savefig('phase_plot.png')