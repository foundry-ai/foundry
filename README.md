# Stanza: A batteries-included toolbox for jax-based Machine Learning
:rocket: :rocket: :rocket:

This repository currently concurrently serves as a monorepo for various of my
research projects (located as private git submodules in the projects/) directory.
However it can also be used independently. Eventually I'll get around to cleanly separating
stanza and the various projects once things are a bit more stable


## Installation

Stanza requires python >= 3.8 and [poetry](https://python-poetry.org/) installed. To install simply run
```bash
	poetry install
```
in the root directory. This will automatically create a virtualenv in .venv (so you can point vscode/your favorite editor to that virtualenv).

## Running

Stanza ships with various "examples" which currently serve as pseudo-documentation/tutorials for how to use stanza. To run an example simply use
```bash
	poetry run python examples/[example_name].py
```
To take a peek at the API, various provided examples include:
 - rollout.py: Using the environment/controls toolbox to generate trajectories
 - distributed.py: Example Pool() API usage.

## Features

Stanza includes the following:
 - Rich-based logging/progressbar utilities which can be used inside of jax-jit'd code.
 - Purely functional datasets/data pipelines capable of being wrapped within jax.lax.scan routines.
 - Haiku/Flax/\[Your-Favorite-Jax-NN-Library\]-agnostic training utilities. Compatible with optax optimizers.
 - Nonstochastic convex optimization toolbox: Currently consists of Newton's method. Includes support
   for automatic implicit differentiation of solutions.
 - Jax-based controls toolbox including 
	- MPC (nonlinear + barrier-based constrained)
	- iLQR (based on trajax)
	- Imitation Learning algorithms.
 - Pluggable Jax-based dynamics systems, including jax-compatible wrapper interfaces for OpenAI gym, brax environments.
 - Jax-compatible multi-process and multi-host pool management. Allows for automatic distrubted deployment via 

Planned features
 - Unopinioned jax global state management with wrappers for jax-related transforms.
 - Experiment sub-step caching (i.e to cache the datagen or model train step)
 - Full RL algorithms toolbox.