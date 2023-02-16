# Stanza: A batteries-included toolbox for jax-based Machine Learning :rocket: :rocket: :rocket:

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

## Running Examples

Stanza ships with various "examples" which currently serve as pseudo-documentation/tutorials for how to use stanza. To run an example simply use
```bash
	poetry run python examples/[example_name].py
```
To take a peek at the API, various provided examples include:
 - rollout.py: Using the environment/controls toolbox to generate trajectories
 - distributed.py: Example Pool() API usage.

## Running Activities

Stanza includes its own entrypoint management and configuration parsing system.
An entrypoint is called an "activity". To run an activity, use
```bash
	poetry run activity module.submodule:activity_name
```
Activities are functions which take (config, run) as arguments and have the @activity(Dataclass) decorator applied (from stanza.util.runtime). This has the advantage that activities interface with the stanza distributed capabilities. For instance, to launch the activity from within a docker container you can use
```bash
	poetry run activity --target docker://localhost module.submodule:activity_name
```
The activity launch script can also be used to load json, yaml or python-code-as-config based configurations. For instance
```bash
	poetry run activity module.submodule:activity_name --json path_to_json_file.json
	poetry run activity module.submodule:activity_name --yaml path_to_yaml_file.yaml
	poetry run activity module.submodule:activity_name --config module.submdoule:config_entrypoint
```
The JSON or YAML files should return a (potentially recursive) dictionary of overrides. The --config python code should return the appropriate dataclass (the expected dataclass is passed as the first argument to the config_entrypoint, along with the remaining, unparsed, command-line arguments).

If the loaded config is a list, it will be interpreted as a sweep. --config, --json, and --yaml can be called multiple times to append configs to the sweep.
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