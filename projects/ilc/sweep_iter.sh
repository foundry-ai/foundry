#!/usr/bin/bash

poetry run activity ilc:iterative_learning --use_gains=False --exp_name iterative_pendulum_no_gains #--samples 0
poetry run activity ilc:iterative_learning --use_gains=True --exp_name iterative_pendulum_gains #--samples 0
