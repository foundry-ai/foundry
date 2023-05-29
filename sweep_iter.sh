#!/usr/bin/bash

poetry run activity ilc:iterative_learning --use_gains=False --exp_name iterative_pendulum_no_gains
poetry run activity ilc:iterative_learning --use_gains=True --exp_name iterative_pendulum_gains
