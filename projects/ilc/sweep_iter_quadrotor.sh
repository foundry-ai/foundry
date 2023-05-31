#!/usr/bin/bash

poetry run activity ilc:iterative_learning --use_gains=False --env_type quadrotor --exp_name iterative_quadrotor_no_gains
poetry run activity ilc:iterative_learning --use_gains=True --env_type quadrotor --exp_name iterative_quadrotor_gains
