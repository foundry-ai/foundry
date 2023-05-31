#!/usr/bin/bash

poetry run activity ilc:ilqr_learning --env pendulum --use_random=True --exp_name ilqr_pendulum_noreg_random
poetry run activity ilc:ilqr_learning --env pendulum --use_random=True --jacobian_regularization 1 --exp_name ilqr_pendulum_reg_random
poetry run activity ilc:ilqr_learning --env pendulum --exp_name ilqr_pendulum_noreg
poetry run activity ilc:ilqr_learning --env pendulum --jacobian_regularization 1 --exp_name ilqr_pendulum_reg
