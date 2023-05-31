#!/usr/bin/bash
#poetry run activity ilc:ilqr_learning --env quadrotor --exp_name ilqr_quadrotor_noreg
#poetry run activity ilc:ilqr_learning --env quadrotor --jacobian_regularization 1 --exp_name ilqr_quadrotor_reg
poetry run activity ilc:ilqr_learning --env quadrotor --use_random=True --exp_name ilqr_quadrotor_noreg_random
poetry run activity ilc:ilqr_learning --env quadrotor --use_random=True --jacobian_regularization 1 --exp_name ilqr_quadrotor_reg_random
