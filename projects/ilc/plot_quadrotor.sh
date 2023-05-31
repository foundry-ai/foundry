#!/usr/bin/bash
poetry run activity ilc:make_plot --name quadrotor --exp iterative_quadrotor_gains,iterative_quadrotor_no_gains,ilqr_quadrotor_noreg,ilqr_quadrotor_reg,ilqr_quadrotor_reg_random,ilqr_quadrotor_noreg_random
