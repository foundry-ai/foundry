#!/usr/bin/bash
poetry run activity ilc:make_plot --name pendulum \
	--exp iterative_pendulum_gains,iterative_pendulum_no_gains,ilqr_pendulum_noreg,ilqr_pendulum_reg,ilqr_pendulum_reg_random,ilqr_pendulum_noreg_random
