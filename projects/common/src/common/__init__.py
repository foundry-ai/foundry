from stanza.dataclasses import dataclass, field, replace
from stanza.runtime import ConfigProvider

import optax
import jax
import jax.numpy as jnp

@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    lr_schedule: str = "constant"
    cycles: int = 1
    # length to multiply the number of
    # iterations per cycle, every cycle
    cycle_mult: float = 1.
    warmup_schedule: str | None = None
    # if none, will use a "good" value based on warmup_schedule and the number of iterations 
    warmup_steps: int | None = None 

    def make_lr_schedule(self, iterations):
        warmup_steps = (
            100
            if self.warmup_steps is None  and self.warmup_schedule is not None 
            else (self.warmup_steps or 0)
        )
        if self.warmup_schedule is None:
            warmup = None
        elif self.warmup_schedule == "linear":
            warmup = optax.linear_schedule(0, self.lr, warmup_steps)
        elif self.warmup_schedule == "cosine":
            warmup = optax.cosine_decay_schedule(0, warmup_steps, self.lr)
        else:
            raise ValueError(f"Unknown warmup schedule: {self.warmup_schedule}")
        if warmup is not None:
            iterations = iterations - warmup_steps

        if self.lr_schedule == "constant":
            schedule_builder = lambda s: optax.constant_schedule(self.lr)
        elif self.lr_schedule == "linear":
            schedule_builder = lambda s: optax.linear_schedule(self.lr, 0., s)
        elif self.lr_schedule == "cosine":
            schedule_builder = lambda s: optax.cosine_decay_schedule(self.lr, s)
        else:
            raise ValueError(f"Unknown learning rate schedule: {self.lr_schedule}")

        if self.cycles > 1:
            # total length is base*(1 + m + m^2 ... + m^(cycles - 1)) = 1/(1 - m)
            if self.cycle_mult == 1.:
                base_units = self.cycles
            else:
                base_units = 1/(1 - self.cycle_mult) - (self.cycle_mult ** self.cycles) / (1 - self.cycle_mult)
            base_steps = iterations // base_units
            schedules = []
            boundaries = []
            for i in range(self.cycles):
                part_iterations = base_steps * (self.cycle_mult ** i)
                schedules.append(schedule_builder(part_iterations))
                if i == 1: boundaries.append(base_steps)
                elif i > 1: boundaries.append(boundaries[-1] + part_iterations)
            schedule = optax.join_schedules(schedules, boundaries)
        else:
            schedule = schedule_builder(iterations)
        if warmup is None:
            return schedule
        else:
            return optax.join_schedules([warmup, schedule], [warmup_steps])

    def make_optimizer(self, iterations):
        raise NotImplementedError()

    @staticmethod
    def default():
        return AdamConfig(lr_schedule="cosine", warmup_schedule="linear")

@dataclass
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float | None = 0.0001

    def make_optimizer(self, iterations):
        if self.weight_decay is not None:
            return optax.adamw(learning_rate=self.make_lr_schedule(iterations),
                            b1=self.beta1, b2=self.beta2, eps=self.epsilon,
                            weight_decay=self.weight_decay)
        else:
            return optax.adam(learning_rate=self.make_lr_schedule(iterations), 
                            b1=self.beta1, b2=self.beta2, eps=self.epsilon)

    def parse(self, config: ConfigProvider) -> "AdamConfig":
        return config.get_dataclass(self)

@dataclass
class SGDConfig(OptimizerConfig):
    lr_schedule: str = "constant" # The learning rate schedule
    momentum: float | None = None
    nesterov: bool = False
    weight_decay: float | None = None

    def make_optimizer(self, iterations):
        optim = optax.sgd(learning_rate=self.make_lr_schedule(iterations),
                         momentum=self.momentum, nesterov=self.nesterov)
        if self.weight_decay:
            optim = optax.chain(optax.add_decayed_weights(self.weight_decay), optim)
        return optim

    def parse(self, config: ConfigProvider) -> "SGDConfig":
        return config.get_dataclass(self)

@dataclass
class SAMConfig:
    forward: OptimizerConfig = OptimizerConfig.default()
    backward: OptimizerConfig = SGDConfig(lr=5e-2) # rho = 0.05
    start_percent: float = 0.
    run_percent: float = 1.
    disable_backward: bool = False
    normalize: bool = True

    def make_optimizer(self, iterations):
        import optax.contrib as sam

        forward_opt = self.forward.make_optimizer(iterations)
        if self.disable_backward:
            return forward_opt
        backward_opt = self.backward.make_optimizer(iterations)
        # normalize before the backward optimizer
        backward_opt = optax.chain(sam.normalize(), backward_opt) if self.normalize else backward_opt
        if self.start_percent > 0 or self.run_percent < 1.:
            start_iter = int(self.start_percent * iterations)
            end_iter = int((iterations - start_iter)*self.run_percent)
            backward_opt = optax.chain(
                backward_opt,
                optax.scale_by_schedule(
                    lambda i: jax.lax.cond(
                        jnp.logical_and(i < start_iter, i > end_iter), 
                    lambda: 0, lambda: 1))
            )
        return sam.sam(
            optimizer=forward_opt,
            adv_optimizer=backward_opt,
            reset_state=False,
            opaque_mode=True
        )


@dataclass
class TrainConfig:
    batch_size: int = 32
    """The batch size to use for training."""
    epochs: int | None = None
    """The number of epochs to train for."""
    iterations: int | None = None
    """The number of iterations to train for."""
    optimizer: OptimizerConfig = None
    
    def fit(self, **kwargs):
        from stanza.train import fit
        data = kwargs.pop("data")
        if self.epochs is None and self.iterations is None:
            raise ValueError("Either epochs or iterations must be specified")
        iterations = self.iterations or (self.epochs * (len(data) // self.batch_size))
        return fit(
            data=data,
            batch_size=self.batch_size,
            max_iterations=iterations,
            optimizer=self.optimizer.make_optimizer(iterations),
            **kwargs
        )
    
    def parse(self, config: ConfigProvider) -> "TrainConfig":
        res = config.get_dataclass(self, {"optimizer"})
        optimizer = config.get_cases("optimizer", "The optimizer to use", {
            "sgd": SGDConfig(),
            "adam": AdamConfig()
        }, "adam")
        return replace(res, optimizer=optimizer)