from stanza import struct, partial
from stanza.struct import args

import optax

@struct.dataclass
class OptimizerConfig:
    lr: float = 1e-4
    lr_schedule: str = "constant"
    warmup_schedule: str | None = None
    # if none, will use a "good" value based on warmup_schedule and the number of iterations 
    warmup_steps: int | None = None 

    def make_lr_schedule(self, iterations):
        if self.lr_schedule == "constant":
            return optax.constant_schedule(self.lr)
        elif self.lr_schedule == "linear":
            return optax.linear_schedule(0, self.lr, iterations)
        elif self.lr_schedule == "cosine":
            return optax.cosine_decay_schedule(self.lr, 0, iterations)
        else:
            raise ValueError(f"Unknown learning rate schedule: {self.lr_schedule}")

    def make_optimizer(self, iterations):
        raise NotImplementedError()

    @staticmethod
    def default():
        return AdamConfig(lr_schedule="cosine", warmup_schedule="linear")

@struct.dataclass
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def make_optimizer(self, iterations):
        return optax.adam(learning_rate=self.make_lr_schedule(iterations), 
                          b1=self.beta1, b2=self.beta2, eps=self.epsilon)

    @staticmethod
    def default():
        return AdamConfig(lr_schedule="cosine", warmup_schedule="linear")

@struct.dataclass
class SGDConfig(OptimizerConfig):
    lr_schedule: str = "constant" # The learning rate schedule
    momentum: float | None = None
    nesterov: bool = False

    def make_optimizer(self, iterations):
        return optax.sgd(learning_rate=self.make_lr_schedule(iterations),
                         momentum=self.momentum, nesterov=self.nesterov)

@struct.dataclass
class SAMConfig(OptimizerConfig):
    optimizer: OptimizerConfig = OptimizerConfig.default()
    adv_optimizer: OptimizerConfig = SGDConfig(lr=5e-2) # rho = 0.05
    reset_backward_state: bool = True
    normalize: bool = True

    def make_optimizer(self, iterations):
        import optax.contrib.sam as sam

        forward_opt = self.optimizer.make_optimizer(iterations)
        backward_opt = self.adv_optimizer.make_optimizer(1 if self.reset_adv_state else iterations)
        # normalize before the backward optimizer
        backward_opt = optax.chain(sam.normalize(), backward_opt) if self.normalize else backward_opt
        return sam.sam(
            optimizer=forward_opt,
            adv_optimizer=backward_opt,
            reset_state=self.reset_backward_state,
            opaque_mode=True
        )

@struct.dataclass
class TrainConfig:
    batch_size: int = 32
    """The batch size to use for training."""
    epochs: int | None = None
    """The number of epochs to train for."""
    iterations: int | None = None
    """The number of iterations to train for."""

    optimizer: OptimizerConfig = struct.field(
        default=OptimizerConfig.default()
    )
    
    def fit(self, **kwargs):
        from stanza.train import fit
        data = kwargs.pop("data")
        iterations = self.iterations or (self.epochs * (len(data) // self.batch_size))
        return fit(
            data=data,
            batch_size=self.batch_size,
            max_epochs=self.epochs,
            max_iterations=iterations,
            optimizer=self.optimizer.make_optimizer(iterations),
            **kwargs
        )