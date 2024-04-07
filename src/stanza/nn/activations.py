from jax.numpy import tanh

from jax.nn import celu
from jax.nn import elu
from jax.nn import gelu
from jax.nn import glu
from jax.nn import hard_sigmoid
from jax.nn import hard_silu
from jax.nn import hard_swish
from jax.nn import hard_tanh
from jax.nn import leaky_relu
from jax.nn import log_sigmoid
from jax.nn import log_softmax
from jax.nn import logsumexp
from jax.nn import one_hot
from jax.nn import relu
from jax.nn import relu6
from jax.nn import selu
from jax.nn import sigmoid
from jax.nn import silu
from jax.nn import soft_sign
from jax.nn import softmax
from jax.nn import softplus
from jax.nn import standardize
from jax.nn import swish

def mish(x):
    return x * tanh(softplus(x))