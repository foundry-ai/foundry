#Implement the noisers
import jax
from jax.random import PRNGKey
from jax.tree_util import Partial  
from stanza.distribution.mvn import MultivariateNormalDiag




def make_gaussian_noiser(mean, scale_diag):
    return MultivariateNormalDiag(mean,scale_diag).sample