import jax
import jax.numpy as jnp
from typing import Generic, TypeVar, Protocol, Callable, Optional
from stanza.dataclasses import dataclass
from functools import partial

# A policy is a function from PolicyInput --> PolicyOutput

Observation = TypeVar("Observation")
PolicyState = TypeVar("PolicyState")
State = TypeVar("State")
Action = TypeVar("Action")
Info = TypeVar("Info")

Model = Callable[[State, Action, jax.Array], State]

@dataclass
class PolicyInput(Generic[Observation, State, PolicyState]):
    observation: Observation
    state: State = None
    policy_state: PolicyState = None
    rng_key : jax.Array = None

@dataclass
class PolicyOutput(Generic[Action, PolicyState, Info]):
    action: Action
    # The policy state
    policy_state: PolicyState = None
    info: Info = None

@dataclass
class Trajectory(Generic[State, Action]):
    states: State
    actions: Action = None

# Rollout contains the trajectory + aux data,
# final policy state, final rng key
@dataclass
class Rollout(Trajectory[State, Action], Generic[State, Action, Observation, Info]):
    observations: Observation = None
    info: Info = None
    final_policy_state: State = None
    final_policy_rng_key: jax.Array = None
    final_model_rng_key: jax.Array = None

class Policy(Protocol[Observation, Action, State, PolicyState, Info]):
    @property
    def rollout_length(self): ...

    def __call__(self, 
        input : PolicyInput[Observation, State, PolicyState]
    ) -> PolicyOutput[Action, PolicyState, Info]: ...

# stanza.jit can handle function arguments
# and intelligently makes them static and allows
# for vectorizing over functins.
@partial(jax.jit, static_argnames=("model", "policy", "observe", "length", "last_input"))
def rollout(model : Model, state0 : State,
            # policy is optional. If policy is not supplied
            # it is assumed that model is for an autonomous system
            policy : Optional[Policy] = None, *,
            # observation function
            # by default just uses the state
            observe : Optional[Callable[[State], Observation]] =  None,
            # The rng_key for the environment
            model_rng_key : Optional[jax.Array] = None,
            # The initial policy state.
            policy_init_state : Optional[PolicyState] = None,
            # The policy rng key
            policy_rng_key : Optional[jax.Array] = None,
            # Apply a transform to the
            # policy before rolling out.
            length : Optional[int] = None, last_input : bool = False) -> Rollout[State, Action, Observation, Info]:
    """ Rollout a model in-loop with policy for a fixed length.
    The length must either be provided via ``policy.rollout_length``
    or by the ``length`` parameter.

    If ``last_input`` is True, the policy is called again with the final
    state of the rollout to get a final action (so that the states and actions are the same length).
    """
    # Look for a fallback to the rollout length
    # in the policy. This is useful mainly for the Actions policy
    if length is None and hasattr(policy, 'rollout_length'):
        length = policy.rollout_length
    if length is None:
        raise ValueError("Rollout length must be specified")
    if length == 0:
        raise ValueError("Rollout length must be > 0")
    if last_input:
        length = length + 1
    if observe is None:
        observe = lambda x: x
    

    def scan_fn(comb_state, _):
        env_state, policy_state, policy_rng, model_rng = comb_state
        new_policy_rng, p_sk = jax.random.split(policy_rng) \
            if policy_rng is not None else (None, None)
        new_model_rng, m_sk = jax.random.split(model_rng) \
            if model_rng is not None else (None, None)
        obs = observe(env_state)
        if policy is not None:
            input = PolicyInput(obs, env_state, policy_state, p_sk)
            policy_output = policy(input)
            action = policy_output.action
            info = policy_output.info
            new_policy_state = policy_output.policy_state
        else:
            action = None
            info = None
            new_policy_state = policy_state
        new_env_state = model(env_state, action, m_sk)
        return (new_env_state, new_policy_state, new_policy_rng, new_model_rng),\
                (env_state, obs, action, info)

    # Do the first step manually to populate the policy state
    state = (state0, policy_init_state, policy_rng_key, model_rng_key)
    new_state, first_output = scan_fn(state, None)
    # outputs is (xs, us, jacs) or (xs, us)
    (state_f, policy_state_f, 
     policy_rng_f, model_rng_f), outputs = jax.lax.scan(scan_fn, new_state,
                                    None, length=length-2)
    outputs = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate((jnp.expand_dims(a,0), b)),
        first_output, outputs)

    states, observations, us, info = outputs
    if not last_input:
        states = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate((a, jnp.expand_dims(b, 0))),
            states, state_f)
        observations = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate((a, jnp.expand_dims(b, 0))),
            observations, observe(state_f))
    return Rollout(states=states, actions=us, observations=observations,
        info=info, final_policy_state=policy_state_f, 
        final_policy_rng_key=policy_rng_f, final_model_rng_key=model_rng_f)

# Shorthand alias for rollout with an actions policy
def rollout_inputs(model, state0, actions, last_state=True):
    return rollout(model, state0, policy=Actions(actions),
                   last_state=last_state)

# An "Actions" policy can be used to replay
# actions from a history buffer
@dataclass
class Actions(Generic[Action]):
    actions: Action

    @property
    def rollout_length(self):
        lengths, _ = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(lambda x: x.shape[0], self.actions)
        )
        return lengths[0] + 1

    @jax.jit
    def __call__(self, input : PolicyInput) -> PolicyOutput[Action, jax.Array, None]:
        T = input.policy_state if input.policy_state is not None else 0
        action = jax.tree_util.tree_map(lambda x: x[T], self.actions)
        return PolicyOutput(action=action, policy_state=T + 1)