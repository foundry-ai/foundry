.. role:: python(code)
  :language: python
  :class: highlight

.. currentmodule:: stanza.policy

Policies and Rollouts
=====================

Stanza contains utilities for defining control policies
ontop of dynamical systems. These tools can be found in the
:py:mod:`stanza.policy` module.

Policies
^^^^^^^^

A policy is a function which takes as an argument 
a :py:class:`PolicyInput` and returns :py:class:`PolicyOutput`

.. autoclass:: Policy
    :no-index:
    :special-members: __call__
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: PolicyInput
    :no-index:
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: PolicyOutput
    :no-index:
    :members:
    :undoc-members:
    :show-inheritance:

A policy can optionally set a 
:py:attr:`PolicyOutput.policy_state` which will be
passed to the next call of the policy. Note that the policy state
must be a valid pytree and the tree shape cannot change.

For example we can define a simple feedback policy as follows:

.. code-block:: python

    def feedback_policy(input: PolicyInput) -> PolicyOutput:
        return PolicyOutput(-input.observation)

A policy can (optionally) have a :py:attr:`rollout_length <Policy.rollout_length>` property
which acts as a hint for how many steps the policy should be run.

For instance, here is a simplified :py:class:`Actions` policy
which will replay a given sequence of actions:

.. code-block:: python

    @dataclass
    class Actions(Policy):
        actions: Any

        def __call__(self, input: PolicyInput) -> PolicyOutput:
            t = input.policy_state if input.policy_state is not None else 0
            action = jax.tree_map(lambda x: x[t], self.actions)
            return PolicyOutput(
                action=action,
                policy_state=t+1
            )

        @property
        def rollout_length(self) -> int:
            return stanza.util.axis_size(self.actions, 0)

Rollouts
^^^^^^^^

The main api for calling a policy is the :py:func:`rollout` function.

.. autofunction:: rollout
    :no-index: